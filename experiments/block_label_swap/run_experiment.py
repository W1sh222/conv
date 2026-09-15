#!/usr/bin/env python3
"""Swap one selected prompt block at a time; measure teacher-forced label NLL.

Uses the repository's Transformers 4.51 adapter, Conv score estimator, and
block_sparse_attn_func. No model code is changed on disk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from swap_core import make_plan, summarize

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local Qwen3 or Llama model directory")
    p.add_argument("--data", required=True, help="JSONL with prompt/label or messages")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--layer", type=int, required=True, help="Zero-based attention layer")
    p.add_argument("--head", type=int, required=True, help="Zero-based QUERY head (not KV head)")
    p.add_argument("--query-block", default="last", help="Zero-based query block, or last")
    p.add_argument("--ratio", type=float, default=0.65)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--selector", choices=["initial", "conv"], default="initial")
    p.add_argument("--conv-weights", help="Required for --selector conv")
    p.add_argument("--background", choices=["sparse", "dense"], default="sparse",
                   help="sparse: all layers/heads use fixed masks; dense: only target head is sparse")
    p.add_argument("--device-map", default="auto", help="auto, balanced, or a single device e.g. cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--chat-template", action="store_true", help="Wrap prompt as user chat; messages always use template")
    p.add_argument("--rope-factor", type=float, default=None, help="Optional static YaRN scaling; no override by default")
    p.add_argument("--rope-original-length", type=int, default=32768)
    p.add_argument("--max-position-embeddings", type=int, default=None)
    p.add_argument("--max-prompt-tokens", type=int, default=0, help="Fail above limit, never silently truncate")
    p.add_argument("--max-label-tokens", type=int, default=0, help="Fail above limit, never silently truncate")
    p.add_argument("--loss-tolerance", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="New/empty output directory")
    p.add_argument("--no-plots", action="store_true")
    return p


def read_example(filename, index):
    if index < 0:
        raise ValueError("sample-index must be nonnegative")
    with open(filename, encoding="utf-8") as f:
        for n, line in enumerate(x for x in f if x.strip()):
            if n == index:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("Each JSONL record must be an object")
                return item
    raise IndexError(f"No sample {index} in {filename}")


def prepare_tokens(tokenizer, example, args):
    messages = example.get("messages")
    label = example.get("label", example.get("answer", example.get("output")))
    if messages is not None:
        messages = list(messages)
        if messages and messages[-1].get("role") == "assistant":
            assistant = messages.pop()["content"]
            if label is None:
                label = assistant
        if not messages or messages[-1].get("role") == "assistant":
            raise ValueError("messages must contain a prompt ending before the target answer")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        add_special = False
    else:
        prompt = example.get("prompt", example.get("input"))
        if not isinstance(prompt, str):
            raise ValueError("Provide prompt/input string and label/answer/output string")
        add_special = True
        if args.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
            add_special = False
    if not isinstance(label, str) or not label.strip():
        raise ValueError("A nonempty string ground-truth label is required (choose one reference explicitly)")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    # Deliberate generation boundary: target continuation IDs are appended to
    # prompt IDs; tokenizing prompt+label jointly can change the prompt suffix.
    label_ids = tokenizer.encode(label, add_special_tokens=False)
    if not prompt_ids or not label_ids:
        raise ValueError("Empty tokenized prompt or label")
    for count, limit, field in [(len(prompt_ids), args.max_prompt_tokens, "prompt"),
                                (len(label_ids), args.max_label_tokens, "label")]:
        if limit and count > limit:
            raise ValueError(f"{field} has {count} tokens, above limit {limit}; no truncation performed")
    return prompt, label, prompt_ids, label_ids


class MaskController:
    def __init__(self, args, prompt_length, torch, adapter, conv, score_fn):
        self.args, self.torch, self.adapter, self.conv = args, torch, adapter, conv
        self.score_fn = score_fn
        self.prompt_length = prompt_length
        self.nblocks = math.ceil(prompt_length / 128)
        self.recording = True
        self.frozen_masks = {}  # CPU, so one mask per layer does not fill GPU memory
        self.initial_scores = None
        self.plan = None
        self.replacement = None
        self.prefill_calls = 0

    def prefill(self, attn, q, k, v, attention_mask):
        torch, args = self.torch, self.args
        layer = attn.layer_idx
        if q.shape[-2] != self.prompt_length:
            raise RuntimeError("Only the unchanged prompt may enter sparse prefill")
        self.prefill_calls += 1
        if args.background == "dense" and layer != args.layer:
            return self.adapter._dense_attention(q, k, v, attention_mask)
        if self.recording:
            initial, meta = self.score_fn(q, k, block_size=128, stride=args.stride,
                                         norm=1.0, chunk_size=0, causal=True)
            ranking = initial
            if args.selector == "conv":
                ranking = self.conv.apply_conv2d_block_map(initial, kernel_size=7,
                            weight_path=args.conv_weights, layer_idx=layer)
            # Same fixed-ratio selector as Conv.py. Offset zero for whole prefill.
            full_mask = self.conv._topk_ratio_mask_from_scores(ranking, args.ratio, offset=0, causal=True)
            mask = self.conv._sanitize_block_sparse_mask(full_mask, self.nblocks, self.nblocks,
                                                        causal=True, keep_sink=False, keep_recent=False)
            if args.background == "dense":
                target = mask[:, args.head].clone()
                causal = torch.ones(self.nblocks, self.nblocks, device=q.device, dtype=torch.bool).tril()
                mask = causal[None, None].expand(1, q.shape[1], -1, -1).clone()
                mask[:, args.head] = target
            self.frozen_masks[layer] = mask.detach().cpu()
            if layer == args.layer:
                self.initial_scores = initial[0, args.head, :self.nblocks, :self.nblocks].float().cpu().numpy().copy()
                qb = self.nblocks - 1 if args.query_block == "last" else int(args.query_block)
                self.plan = make_plan(self.initial_scores, mask[0, args.head].cpu().numpy(), qb)
            del initial, ranking, full_mask
        else:
            mask = self.frozen_masks[layer].to(device=q.device).clone()
        if layer == args.layer and self.replacement is not None:
            qb, removed = self.plan["query_block"], self.plan["removed_key_block"]
            candidate = self.replacement
            if candidate not in self.plan["candidate_key_blocks"]:
                raise RuntimeError("Invalid replacement")
            if not mask[0, args.head, qb, removed].item() or mask[0, args.head, qb, candidate].item():
                raise RuntimeError("Baseline mask mutated across trials")
            mask[0, args.head, qb, removed] = False
            mask[0, args.head, qb, candidate] = True
        length, heads, dim = q.shape[-2], q.shape[1], q.shape[-1]
        cuq, cuk, kinds = self.conv._get_static_prefill_tensors(length, length, heads, q.device)
        # Use precisely the deployed sparse attention backend, not dense-matrix
        # zeroing without softmax renormalization.
        out = self.conv.block_sparse_attn_func(
            q.transpose(1, 2).reshape(length, heads, dim),
            k.transpose(1, 2).reshape(length, heads, dim),
            v.transpose(1, 2).reshape(length, heads, dim),
            cuq, cuk, kinds, None, mask.contiguous(), length, length,
            p_dropout=0.0, deterministic=True, is_causal=True)
        return out.view(1, length, heads, dim).transpose(1, 2)


def label_nll(model, prompt_ids, label_ids, torch):
    """Prompt-only prefill, then teacher-forced normal decode. Fresh KV each trial."""
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    token_losses, predictions = [], []
    cache = None
    with torch.inference_mode():
        for step, target_id in enumerate(label_ids):
            result = model.model(input_ids=ids, past_key_values=cache, use_cache=True,
                                 return_dict=True)
            cache = result.past_key_values
            # Only compute the vocabulary projection for the next-token position.
            last = result.last_hidden_state[:, -1, :].to(model.lm_head.weight.device)
            logits = model.lm_head(last).float()
            target = torch.tensor([target_id], device=logits.device, dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite label loss")
            token_losses.append(float(loss.item()))
            predictions.append(int(logits.argmax(-1).item()))
            ids = target.to(device).view(1, 1)  # correct previous label, not argmax
            del result, last, logits, loss
    del cache
    return {"label_loss": float(np.mean(token_losses)), "token_losses": token_losses,
            "teacher_forced_argmax_ids": predictions}


def main():
    args = parser().parse_args()
    if not 0 < args.ratio < 1:
        raise ValueError("ratio must be in (0,1) to leave replacement candidates")
    if args.stride <= 0 or 128 % args.stride:
        raise ValueError("stride must be a positive divisor of block size 128")
    if args.loss_tolerance < 0:
        raise ValueError("loss-tolerance must be nonnegative")
    if args.selector == "conv" and not args.conv_weights:
        raise ValueError("--selector conv requires --conv-weights; no implicit averaging kernel")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new/empty output directory; previous results are not overwritten")
    # Fail on missing plotting dependencies before expensive model evaluation.
    if not args.no_plots:
        import matplotlib  # noqa: F401
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA plus the repository's Triton/block_sparse_attn environment is required")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ft_scripts" / "sparse_ruler"))
    from xattn.src import load_transformers_451 as adapter
    from xattn.src import Conv as conv
    from conv_sparse_ops import kernels_conv_block_scores_infer_full
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type not in {"llama", "qwen3"}:
        raise ValueError("Only Llama/Qwen3 Transformers 4.51 interfaces are supported")
    if getattr(config, "use_sliding_window", False):
        raise ValueError("Sliding-window attention is not supported by this experiment")
    if not 0 <= args.layer < config.num_hidden_layers or not 0 <= args.head < config.num_attention_heads:
        raise ValueError("layer/head index outside model layout")
    if args.rope_factor is not None:
        if args.rope_factor <= 0:
            raise ValueError("rope-factor must be positive")
        config.rope_scaling = {"rope_type": "yarn", "factor": args.rope_factor,
                              "original_max_position_embeddings": args.rope_original_length}
        config.max_position_embeddings = math.ceil(args.rope_original_length * args.rope_factor)
    if args.max_position_embeddings:
        config.max_position_embeddings = args.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    example = read_example(args.data, args.sample_index)
    prompt, label, prompt_ids, label_ids = prepare_tokens(tokenizer, example, args)
    if len(prompt_ids) + len(label_ids) > config.max_position_embeddings:
        raise ValueError("Prompt+label exceeds configured context; set correct RoPE options explicitly")
    controller = MaskController(args, len(prompt_ids), torch, adapter, conv,
                                kernels_conv_block_scores_infer_full)
    dm = args.device_map if args.device_map in {"auto", "balanced", "balanced_low_0", "sequential"} else {"": args.device_map}
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config,
              torch_dtype=getattr(torch, args.dtype), device_map=dm, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    # SDPA lets Transformers avoid allocating the global N*N additive causal mask.
    fast_config = adapter.BaseFastPrefillConfig(metric="conv", stride=args.stride,
                          block_topk_ratio=args.ratio, print_detail=False)
    original_prefill = adapter._run_prefill
    originals = []
    adapter._run_prefill = controller.prefill
    for layer in model.model.layers:
        attn = layer.self_attn
        originals.append((attn, attn.forward, getattr(attn, "fastprefillconfig", None)))
        attn.fastprefillconfig = fast_config
        attn.forward = adapter.forward_eval_451.__get__(attn, type(attn))
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        print(f"Prompt={len(prompt_ids)} label={len(label_ids)} tokens; baseline recording", flush=True)
        baseline = label_nll(model, prompt_ids, label_ids, torch)
        controller.recording = False
        plan = controller.plan
        if plan is None:
            raise RuntimeError("Target layer was not visited")
        candidate_list = plan["candidate_key_blocks"]
        target_mask = controller.frozen_masks[args.layer][0, args.head].numpy().copy()
        np.savez_compressed(out / "block_map.npz", initial_scores=controller.initial_scores,
                            selected_mask=target_mask)
        torch.save(controller.frozen_masks, out / "frozen_masks.pt")
        meta = {"arguments": vars(args), "plan": plan, "baseline": baseline,
                "prompt_token_count": len(prompt_ids), "label_token_count": len(label_ids),
                "label": label, "label_ids": label_ids,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "model_type": config.model_type, "transformers_version": transformers.__version__,
                "torch_version": torch.__version__, "rope_scaling": config.rope_scaling,
                "block_size": 128, "status": "running",
                "loss_definition": "Mean teacher-forced NLL of label tokens only; no EOS added",
                "selection_inputs": "Prompt only; no label tokens used for score estimation",
                "frozen_selection": "All masks fixed from baseline, including downstream layers",
                "scope": "Single layer/query-head/query-block row, one-for-one swaps"}
        (out / "experiment.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = []
        with (out / "trials.jsonl").open("w", encoding="utf-8") as log:
            for index, candidate in enumerate(candidate_list):
                controller.replacement = candidate
                result = label_nll(model, prompt_ids, label_ids, torch)
                row = {"key_block": candidate,
                       "initial_score": float(controller.initial_scores[plan["query_block"], candidate]),
                       "removed_initial_score": plan["removed_initial_score"],
                       "baseline_loss": baseline["label_loss"],
                       "delta_loss": result["label_loss"] - baseline["label_loss"], **result}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"[{index+1}/{len(candidate_list)}] {plan['removed_key_block']} -> {candidate}: "
                      f"loss={row['label_loss']:.9f}, delta={row['delta_loss']:+.9f}", flush=True)
        controller.replacement = None
        repeated = label_nll(model, prompt_ids, label_ids, torch)
        summary = summarize(baseline["label_loss"], repeated["label_loss"], rows,
                            plan["removed_initial_score"], args.loss_tolerance)
        meta.update(status="complete", summary=summary, repeated_baseline=repeated,
                    elapsed_seconds=time.time()-start)
        (out / "experiment.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        fields = ["key_block", "initial_score", "removed_initial_score", "baseline_loss", "label_loss", "delta_loss"]
        with (out / "trials.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        if not args.no_plots:
            from plot_results import plot_results
            plot_results(out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        adapter._run_prefill = original_prefill
        for attn, forward, old_config in originals:
            attn.forward = forward
            if old_config is None:
                delattr(attn, "fastprefillconfig")
            else:
                attn.fastprefillconfig = old_config


if __name__ == "__main__":
    main()
