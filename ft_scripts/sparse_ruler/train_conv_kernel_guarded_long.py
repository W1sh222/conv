#!/usr/bin/env python3
"""Low-memory long-context guarded training for the original sparse Conv kernel.

The inference-matched no-spaced Conv score map and trainable 7x7 convolution
are retained.  Quadratic dense teacher/output tensors are replaced by exact
attention mass for a small set of query tokens, computed with online key
chunks.  Only selected layer inputs are captured from the frozen model.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path.cwd()))

from conv_sparse_ops import (  # type: ignore  # noqa: E402
    apply_conv_energy,
    kernels_conv_block_scores_infer_full,
    make_causal_block_mask,
    make_inference_chunked_block_mask,
    repeat_kv_to_q_heads,
)
from train_conv_kernel_guarded import (  # type: ignore  # noqa: E402
    BoundedResidualConvKernel,
    EMA,
    aggregation_loss,
    current_budget,
    effective_budget_loss,
    entropy_loss,
    parse_target_blocks_schedule,
    ranking_loss,
    soft_effective_blocks,
)


class _StopFrozenForward(Exception):
    pass


def format_prompt_only(example: Dict[str, Any], tokenizer) -> str:
    """Match evaluation: no assistant answer and no synthetic marker tokens."""
    if example.get("messages") is not None:
        messages = list(example["messages"])
        if messages and str(messages[-1].get("role", "")).lower() == "assistant":
            messages = messages[:-1]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif "text" in example:
        text = str(example["text"])
    else:
        text = json.dumps(example, ensure_ascii=False)

    meta = example.get("meta", {}) or {}
    for key in ("target_record_spans", "aggregation_spans"):
        for span in meta.get(key, []) or []:
            for marker_key in ("start_marker", "end_marker"):
                marker = span.get(marker_key)
                if marker:
                    marker = str(marker)
                    text = text.replace(marker, " " * len(marker))
    return text


def tokenize_one(tokenizer, text: str, device: torch.device):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        padding=False,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device, non_blocking=True)
    attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
    offsets = encoded["offset_mapping"][0].tolist()
    position_ids = torch.arange(
        input_ids.shape[1],
        device=device,
        dtype=torch.long,
    )[None]
    return input_ids, attention_mask, position_ids, offsets


def char_span_to_blocks(
    offsets: Sequence[Sequence[int]],
    char_span: Tuple[int, int],
    block_size: int,
    seq_len: int,
) -> Optional[List[int]]:
    start_char, end_char = char_span
    tokens = [
        idx
        for idx, (start, end) in enumerate(offsets[:seq_len])
        if end > start_char and start < end_char
    ]
    if not tokens:
        return None
    return list(range(tokens[0] // block_size, tokens[-1] // block_size + 1))


def resolve_clean_groups(
    text: str,
    offsets: Sequence[Sequence[int]],
    meta: Dict[str, Any],
    key: str,
    block_size: int,
    seq_len: int,
) -> List[List[int]]:
    groups: List[List[int]] = []
    cursors: Dict[str, int] = {}
    for span in meta.get(key, []) or []:
        evidence = str(span.get("text", "")).strip()
        if not evidence:
            continue
        start = text.find(evidence, cursors.get(evidence, 0))
        if start < 0:
            start = text.find(evidence)
        if start < 0:
            continue
        cursors[evidence] = start + len(evidence)
        blocks = char_span_to_blocks(
            offsets,
            (start, start + len(evidence)),
            block_size,
            seq_len,
        )
        if blocks:
            groups.append(blocks)
    return groups


@torch.inference_mode()
def capture_layer_inputs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """Capture only requested layer inputs and stop after the highest layer."""
    wanted = sorted({int(x) for x in layer_indices})
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    final_layer = wanted[-1]

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            captured[layer_idx] = inputs[0].detach()
            if layer_idx == final_layer:
                raise _StopFrozenForward()
        return hook

    for layer_idx in wanted:
        handles.append(
            model.model.layers[layer_idx].register_forward_pre_hook(
                make_hook(layer_idx)
            )
        )
    try:
        model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    except _StopFrozenForward:
        pass
    finally:
        for handle in handles:
            handle.remove()
    missing = [idx for idx in wanted if idx not in captured]
    if missing:
        raise RuntimeError(f"failed to capture layer inputs: {missing}")
    return captured


@torch.inference_mode()
def extract_qk(
    model,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    attention = model.model.layers[layer_idx].self_attn
    batch, seq_len, _ = hidden_states.shape
    q_heads = getattr(
        attention,
        "num_heads",
        model.config.num_attention_heads,
    )
    kv_heads = getattr(
        attention,
        "num_key_value_heads",
        model.config.num_key_value_heads,
    )
    head_dim = getattr(
        attention,
        "head_dim",
        model.config.hidden_size // q_heads,
    )
    q = attention.q_proj(hidden_states)
    k = attention.k_proj(hidden_states)
    q = q.view(batch, seq_len, q_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    cos, sin = model.model.rotary_emb(hidden_states, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q.contiguous(), k.contiguous()


def choose_query_rows(
    q_blocks: int,
    row_count: int,
    tail_count: int,
    rng: random.Random,
) -> List[int]:
    tail_count = min(q_blocks, max(1, int(tail_count)))
    chosen = set(range(q_blocks - tail_count, q_blocks))
    remaining = max(0, int(row_count) - len(chosen))
    candidates = list(range(0, q_blocks - tail_count))
    for slot in range(remaining):
        if not candidates:
            break
        lo = slot * len(candidates) // remaining
        hi = max(lo + 1, (slot + 1) * len(candidates) // remaining)
        chosen.add(candidates[rng.randrange(lo, min(hi, len(candidates)))])
    return sorted(chosen)


def query_token_grid(
    rows: Sequence[int],
    seq_len: int,
    block_size: int,
    tokens_per_row: int,
    device: torch.device,
) -> torch.Tensor:
    output = []
    for row in rows:
        start = row * block_size
        end = min(seq_len, start + block_size)
        if tokens_per_row == 1:
            positions = [end - 1]
        else:
            positions = [
                int(round(start + idx * (end - 1 - start) / (tokens_per_row - 1)))
                for idx in range(tokens_per_row)
            ]
        output.append(positions)
    return torch.tensor(output, device=device, dtype=torch.long)


@torch.inference_mode()
def exact_teacher_block_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    rows: Sequence[int],
    *,
    block_size: int,
    tokens_per_row: int,
    head_chunk: int,
    key_chunk: int,
    norm: float,
) -> torch.Tensor:
    """Exact causal attention mass without allocating an N by N matrix."""
    if key_chunk % block_size:
        raise ValueError("teacher_key_chunk must be divisible by block_size")
    _, q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    kv_repeat = q_heads // kv_heads
    key_blocks = math.ceil(seq_len / block_size)
    grid = query_token_grid(
        rows,
        seq_len,
        block_size,
        tokens_per_row,
        q.device,
    )
    flat_tokens = grid.flatten()
    sampled_tokens = flat_tokens.numel()
    output = torch.empty(
        1,
        q_heads,
        len(rows),
        key_blocks,
        device=q.device,
        dtype=torch.float32,
    )
    scale = 1.0 / (math.sqrt(head_dim) * float(norm))

    for head_start in range(0, q_heads, head_chunk):
        head_end = min(q_heads, head_start + head_chunk)
        q_chunk = q[0, head_start:head_end].index_select(1, flat_tokens)
        q_ids = torch.arange(head_start, head_end, device=q.device)
        kv_ids = torch.div(q_ids, kv_repeat, rounding_mode="floor")
        k_heads = k[0].index_select(0, kv_ids)
        chunk_heads = head_end - head_start
        running_max = torch.full(
            (chunk_heads, sampled_tokens),
            -torch.inf,
            device=q.device,
            dtype=torch.float32,
        )
        denominator = torch.zeros_like(running_max)
        accumulator = torch.zeros(
            chunk_heads,
            sampled_tokens,
            key_blocks,
            device=q.device,
            dtype=torch.float32,
        )

        for key_start in range(0, seq_len, key_chunk):
            key_end = min(seq_len, key_start + key_chunk)
            logits = torch.matmul(
                q_chunk,
                k_heads[:, key_start:key_end].transpose(-1, -2),
            ).float().mul_(scale)
            key_positions = torch.arange(key_start, key_end, device=q.device)
            visible = key_positions[None, :] <= flat_tokens[:, None]
            logits.masked_fill_(~visible[None], -torch.inf)
            local_max = logits.amax(-1)
            new_max = torch.maximum(running_max, local_max)
            old_scale = torch.exp(running_max - new_max)
            exp_logits = torch.exp(logits - new_max[:, :, None])
            exp_logits.masked_fill_(~visible[None], 0.0)
            denominator.mul_(old_scale).add_(exp_logits.sum(-1))
            accumulator.mul_(old_scale[:, :, None])
            padded = math.ceil((key_end - key_start) / block_size) * block_size
            if padded != key_end - key_start:
                exp_logits = F.pad(
                    exp_logits,
                    (0, padded - (key_end - key_start)),
                )
            chunk_mass = exp_logits.view(
                chunk_heads,
                sampled_tokens,
                padded // block_size,
                block_size,
            ).sum(-1)
            block_start = key_start // block_size
            accumulator[
                :,
                :,
                block_start:block_start + chunk_mass.shape[-1],
            ].add_(chunk_mass)
            running_max = new_max

        token_mass = accumulator / denominator.clamp_min(1.0e-20)[..., None]
        output[0, head_start:head_end].copy_(
            token_mass.view(
                chunk_heads,
                len(rows),
                tokens_per_row,
                key_blocks,
            ).mean(2)
        )
    return output


def teacher_distribution_loss(
    energy: torch.Tensor,
    teacher: torch.Tensor,
    rows: Sequence[int],
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_idx = torch.tensor(rows, device=energy.device, dtype=torch.long)
    selected = energy.index_select(2, row_idx).float()
    positive = F.softplus(selected / float(temperature)) * float(temperature)
    key_idx = torch.arange(energy.shape[-1], device=energy.device)
    causal = key_idx[None, :] <= row_idx[:, None]
    positive.masked_fill_(~causal[None, None], 0.0)
    predicted = positive / positive.sum(-1, keepdim=True).clamp_min(1.0e-8)
    teacher_copy = teacher.clone()
    kl = (
        teacher_copy
        * (
            teacher_copy.clamp_min(1.0e-8).log()
            - predicted.clamp_min(1.0e-8).log()
        )
    ).sum(-1).mean()
    l1 = F.smooth_l1_loss(predicted, teacher_copy, beta=0.01)
    return kl, l1, predicted


def topk_boundary_recall_loss(
    energy: torch.Tensor,
    teacher: torch.Tensor,
    rows: Sequence[int],
    *,
    topk_ratio: float,
    positive_mass: float,
    boundary_negatives: int,
    margin: float,
) -> torch.Tensor:
    """Push teacher-important blocks above the inference top-k boundary."""
    row_idx = torch.tensor(rows, device=energy.device, dtype=torch.long)
    selected = energy.index_select(2, row_idx).float()  # [B,H,R,K]
    teacher_mass = teacher.detach().float()
    key_idx = torch.arange(energy.shape[-1], device=energy.device)
    causal = key_idx[None, :] <= row_idx[:, None]  # [R,K]
    causal_bhrk = causal[None, None]

    teacher_mass = teacher_mass.masked_fill(~causal_bhrk, 0.0)
    teacher_mass = teacher_mass / teacher_mass.sum(-1, keepdim=True).clamp_min(
        1.0e-20
    )

    # Select the smallest teacher-ranked set that carries positive_mass.
    sorted_mass, sorted_idx = teacher_mass.sort(dim=-1, descending=True)
    cumulative = sorted_mass.cumsum(dim=-1)
    positive_count = (cumulative < float(positive_mass)).sum(-1) + 1
    visible_count = (row_idx + 1)[None, None, :]
    keep_count = torch.ceil(
        visible_count.float() * float(topk_ratio)
    ).long().clamp_min(1)
    positive_count = torch.minimum(
        positive_count.clamp_min(1),
        keep_count,
    )
    rank = torch.arange(energy.shape[-1], device=energy.device)
    sorted_positive = rank[None, None, None, :] < positive_count[..., None]
    positive_mask = torch.zeros_like(sorted_positive)
    positive_mask.scatter_(-1, sorted_idx, sorted_positive)
    positive_mask &= causal_bhrk

    # Mine negatives nearest the current inference keep/drop edge. Index
    # selection is discrete, while gathered positive/negative scores retain
    # gradients.
    masked_scores = selected.masked_fill(~causal_bhrk, -torch.inf)
    sorted_scores = masked_scores.sort(dim=-1, descending=True).values
    boundary_index = (keep_count - 1).expand(
        selected.shape[0], selected.shape[1], -1
    )
    boundary = sorted_scores.gather(-1, boundary_index[..., None]).squeeze(-1)

    negative_mask = causal_bhrk & ~positive_mask
    distance = (selected - boundary.detach()[..., None]).abs()
    distance = distance.masked_fill(~negative_mask, torch.inf)
    mined_count = min(int(boundary_negatives), energy.shape[-1])
    mined_idx = distance.topk(mined_count, dim=-1, largest=False).indices
    mined_negative = selected.gather(-1, mined_idx)
    mined_valid = negative_mask.gather(-1, mined_idx)

    hinge = F.relu(
        float(margin)
        - selected[..., :, None]
        + mined_negative[..., None, :]
    )
    valid_pair = positive_mask[..., :, None] & mined_valid[..., None, :]
    negative_denom = mined_valid.sum(-1).clamp_min(1)[..., None]
    per_positive = (hinge * valid_pair.float()).sum(-1) / negative_denom
    positive_weight = teacher_mass * positive_mask.float()
    positive_weight = positive_weight / positive_weight.sum(
        -1, keepdim=True
    ).clamp_min(1.0e-20)
    per_row = (per_positive * positive_weight).sum(-1)
    valid_row = mined_valid.any(-1) & positive_mask.any(-1)
    return (per_row * valid_row.float()).sum() / valid_row.float().sum().clamp_min(
        1.0
    )


def target_coverage(
    hard_mask: torch.Tensor,
    groups: List[List[int]],
    tail_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not groups:
        one = hard_mask.new_tensor(1.0, dtype=torch.float32)
        return one, one
    tail = hard_mask[:, :, -max(1, int(tail_blocks)):, :]
    hits = []
    for group in groups:
        valid = [idx for idx in group if 0 <= idx < hard_mask.shape[-1]]
        if valid:
            indices = torch.tensor(valid, device=hard_mask.device)
            hits.append(tail.index_select(-1, indices).any(-1).any(-1))
    if not hits:
        one = hard_mask.new_tensor(1.0, dtype=torch.float32)
        return one, one
    group_hits = torch.stack(hits)  # [G,B,H]
    mean_coverage = group_hits.float().mean()
    same_head_all = group_hits.all(0).any(-1).float().mean()
    return mean_coverage, same_head_all


def balanced_layers(
    queue: List[int],
    num_layers: int,
    count: int,
    rng: random.Random,
) -> List[int]:
    while len(queue) < count:
        refill = list(range(num_layers))
        rng.shuffle(refill)
        queue.extend(refill)
    chosen = queue[:count]
    del queue[:count]
    return sorted(chosen)


def load_frozen_model(model_path: str, precision: str):
    precision = precision.lower()
    quantization_config = None
    dtype = torch.bfloat16
    if precision == "nf4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif precision == "fp16":
        dtype = torch.float16
    elif precision != "bf16":
        raise ValueError(f"unsupported model_precision={precision}")
    attention_impl = (
        "flash_attention_2"
        if importlib.util.find_spec("flash_attn") is not None
        else "sdpa"
    )
    print(f"[model] precision={precision} compute_dtype={dtype}")
    print(f"[model] attention_implementation={attention_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation=attention_impl,
    )
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def save_state(path, step, conv, optimizer, ema):
    state_path = path.replace(".pt", "_train_state.pt")
    torch.save(
        {
            "step": int(step),
            "anchor": conv.anchor.detach().float().cpu(),
            "delta": conv.delta.detach().float().cpu(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.shadow.detach().float().cpu(),
            "weight_min": conv.weight_min,
            "weight_max": conv.weight_max,
            "bounded_delta_alpha": conv.alpha,
        },
        state_path,
    )
    return state_path


def learning_rate_for_step(
    step: int,
    total_steps: int,
    base_lr: float,
    schedule: str,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """Resume-safe learning rate determined only by the global step."""
    if schedule == "constant":
        return base_lr
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * max(1, step) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init_path", required=True)
    parser.add_argument("--resume_state", default="")
    parser.add_argument(
        "--model_precision",
        choices=["nf4", "bf16", "fp16"],
        default="nf4",
    )
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--layers_per_sample", type=int, default=1)
    parser.add_argument("--min_seq_length", type=int, default=24576)
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "cosine"],
        default="constant",
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=24032)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--block_topk_ratio",
        type=float,
        default=None,
        help=(
            "Keep this fraction of each query row's causal-visible blocks. "
            "When set, this overrides threshold selection exactly as Conv.py."
        ),
    )
    parser.add_argument("--score_stride", type=int, default=16)
    parser.add_argument("--score_chunk_size", type=int, default=0)
    parser.add_argument("--score_sample_kernel_size", type=int, default=7)
    parser.add_argument("--score_norm", type=float, default=1.0)
    parser.add_argument("--teacher_rows", type=int, default=8)
    parser.add_argument("--teacher_tail_rows", type=int, default=4)
    parser.add_argument("--teacher_tokens_per_row", type=int, default=4)
    parser.add_argument("--teacher_head_chunk", type=int, default=4)
    parser.add_argument("--teacher_key_chunk", type=int, default=2048)
    parser.add_argument("--positive_temperature", type=float, default=0.01)
    parser.add_argument("--teacher_kl_weight", type=float, default=0.55)
    parser.add_argument("--teacher_l1_weight", type=float, default=0.15)
    parser.add_argument("--topk_recall_loss_weight", type=float, default=0.0)
    parser.add_argument("--topk_train_ratio", type=float, default=None)
    parser.add_argument("--topk_positive_mass", type=float, default=0.95)
    parser.add_argument("--topk_boundary_negatives", type=int, default=16)
    parser.add_argument("--topk_boundary_margin", type=float, default=0.08)
    parser.add_argument("--target_loss_weight", type=float, default=0.35)
    parser.add_argument("--aggregation_loss_weight", type=float, default=0.12)
    parser.add_argument("--aggregation_cover_mass", type=float, default=0.28)
    parser.add_argument("--target_margin", type=float, default=0.10)
    parser.add_argument("--target_joint_weight", type=float, default=0.90)
    parser.add_argument("--target_query_tail_blocks", type=int, default=16)
    parser.add_argument("--negative_weight", type=float, default=0.02)
    parser.add_argument("--compression_loss_weight", type=float, default=0.0001)
    parser.add_argument("--compression_loss_weight_final", type=float, default=0.0005)
    parser.add_argument("--budget_loss_weight", type=float, default=0.001)
    parser.add_argument(
        "--target_blocks_schedule",
        default="0:96,4000:88,8000:80",
    )
    parser.add_argument("--bounded_delta_alpha", type=float, default=0.06)
    parser.add_argument("--weight_min", type=float, default=None)
    parser.add_argument("--weight_max", type=float, default=None)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--max_grad_norm", type=float, default=0.04)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=250)
    args = parser.parse_args()

    if args.block_size != 128:
        raise ValueError("inference requires block_size=128")
    if args.block_topk_ratio is not None and not (
        0.0 < args.block_topk_ratio <= 1.0
    ):
        raise ValueError("block_topk_ratio must be in (0,1]")
    if args.teacher_key_chunk % args.block_size:
        raise ValueError("teacher_key_chunk must be divisible by block_size")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("warmup_steps must be in [0, steps)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0,1]")
    if (args.weight_min is None) != (args.weight_max is None):
        raise ValueError("weight_min and weight_max must be set together")
    if (
        args.weight_min is not None
        and args.weight_max is not None
        and args.weight_min >= args.weight_max
    ):
        raise ValueError("weight_min must be smaller than weight_max")
    if args.topk_recall_loss_weight < 0.0:
        raise ValueError("topk_recall_loss_weight must be non-negative")
    if args.topk_recall_loss_weight > 0.0 and args.topk_train_ratio is None:
        raise ValueError(
            "topk_train_ratio is required when topk_recall_loss_weight > 0"
        )
    if args.topk_train_ratio is not None and not (
        0.0 < args.topk_train_ratio <= 1.0
    ):
        raise ValueError("topk_train_ratio must be in (0,1]")
    if not 0.0 < args.topk_positive_mass <= 1.0:
        raise ValueError("topk_positive_mass must be in (0,1]")
    if args.topk_boundary_negatives <= 0:
        raise ValueError("topk_boundary_negatives must be positive")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    print("=" * 80)
    print(
        "Sparse guarded long-context training "
        f"({args.min_seq_length}-{args.max_seq_length} tokens)"
    )
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    if args.weight_min is not None and args.weight_max is not None:
        print(
            "kernel_parameterization: absolute_tanh "
            f"range=({args.weight_min},{args.weight_max}); "
            "bounded_delta_alpha is ignored"
        )
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"dataset size: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("empty dataset")
    model = load_frozen_model(args.model, args.model_precision)

    conv = BoundedResidualConvKernel(
        init_path=args.init_path,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kernel_size=args.kernel_size,
        alpha=args.bounded_delta_alpha,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
    ).to(device)
    optimizer = torch.optim.AdamW([conv.delta], lr=args.lr, weight_decay=0.0)
    ema = EMA(conv, decay=args.ema_decay)
    start_step = 1
    if args.resume_state:
        state = torch.load(
            args.resume_state,
            map_location="cpu",
            weights_only=False,
        )
        if (
            state.get("weight_min") != conv.weight_min
            or state.get("weight_max") != conv.weight_max
        ):
            raise ValueError(
                "resume-state weight bounds do not match this run: "
                f"state=({state.get('weight_min')},{state.get('weight_max')}) "
                f"args=({conv.weight_min},{conv.weight_max})"
            )
        with torch.no_grad():
            conv.anchor.copy_(state["anchor"].to(device))
            conv.delta.copy_(state["delta"].to(device))
            ema.shadow.copy_(state["ema"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        print(f"[resume] {args.resume_state} next_step={start_step}")
    else:
        # Preserve the exact pre-training selector baseline. For
        # INIT_PATH=identity this is the no-convolution checkpoint and provides
        # a guaranteed comparison/fallback when learned smoothing is harmful.
        step0_path = args.out.replace(".pt", "_step0.pt")
        conv.save(step0_path)
        ema.save(args.out.replace(".pt", "_ema_step0.pt"))
        print(f"[save initial] raw={step0_path}")
        print(f"[save initial] ema={args.out.replace('.pt', '_ema_step0.pt')}")

    schedule = parse_target_blocks_schedule(args.target_blocks_schedule)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    data_ptr = 0
    layer_queue: List[int] = []
    running: Dict[str, float] = {}
    running_count = 0
    skipped_length = 0

    for step in range(start_step, args.steps + 1):
        step_lr = learning_rate_for_step(
            step=step,
            total_steps=args.steps,
            base_lr=args.lr,
            schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = step_lr
        completed = False
        attempts_for_step = 0
        while not completed:
            attempts_for_step += 1
            if attempts_for_step > len(dataset):
                raise RuntimeError(
                    "No dataset sample remained inside the requested "
                    f"{args.min_seq_length}-{args.max_seq_length} prompt-only "
                    "range after removing markers. Rebuild the dataset."
                )
            if data_ptr >= len(indices):
                rng.shuffle(indices)
                data_ptr = 0
            example = dataset[indices[data_ptr]]
            data_ptr += 1
            text = format_prompt_only(example, tokenizer)
            input_ids = attention_mask = position_ids = None
            hidden_by_layer = None
            q = k = k_repeated = None
            try:
                input_ids, attention_mask, position_ids, offsets = tokenize_one(
                    tokenizer,
                    text,
                    device,
                )
                seq_len = int(input_ids.shape[1])
                if not args.min_seq_length <= seq_len <= args.max_seq_length:
                    skipped_length += 1
                    continue
                meta = example.get("meta", {}) or {}
                target_groups = resolve_clean_groups(
                    text,
                    offsets,
                    meta,
                    "target_record_spans",
                    args.block_size,
                    seq_len,
                )
                aggregation_groups = resolve_clean_groups(
                    text,
                    offsets,
                    meta,
                    "aggregation_spans",
                    args.block_size,
                    seq_len,
                )
                loss_type = str(meta.get("loss_type", "ranking"))
                layers = balanced_layers(
                    layer_queue,
                    args.num_layers,
                    args.layers_per_sample,
                    rng,
                )
                hidden_by_layer = capture_layer_inputs(
                    model,
                    input_ids,
                    attention_mask,
                    layers,
                )
                # Assignment releases the large token tensors while keeping the
                # local names valid for the unconditional cleanup in `finally`.
                input_ids = None
                attention_mask = None
                optimizer.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=device)
                merged: Dict[str, torch.Tensor] = {}

                for layer_idx in layers:
                    with torch.inference_mode():
                        q, k = extract_qk(
                            model,
                            hidden_by_layer.pop(layer_idx),
                            position_ids,
                            layer_idx,
                        )
                        q_blocks = math.ceil(seq_len / args.block_size)
                        rows = choose_query_rows(
                            q_blocks,
                            args.teacher_rows,
                            args.teacher_tail_rows,
                            rng,
                        )
                        teacher = exact_teacher_block_mass(
                            q,
                            k,
                            rows,
                            block_size=args.block_size,
                            tokens_per_row=args.teacher_tokens_per_row,
                            head_chunk=args.teacher_head_chunk,
                            key_chunk=args.teacher_key_chunk,
                            norm=args.score_norm,
                        )
                        k_repeated = repeat_kv_to_q_heads(k, q.shape[1])
                        block_scores_full, score_meta = (
                            kernels_conv_block_scores_infer_full(
                                q,
                                k_repeated,
                                block_size=args.block_size,
                                stride=args.score_stride,
                                norm=args.score_norm,
                                chunk_size=args.score_chunk_size,
                                sample_kernel_size=args.score_sample_kernel_size,
                                causal=True,
                            )
                        )
                    # Tensors created under inference_mode cannot be saved by
                    # conv2d for the trainable kernel's backward pass.
                    block_scores_full = block_scores_full.clone()
                    energy_full = apply_conv_energy(
                        block_scores_full,
                        conv.get_layer_weight(layer_idx),
                    )
                    q_real = score_meta["q_real_blocks"]
                    k_real = score_meta["k_real_blocks"]
                    energy = energy_full[:, :, :q_real, :k_real]
                    kl, l1, predicted = teacher_distribution_loss(
                        energy,
                        teacher,
                        rows,
                        args.positive_temperature,
                    )
                    topk_recall_loss = energy.new_zeros(())
                    if args.topk_recall_loss_weight > 0.0:
                        topk_recall_loss = topk_boundary_recall_loss(
                            energy,
                            teacher,
                            rows,
                            topk_ratio=args.topk_train_ratio,
                            positive_mass=args.topk_positive_mass,
                            boundary_negatives=args.topk_boundary_negatives,
                            margin=args.topk_boundary_margin,
                        )
                    rank = energy.new_zeros(())
                    aggregation = energy.new_zeros(())
                    if loss_type == "ranking" and target_groups:
                        rank = ranking_loss(
                            energy,
                            target_groups,
                            args.target_margin,
                            args.target_joint_weight,
                            args.target_query_tail_blocks,
                        )
                    elif loss_type == "aggregation" and aggregation_groups:
                        aggregation = aggregation_loss(
                            energy,
                            aggregation_groups,
                            args.target_query_tail_blocks,
                            args.aggregation_cover_mass,
                        )
                    negative = F.relu(-energy.float()).mean()
                    compression_weight = (
                        args.compression_loss_weight
                        + (args.compression_loss_weight_final - args.compression_loss_weight)
                        * step / max(1, args.steps)
                    )
                    entropy = entropy_loss(energy)
                    budget_value = current_budget(schedule, step)
                    budget = effective_budget_loss(energy, budget_value)
                    layer_loss = (
                        args.teacher_kl_weight * kl
                        + args.teacher_l1_weight * l1
                        + args.topk_recall_loss_weight * topk_recall_loss
                        + args.target_loss_weight * rank
                        + args.aggregation_loss_weight * aggregation
                        + args.negative_weight * negative
                        + compression_weight * entropy
                        + args.budget_loss_weight * budget
                    )
                    total_loss = total_loss + layer_loss / len(layers)

                    with torch.no_grad():
                        hard_mask = make_inference_chunked_block_mask(
                            energy_full.detach(),
                            threshold=args.threshold,
                            q_real_blocks=q_real,
                            k_real_blocks=k_real,
                            num_blocks_per_chunk=score_meta["num_blocks_per_chunk"],
                            causal=True,
                            topk_ratio=args.block_topk_ratio,
                        )
                        row_idx = torch.tensor(
                            rows,
                            device=device,
                            dtype=torch.long,
                        )
                        selected_mask = hard_mask.index_select(2, row_idx)
                        teacher_recall = (
                            teacher * selected_mask.float()
                        ).sum(-1).mean()
                        causal = make_causal_block_mask(q_real, k_real, device)
                        density = hard_mask.float().sum() / (
                            causal.float().sum() * hard_mask.shape[1]
                        ).clamp_min(1)
                        target_mean, target_all = target_coverage(
                            hard_mask,
                            target_groups or aggregation_groups,
                            args.target_query_tail_blocks,
                        )
                        stats = {
                            "teacher_kl": kl,
                            "teacher_l1": l1,
                            "topk_boundary_loss": topk_recall_loss,
                            "teacher_recall": teacher_recall,
                            "causal_density": density,
                            "target_group_coverage": target_mean,
                            "same_head_all_targets_hit": target_all,
                            "rank_loss": rank,
                            "aggregation_loss": aggregation,
                            "negative_penalty": negative,
                            "soft_effective_blocks": soft_effective_blocks(energy),
                            "current_block_budget": energy.new_tensor(float(budget_value)),
                            "energy_mean": energy.mean(),
                            "energy_std": energy.std(),
                        }
                    for name, value in stats.items():
                        merged[name] = (
                            merged.get(name, torch.zeros_like(value))
                            + value.detach() / len(layers)
                        )
                    del (
                        q,
                        k,
                        k_repeated,
                        teacher,
                        predicted,
                        block_scores_full,
                        energy_full,
                        energy,
                    )

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [conv.delta],
                    args.max_grad_norm,
                )
                optimizer.step()
                ema.update(conv)
                merged["lr"] = torch.as_tensor(step_lr, device=device)
                merged["grad_norm"] = torch.as_tensor(
                    float(grad_norm),
                    device=device,
                )
                running["loss"] = running.get("loss", 0.0) + float(
                    total_loss.detach().cpu()
                )
                for name, value in merged.items():
                    running[name] = running.get(name, 0.0) + float(
                        value.detach().float().cpu()
                    )
                running_count += 1
                completed = True

            except torch.OutOfMemoryError as error:
                free, total = torch.cuda.mem_get_info()
                raise RuntimeError(
                    "OOM during untruncated long-context training. "
                    "The sample is intentionally not truncated because that "
                    "would remove the evaluation query. Try "
                    "--layers_per_sample 1, --teacher_head_chunk 1, or "
                    "--teacher_key_chunk 512. "
                    f"CUDA free/total GiB={free / 2**30:.2f}/{total / 2**30:.2f}"
                ) from error
            finally:
                # Never `del` these names: some normal paths already release
                # them early, and `finally` also runs after `continue`.
                input_ids = None
                attention_mask = None
                position_ids = None
                hidden_by_layer = None
                if step % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        if step % args.log_steps == 0:
            denom = max(1, running_count)
            metrics = " ".join(
                f"{name}={value / denom:.6f}"
                for name, value in running.items()
            )
            print(
                f"step={step:05d} layers={','.join(map(str, layers))} "
                f"seq_len={seq_len} skipped_length={skipped_length} {metrics}"
            )
            running.clear()
            running_count = 0

        if step % args.save_steps == 0:
            conv.save(args.out)
            raw_step = args.out.replace(".pt", f"_step{step}.pt")
            ema_step = args.out.replace(".pt", f"_ema_step{step}.pt")
            conv.save(raw_step)
            ema.save(ema_step)
            state_path = save_state(args.out, step, conv, optimizer, ema)
            print(f"[save] raw={raw_step}")
            print(f"[save] ema={ema_step}")
            print(f"[save] state={state_path}")

    conv.save(args.out)
    ema.save(args.out.replace(".pt", "_ema.pt"))
    state_path = save_state(args.out, args.steps, conv, optimizer, ema)
    print(f"[final] raw={args.out}")
    print(f"[final] ema={args.out.replace('.pt', '_ema.pt')}")
    print(f"[final] state={state_path}")


if __name__ == "__main__":
    main()
