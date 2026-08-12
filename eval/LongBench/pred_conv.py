import os
from pathlib import Path
import zipfile
import torch
import json
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    repeat_kv,
    apply_rotary_pos_emb,
    nn,
)
import math
from xattn.src.Xattention import Xattention_prefill
from xattn.src.Flexprefill import Flexprefill_prefill
from xattn.src.Minference import Minference_prefill
from xattn.src.Conv import Conv_prefill
from flash_attn import flash_attn_func
import types
from ratio import max_ratio, max


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")

    parser.add_argument("--task", type=str, help="task name", required=True)

    parser.add_argument(
        "--method",
        type=str,
        default="full",
    )
    parser.add_argument(
        "--conv_weight_path",
        type=str,
        default="xattn/conv_weights/conv_kernel_7x7.pt",
        help="Path to conv kernel .pt. Supports [1,1,7,7], [H,1,7,7], or [L,H,1,7,7].",
    )
    parser.add_argument(
        "--conv_safe_topk",
        action="store_true",
        help="Use safe causal top-k fallback instead of chunk-wise find_blocks_chunked. Default: False, for fair comparison with xattn.",
        default=False,
    )

    parser.add_argument(
        "--conv_use_triton",
        action="store_true",
        help="Use Conv_prefill triton estimation path. Default is False for safety.",
        default=True,
    )

    parser.add_argument(
        "--conv_fallback_topk",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--no_conv_fallback_full",
        action="store_true",
        help="Disable fallback to full flash attention when conv path fails.",
        default=True,
    )
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt

import re


def remove_repeated_tail(text: str, max_unit_chars: int = 20, min_repeats: int = 3) -> str:
    """
    删除末尾重复片段，例如：
    会议结束。会议结束。会议结束。 -> 会议结束。
    """
    text = text.strip()
    if not text:
        return text

    changed = True
    while changed:
        changed = False
        max_len = min(max_unit_chars, len(text) // min_repeats)

        for unit_len in range(2, max_len + 1):
            unit = text[-unit_len:]

            if not unit.strip():
                continue

            pos = len(text)
            count = 0

            while pos >= unit_len and text[pos - unit_len:pos] == unit:
                count += 1
                pos -= unit_len

            if count >= min_repeats:
                # 保留一次重复单元
                text = text[:pos + unit_len].rstrip()
                changed = True
                break

    # 专门处理中文会议总结常见坏尾巴
    text = re.sub(r"(会议结束[。.\s]*){3,}$", "会议结束。", text).strip()
    return text


def is_repeating_tail(text: str) -> bool:
    """
    用于生成阶段提前停止。
    """
    text = text.strip()
    if len(text) < 20:
        return False

    if re.search(r"(会议结束[。.\s]*){4,}$", text):
        return True

    # 检查任意短片段在末尾重复多次
    for unit_len in range(2, min(20, len(text) // 4) + 1):
        unit = text[-unit_len:]
        if not unit.strip():
            continue

        pos = len(text)
        count = 0
        while pos >= unit_len and text[pos - unit_len:pos] == unit:
            count += 1
            pos -= unit_len

        if count >= 4:
            return True

    return False

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .split("<|eot_id|>")[0]
            .split("<|end_of_text|>")[0]
            .strip()
        )

        # 清理 LongBench summarization 里常见的助手式尾巴
        stop_phrases = [
            "\n\nNote:",
            "\nNote:",
            "\n\nWord Count:",
            "\nWord Count:",
            "\n\nReferences:",
            "\nReferences:",
            "\n\nPlease let me know",
            "\nPlease let me know",
            "\n\nHere is the summary",
            "\nHere is the summary",
            "\n\nBest regards",
            "\nBest regards",
        ]
        for s in stop_phrases:
            if s in response:
                response = response.split(s)[0].strip()

        response = remove_repeated_tail(response)

    elif "Llama-2-7B-32K-Instruct" in model_name:
        response = (
            response.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
        response = remove_repeated_tail(response)

    return response


@torch.no_grad()
def new_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.config.num_attention_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if key_states.shape[2] == query_states.shape[2]:
        if self.method == "xattn":
            self.threshold = self.threshold.to(key_states.device)
            threshold = self.threshold
            attn_output = Xattention_prefill(
                query_states,
                key_states,
                value_states,
                norm=1,
                stride=8,
                threshold=threshold,
                use_triton=True,
                keep_sink=True,
                keep_recent=True,
            )
        elif self.method == "flex":
            attn_output = Flexprefill_prefill(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                gamma=0.9,
                tau=0.1,
            ).transpose(1, 2)
        elif self.method == "minference":
            attn_output = Minference_prefill(
                query_states, key_states, value_states
            )
        elif self.method == "full":
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=True,
            ).transpose(1, 2)
        elif self.method == "conv":
            # Keep the same threshold object as xattn.
            # Do NOT reduce tensor thresholds to float here; find_blocks_chunked can
            # receive the original self.threshold tensor, which keeps layer/head-wise
            # threshold behavior aligned with Xattention_prefill.
            threshold = self.threshold
            if isinstance(threshold, torch.Tensor):
                threshold = threshold.to(key_states.device)

            conv_weight_path = getattr(
                self,
                "conv_weight_path",
                "xattn/conv_weights/conv_kernel_7x7.pt",
            )
            conv_layer_idx = getattr(self, "conv_layer_idx", None)
            conv_safe_topk = getattr(self, "conv_safe_topk", False)
            conv_use_triton = getattr(self, "conv_use_triton", False)
            conv_fallback_topk = getattr(self, "conv_fallback_topk", 8)
            conv_fallback_full = getattr(self, "conv_fallback_full", True)

            try:
                attn_output = Conv_prefill(
                    query_states,
                    key_states,
                    value_states,
                    norm=1,
                    stride=8,
                    threshold=threshold,
                    use_triton=conv_use_triton,
                    keep_sink=True,
                    keep_recent=True,
                    conv_weight_path=conv_weight_path,
                    layer_idx=conv_layer_idx,
                    conv_safe_topk=conv_safe_topk,
                    fallback_topk=conv_fallback_topk,
                )

                # 尽早同步，避免 CUDA 错误延迟到后续样本才报
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            except Exception as e:
                print(
                    f"[pred.py WARN] Conv_prefill failed at layer={conv_layer_idx}, "
                    f"q_len={query_states.shape[2]}, k_len={key_states.shape[2]}, "
                    f"fallback_full={conv_fallback_full}, error={repr(e)}"
                )

                if not conv_fallback_full:
                    raise

                # 注意：如果已经是真正的 illegal memory access，
                # 当前 CUDA context 可能已损坏，fallback 也可能失败。
                torch.cuda.empty_cache()

                attn_output = flash_attn_func(
                    query_states.transpose(1, 2),
                    key_states.transpose(1, 2),
                    value_states.transpose(1, 2),
                    causal=True,
                ).transpose(1, 2)
    else:
        ########################################################################################################################
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
):
    preds = []
    pbar = tqdm(data)
    for idx, json_obj in enumerate(pbar):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        pbar.set_description(f"Generating for {idx}, len = {input.input_ids.shape[-1]}")
        with torch.no_grad():
            output = model(
                input_ids=input.input_ids,
                past_key_values=None,
                use_cache=True,
                num_logits_to_keep=1,
            )
            past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                    num_logits_to_keep=1,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() in eos_token_ids:
                    break

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        print(f"Prediction: {pred}")
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def load_longbench_compatible(task_name, cache_dir="eval/LongBench/data_cache"):
    """
    Compatible with datasets>=4.0.
    Loads LongBench task from data.zip instead of executing LongBench.py.
    """
    cache_dir = Path(cache_dir)
    extract_dir = cache_dir / "LongBench"
    jsonl_path = extract_dir / "data" / f"{task_name}.jsonl"

    if not jsonl_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

        # If cluster cannot connect to internet, manually download data.zip and set:
        # export LONGBENCH_DATA_ZIP=/path/to/data.zip
        local_zip = os.environ.get("LONGBENCH_DATA_ZIP")

        if local_zip:
            zip_path = Path(local_zip)
        else:
            zip_path = Path(
                hf_hub_download(
                    repo_id="zai-org/LongBench",
                    filename="data.zip",
                    repo_type="dataset",
                    cache_dir=str(cache_dir),
                )
            )

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Cannot find {jsonl_path}. Check whether data.zip was extracted correctly."
        )

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return Dataset.from_list(records)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
    )

    generation_config = GenerationConfig.from_pretrained(path)

    eos_token_ids = generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    # 关键：补充 Llama-3 / Llama-3.1 chat stop token
    for tok in ["<|eot_id|>", "<|end_of_text|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tok_id, int) and tok_id >= 0 and tok_id not in eos_token_ids:
            eos_token_ids.append(tok_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.eval()
    return model, tokenizer, eos_token_ids


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("eval/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("eval/LongBench/config/model2maxlen.json", "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model
    # define your model
    # breakpoint()
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        model2path[model_name], model_name
    )

    for name, module in model.named_modules():
        if name.split(".")[-1] == "self_attn":
            layer_idx = int(name.split(".")[2])

            module.method = args.method

            # conv 专用信息
            module.conv_layer_idx = layer_idx
            module.conv_weight_path = args.conv_weight_path
            module.conv_safe_topk = args.conv_safe_topk
            module.conv_use_triton = args.conv_use_triton
            module.conv_fallback_topk = args.conv_fallback_topk
            module.conv_fallback_full = not args.no_conv_fallback_full

            if args.method == "xattn" or args.method == "conv":
                module.threshold = torch.tensor(max[layer_idx])
                # module.threshold = torch.tensor(0.9)

            module.forward = types.MethodType(new_attention_forward, module)

    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("eval/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/LongBench/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("eval/LongBench/pred"):
        os.makedirs("eval/LongBench/pred")
    if not os.path.exists("eval/LongBench/pred_e"):
        os.makedirs("eval/LongBench/pred_e")
    for dataset in datasets:
        load_name = f"{dataset}_e" if args.e else dataset
        data = load_longbench_compatible(load_name)

        pred_root = "eval/LongBench/pred_e" if args.e else "eval/LongBench/pred"
        pred_dir = f"{pred_root}/{model_name}"
        os.makedirs(pred_dir, exist_ok=True)

        if args.method == "full":
            out_path = f"{pred_dir}/full/{dataset}-full.jsonl"
        elif args.method == "xattn":
            out_path = f"{pred_dir}/xattn/{dataset}-xattn-stride=8.jsonl"
        elif args.method == "flex":
            out_path = f"{pred_dir}/flex/{dataset}-flex.jsonl"
        elif args.method == "minference":
            out_path = f"{pred_dir}/minference/{dataset}-minference.jsonl"
        elif args.method == "conv":
            out_path = f"{pred_dir}/conv/{dataset}-conv-stride=8.jsonl"
        else:
            raise ValueError(f"Unknown method: {args.method}")

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        preds = get_pred(
            model,
            tokenizer,
            eos_token_ids,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")