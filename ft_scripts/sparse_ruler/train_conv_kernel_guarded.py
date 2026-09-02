#!/usr/bin/env python3
"""
Guarded RULER-Mix conv-kernel training.

Design goals:
1. Use frozen full-attention Llama as dense-attention teacher.
2. Use synthetic metadata as evidence/aggregation supervision.
3. Use bounded residual parameterization: W = W0 + alpha * tanh(delta).
4. Use dense guards and optional bad-batch rejection to prevent late collapse.
5. Match inference-time sampled kernels_conv + full-map conv + find_blocks_chunked selection.

Expected repo placement:
  ft_scripts/conv_ruler/train_conv_kernel_guarded.py

It imports existing local modules from xattn/conv code:
  conv_sparse_ops.py
  conv_kernel.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

CURRENT_DIR = Path(__file__).resolve().parent
# Make this work when copied into ft_scripts/conv_ruler while conv_kernel.py is in sibling/parent dirs.
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR.parent))
sys.path.insert(0, str(CURRENT_DIR.parent.parent))
sys.path.insert(0, str(Path.cwd()))

import conv_kernel as _conv_kernel_module  # type: ignore
import conv_sparse_ops as _conv_sparse_ops_module  # type: ignore
from conv_kernel import make_layer_head_kernel  # type: ignore
from conv_sparse_ops import (  # type: ignore
    apply_conv_energy,
    dense_attention_teacher,
    hard_block_sparse_attention,
    kernels_conv_block_scores,
    kernels_conv_block_scores_infer_full,
    make_causal_block_mask,
    make_hard_block_mask,
    make_inference_chunked_block_mask,
    probs_to_block_scores,
    repeat_kv_to_q_heads,
    soft_surrogate_attention,
)


# ----------------------------- model utils -----------------------------


def format_example_to_text(example: Dict[str, Any], tokenizer) -> str:
    if "messages" in example and example["messages"] is not None:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    if "text" in example:
        return str(example["text"])
    return json.dumps(example, ensure_ascii=False)


def get_truncated_token_length(tokenizer, text: str, max_seq_length: int) -> int:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False,
    )
    return len(encoded["input_ids"])


def filter_dataset_by_truncated_length(ds, tokenizer, min_seq_length: int, max_seq_length: int):
    valid_indices = []
    too_short = 0
    failed = 0
    print("=" * 80)
    print("Filtering dataset by truncated token length")
    print(f"min_seq_length={min_seq_length}")
    print(f"max_seq_length={max_seq_length}")
    print("=" * 80)
    for i, example in enumerate(ds):
        try:
            text = format_example_to_text(example, tokenizer)
            n = get_truncated_token_length(tokenizer, text, max_seq_length)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"[filter warning] failed idx={i}: {repr(e)}")
            continue
        if n >= min_seq_length:
            valid_indices.append(i)
        else:
            too_short += 1
        if (i + 1) % 1000 == 0:
            print(f"[filter] scanned={i+1} valid={len(valid_indices)} too_short={too_short} failed={failed}")
    print("=" * 80)
    print(f"Original dataset size: {len(ds)}")
    print(f"Valid dataset size:    {len(valid_indices)}")
    print(f"Too short skipped:     {too_short}")
    print(f"Failed skipped:        {failed}")
    print("=" * 80)
    if not valid_indices:
        raise RuntimeError("No valid samples after length filtering.")
    return ds.select(valid_indices)


def tokenize_one(tokenizer, text: str, min_seq_length: int, max_seq_length: int, device: str):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offsets = encoded["offset_mapping"][0].tolist()
    seq_len = input_ids.shape[1]
    if seq_len < min_seq_length:
        return None
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    return input_ids, attention_mask, position_ids, offsets


@torch.no_grad()
def get_all_hidden_inputs(model, input_ids, attention_mask):
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return outputs.hidden_states


@torch.no_grad()
def extract_llama_qkv(model, hidden_states, position_ids, layer_idx: int):
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    bsz, seq_len, _ = hidden_states.shape
    num_heads = getattr(attn, "num_heads", model.config.num_attention_heads)
    num_kv_heads = getattr(attn, "num_key_value_heads", model.config.num_key_value_heads)
    head_dim = getattr(attn, "head_dim", model.config.hidden_size // model.config.num_attention_heads)

    q = attn.q_proj(hidden_states)
    k = attn.k_proj(hidden_states)
    v = attn.v_proj(hidden_states)

    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()

    cos, sin = model.model.rotary_emb(hidden_states, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q.detach(), k.detach(), v.detach()


# ----------------------------- bounded kernel -----------------------------


def load_or_init_weight(init_path: Optional[str], num_layers: int, num_heads: int, kernel_size: int) -> torch.Tensor:
    if str(init_path or "").strip().lower() in {
        "identity",
        "scratch_identity",
        "no_conv",
    }:
        # Exact no-convolution baseline: grouped conv2d returns the original
        # score map before any learned spatial redistribution.
        w = torch.zeros(
            num_layers,
            num_heads,
            kernel_size,
            kernel_size,
            dtype=torch.float32,
        )
        w[:, :, kernel_size // 2, kernel_size // 2] = 1.0
        print("[init] scratch identity (no-conv baseline)")
        return w.contiguous()
    if init_path is not None and os.path.exists(init_path):
        w = torch.load(init_path, map_location="cpu", weights_only=True).float()
        if tuple(w.shape) == (1, 1, kernel_size, kernel_size):
            w = w[0, 0][None, None, :, :].repeat(num_layers, num_heads, 1, 1)
        elif w.dim() == 5 and tuple(w.shape[-3:]) == (1, kernel_size, kernel_size):
            w = w[:, :, 0, :, :]
        elif tuple(w.shape) != (num_layers, num_heads, kernel_size, kernel_size):
            raise ValueError(f"Unsupported init weight shape={tuple(w.shape)}")
        return w.contiguous()
    return make_layer_head_kernel(num_layers, num_heads, kernel_size).float().contiguous()


class BoundedResidualConvKernel(nn.Module):
    """Train residual weights or absolute weights inside explicit bounds."""

    def __init__(
        self,
        init_path: Optional[str],
        num_layers: int,
        num_heads: int,
        kernel_size: int,
        alpha: float,
        weight_min: Optional[float] = None,
        weight_max: Optional[float] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.alpha = float(alpha)
        if (weight_min is None) != (weight_max is None):
            raise ValueError("weight_min and weight_max must be set together")
        if weight_min is not None and weight_max is not None:
            if not float(weight_min) < float(weight_max):
                raise ValueError("weight_min must be smaller than weight_max")
            self.weight_min = float(weight_min)
            self.weight_max = float(weight_max)
        else:
            self.weight_min = None
            self.weight_max = None
        w0 = load_or_init_weight(init_path, num_layers, num_heads, kernel_size)
        self.register_buffer("anchor", w0)
        if self.weight_min is None:
            initial_delta = torch.zeros_like(w0)
        else:
            midpoint = 0.5 * (self.weight_min + self.weight_max)
            half_range = 0.5 * (self.weight_max - self.weight_min)
            normalized = (w0 - midpoint) / half_range
            if torch.any(normalized <= -1.0) or torch.any(normalized >= 1.0):
                raise ValueError(
                    "initial kernel must lie strictly inside "
                    f"({self.weight_min}, {self.weight_max})"
                )
            initial_delta = torch.atanh(normalized)
        self.delta = nn.Parameter(initial_delta)

    def forward(self) -> torch.Tensor:
        if self.weight_min is not None and self.weight_max is not None:
            midpoint = 0.5 * (self.weight_min + self.weight_max)
            half_range = 0.5 * (self.weight_max - self.weight_min)
            return midpoint + half_range * torch.tanh(self.delta)
        return self.anchor + self.alpha * torch.tanh(self.delta)

    def get_layer_weight(self, layer_idx: int) -> torch.Tensor:
        return self.forward()[layer_idx]

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        return self.forward().detach().float().cpu().contiguous()

    @torch.no_grad()
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.effective_weight(), path)

    @torch.no_grad()
    def print_kernel(self, layer_idx: int = 0, head_idx: int = 0):
        w = self.effective_weight()
        print(f"kernel[layer={layer_idx}, head={head_idx}], shape={tuple(w.shape)}:")
        print(w[layer_idx, head_idx])


# ----------------------------- span -> block groups -----------------------------


def marker_span_to_char_span(text: str, start_marker: str, end_marker: str) -> Optional[Tuple[int, int]]:
    s = text.find(start_marker)
    if s < 0:
        return None
    e = text.find(end_marker, s + len(start_marker))
    if e < 0:
        return None
    return s, e + len(end_marker)


def char_span_to_token_blocks(
    offsets: List[Tuple[int, int]] | List[List[int]],
    char_span: Tuple[int, int],
    block_size: int,
    seq_len: int,
) -> Optional[List[int]]:
    cs, ce = char_span
    toks: List[int] = []
    for i, off in enumerate(offsets[:seq_len]):
        if i >= seq_len:
            break
        if len(off) != 2:
            continue
        ts, te = int(off[0]), int(off[1])
        # special tokens sometimes have (0,0)
        if te <= ts:
            continue
        if te > cs and ts < ce:
            toks.append(i)
    if not toks:
        return None
    b0 = min(toks) // block_size
    b1 = max(toks) // block_size
    return list(range(b0, b1 + 1))


def resolve_block_groups(
    text: str,
    offsets: List[Tuple[int, int]] | List[List[int]],
    meta: Dict[str, Any],
    key: str,
    block_size: int,
    seq_len: int,
) -> List[List[int]]:
    spans = meta.get(key, []) or []
    groups: List[List[int]] = []
    for sp in spans:
        start_marker = sp.get("start_marker")
        end_marker = sp.get("end_marker")
        if not start_marker or not end_marker:
            continue
        ch = marker_span_to_char_span(text, start_marker, end_marker)
        if ch is None:
            continue
        blocks = char_span_to_token_blocks(offsets, ch, block_size, seq_len)
        if blocks is not None:
            groups.append(blocks)
    return groups


# ----------------------------- losses and stats -----------------------------


def masked_softmax(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x.float().masked_fill(~mask, -1e9)
    return F.softmax(x, dim=dim)


def block_distribution_kl(energy: torch.Tensor, block_scores: torch.Tensor) -> torch.Tensor:
    """KL(dense_block_distribution || energy_distribution), averaged."""
    b, h, qb, kb = block_scores.shape
    causal = make_causal_block_mask(qb, kb, block_scores.device)[None, None, :, :]
    dense = block_scores.float().clamp_min(0.0) * causal.float()
    dense = dense / (dense.sum(dim=-1, keepdim=True) + 1e-8)
    log_pred = F.log_softmax(energy.float().masked_fill(~causal, -1e9), dim=-1)
    kl = (dense * (torch.log(dense + 1e-8) - log_pred)).sum(dim=-1)
    return kl.mean()


def entropy_loss(energy: torch.Tensor) -> torch.Tensor:
    b, h, qb, kb = energy.shape
    causal = make_causal_block_mask(qb, kb, energy.device)[None, None, :, :]
    p = masked_softmax(energy, causal, dim=-1)
    ent = -(p * torch.log(p + 1e-8)).sum(dim=-1)
    # normalize by log(kb) so scale is roughly 0..1
    return ent.mean() / max(math.log(max(kb, 2)), 1e-8)


def select_tail_q_indices(qb: int, n_tail: int, device) -> torch.Tensor:
    start = max(0, qb - int(n_tail))
    return torch.arange(start, qb, device=device)


def group_scores_from_energy(
    energy: torch.Tensor,
    groups: List[List[int]],
    target_query_tail_blocks: int,
) -> Optional[torch.Tensor]:
    """
    Return [num_groups] differentiable group scores from tail query blocks.
    For each group, score=max energy over group blocks, averaged over tail/head.
    """
    if not groups:
        return None
    _, _, qb, kb = energy.shape
    q_idx = select_tail_q_indices(qb, target_query_tail_blocks, energy.device)
    e_tail = energy[:, :, q_idx, :]  # [1,H,T,K]
    scores: List[torch.Tensor] = []
    for g in groups:
        valid = [b for b in g if 0 <= b < kb]
        if not valid:
            continue
        idx = torch.tensor(valid, device=energy.device, dtype=torch.long)
        sg = e_tail.index_select(-1, idx).amax(dim=-1).mean()
        scores.append(sg)
    if not scores:
        return None
    return torch.stack(scores)


def hard_negative_score(
    energy: torch.Tensor,
    positive_groups: List[List[int]],
    target_query_tail_blocks: int,
    far_distance_blocks: int = 0,
) -> torch.Tensor:
    _, _, qb, kb = energy.shape
    q_idx = select_tail_q_indices(qb, target_query_tail_blocks, energy.device)
    e_tail = energy[:, :, q_idx, :].float()
    mask = torch.ones(kb, dtype=torch.bool, device=energy.device)
    for g in positive_groups:
        for b in g:
            if 0 <= b < kb:
                mask[b] = False
    # causal valid for each tail q block
    causal = make_causal_block_mask(qb, kb, energy.device)[q_idx]
    full_mask = mask[None, None, None, :] & causal[None, None, :, :]
    if far_distance_blocks > 0:
        q_blocks = q_idx[:, None]
        k_blocks = torch.arange(kb, device=energy.device)[None, :]
        far = (q_blocks - k_blocks) >= far_distance_blocks
        full_mask = full_mask & far[None, None, :, :]
    if not full_mask.any():
        full_mask = mask[None, None, None, :] & causal[None, None, :, :]
    return e_tail.masked_fill(~full_mask, -1e9).amax()


def ranking_loss(
    energy: torch.Tensor,
    groups: List[List[int]],
    margin: float,
    joint_weight: float,
    target_query_tail_blocks: int,
) -> torch.Tensor:
    """
    Head/query-aligned target ranking loss.

    The previous implementation compared a positive score averaged over every
    head and tail query against one global maximum negative taken over all heads
    and queries.  A single outlier negative therefore dominated the whole batch
    and made the hinge loss stay around tens even when target-hit metrics were
    already high.

    Here every target is compared with the hardest negative from the SAME
    head/query.  The individual term allows different heads to retrieve
    different targets; the joint term asks one head to cover all targets.
    """
    if not groups:
        return energy.new_tensor(0.0)

    b, _, qb, kb = energy.shape
    q_idx = select_tail_q_indices(qb, target_query_tail_blocks, energy.device)
    e_tail = energy[:, :, q_idx, :].float()  # [B,H,T,K]
    causal = make_causal_block_mask(qb, kb, energy.device)[q_idx]  # [T,K]

    valid_groups: List[List[int]] = []
    positive_keys = torch.zeros(kb, dtype=torch.bool, device=energy.device)
    for group in groups:
        valid = sorted({int(x) for x in group if 0 <= int(x) < kb})
        if valid:
            valid_groups.append(valid)
            positive_keys[valid] = True
    if not valid_groups:
        return energy.new_tensor(0.0)

    negative_mask = causal & ~positive_keys[None, :]
    # A causal query always has non-target keys in these RULER samples.  Keep a
    # finite fallback for defensive handling of degenerate synthetic examples.
    has_negative = negative_mask.any(dim=-1)
    neg = e_tail.masked_fill(
        ~negative_mask[None, None, :, :],
        -1e4,
    ).amax(dim=-1)  # [B,H,T]
    neg = torch.where(has_negative[None, None, :], neg, torch.zeros_like(neg))

    group_losses: List[torch.Tensor] = []
    group_visible: List[torch.Tensor] = []
    for valid in valid_groups:
        idx = torch.tensor(valid, device=energy.device, dtype=torch.long)
        visible = causal.index_select(-1, idx).any(dim=-1)  # [T]
        pos = e_tail.index_select(-1, idx)
        pos = pos.masked_fill(
            ~causal.index_select(-1, idx)[None, None, :, :],
            -1e4,
        ).amax(dim=-1)  # [B,H,T]
        hinge = F.relu(float(margin) - pos + neg)
        hinge = torch.where(visible[None, None, :], hinge, torch.zeros_like(hinge))
        group_losses.append(hinge)
        group_visible.append(visible)

    losses = torch.stack(group_losses, dim=0)  # [G,B,H,T]
    visible = torch.stack(group_visible, dim=0)  # [G,T]

    # Each target group may use its best head.
    individual_per_group = losses.amin(dim=2)  # [G,B,T]
    individual_mask = visible[:, None, :].expand(-1, b, -1)
    individual = (
        individual_per_group * individual_mask.float()
    ).sum() / individual_mask.float().sum().clamp_min(1.0)

    # One shared head must handle the worst visible target group.
    joint_per_head = losses.amax(dim=0)  # [B,H,T]
    joint_per_query = joint_per_head.amin(dim=1)  # [B,T]
    joint_mask = visible.any(dim=0)[None, :].expand(b, -1)
    joint = (
        joint_per_query * joint_mask.float()
    ).sum() / joint_mask.float().sum().clamp_min(1.0)

    return (1.0 - float(joint_weight)) * individual + float(joint_weight) * joint


def aggregation_loss(
    energy: torch.Tensor,
    groups: List[List[int]],
    target_query_tail_blocks: int,
    cover_mass_target: float = 0.20,
) -> torch.Tensor:
    """
    FWE/CWE style: encourage softmax mass over many occurrence groups.
    Unlike ranking loss, this does not require every occurrence to be top-1; it
    asks the tail query blocks to reserve enough total probability mass for all
    useful occurrence regions.
    """
    if not groups:
        return energy.new_tensor(0.0)
    _, _, qb, kb = energy.shape
    q_idx = select_tail_q_indices(qb, target_query_tail_blocks, energy.device)
    causal = make_causal_block_mask(qb, kb, energy.device)[q_idx]
    p = masked_softmax(energy[:, :, q_idx, :], causal[None, None, :, :], dim=-1)
    pos_mask = torch.zeros(kb, dtype=torch.bool, device=energy.device)
    for g in groups:
        for b in g:
            if 0 <= b < kb:
                pos_mask[b] = True
    if not pos_mask.any():
        return energy.new_tensor(0.0)
    cover = p[..., pos_mask].sum(dim=-1).mean()
    cover_loss = F.relu(float(cover_mass_target) - cover).pow(2)
    # also make top occurrence segments stronger than average negatives
    rank = ranking_loss(energy, groups[: min(len(groups), 12)], margin=0.05, joint_weight=0.2, target_query_tail_blocks=target_query_tail_blocks)
    return cover_loss + 0.1 * rank


def compute_mass_recall(block_scores: torch.Tensor, hard_mask: torch.Tensor, key_mask: Optional[torch.Tensor] = None, q_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    bs = block_scores.float()
    hm = hard_mask.float()
    if key_mask is not None:
        bs = bs * key_mask[None, None, None, :].float()
        hm = hm * key_mask[None, None, None, :].float()
    if q_mask is not None:
        bs = bs * q_mask[None, None, :, None].float()
        hm = hm * q_mask[None, None, :, None].float()
    denom = bs.sum(dim=-1) + 1e-8
    rec = (bs * hm).sum(dim=-1) / denom
    valid = denom > 1e-7
    if valid.any():
        return rec[valid].mean()
    return rec.mean()


def compute_proxy_stats(
    block_scores: torch.Tensor,
    hard_mask: torch.Tensor,
    target_groups: List[List[int]],
    target_query_tail_blocks: int,
    far_distance_blocks: int,
) -> Dict[str, torch.Tensor]:
    b, h, qb, kb = block_scores.shape
    device = block_scores.device
    causal = make_causal_block_mask(qb, kb, device)
    causal_density = hard_mask.float().sum() / (causal.float().sum() * b * h + 1e-8)
    mass = compute_mass_recall(block_scores, hard_mask)

    q_tail = torch.zeros(qb, dtype=torch.bool, device=device)
    q_tail[max(0, qb - target_query_tail_blocks):] = True
    tail = compute_mass_recall(block_scores, hard_mask, q_mask=q_tail)

    p10_mask = torch.zeros(kb, dtype=torch.bool, device=device)
    p10_mask[: max(1, int(math.ceil(kb * 0.10)))] = True
    p10 = compute_mass_recall(block_scores, hard_mask, key_mask=p10_mask, q_mask=q_tail)

    q_ids = torch.arange(qb, device=device)[:, None]
    k_ids = torch.arange(kb, device=device)[None, :]
    far_matrix = (q_ids - k_ids) >= far_distance_blocks
    bs_far = block_scores.float() * far_matrix[None, None, :, :].float()
    denom = bs_far.sum(dim=-1) + 1e-8
    far_rec = (bs_far * hard_mask.float()).sum(dim=-1) / denom
    valid = denom > 1e-7
    far = far_rec[valid].mean() if valid.any() else far_rec.mean()

    target_any = hard_mask.new_tensor(0.0, dtype=torch.float32)
    target_mean = hard_mask.new_tensor(0.0, dtype=torch.float32)
    same_head_all = hard_mask.new_tensor(1.0, dtype=torch.float32)
    if target_groups:
        tail_mask = hard_mask[:, :, q_tail, :]  # [B,H,T,K]
        group_hits: List[torch.Tensor] = []
        for g in target_groups:
            valid_g = [x for x in g if 0 <= x < kb]
            if not valid_g:
                continue
            idx = torch.tensor(valid_g, device=device)
            hit_per_head = tail_mask.index_select(-1, idx).any(dim=-1).any(dim=-1).float()  # [B,H]
            group_hits.append(hit_per_head)
        if group_hits:
            gh = torch.stack(group_hits, dim=0)  # [G,B,H]
            target_mean = gh.mean()
            target_any = gh.any(dim=-1).float().mean()  # group has any head
            same_head_all = gh.all(dim=0).float().max(dim=-1).values.mean()  # any head hits all groups

    selected_blocks = hard_mask.float().sum(dim=-1).mean()
    return {
        "causal_density": causal_density.detach(),
        "mass_recall": mass.detach(),
        "tail_mass_recall": tail.detach(),
        "p10_mass_recall": p10.detach(),
        "far_mass_recall": far.detach(),
        "target_hit_mean": target_mean.detach(),
        "target_hit_any_head": target_any.detach(),
        "same_head_all_targets_hit": same_head_all.detach(),
        "selected_blocks": selected_blocks.detach(),
    }


def parse_target_blocks_schedule(raw: str) -> List[Tuple[int, int]]:
    # "0:16,2000:14,4000:12"
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split(":")
        out.append((int(a), int(b)))
    return sorted(out)


def current_budget(schedule: List[Tuple[int, int]], step: int) -> int:
    budget = schedule[0][1] if schedule else 16
    for s, b in schedule:
        if step >= s:
            budget = b
    return budget


def linear_weight(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    t = min(1.0, max(0.0, step / float(total_steps)))
    return start + (end - start) * t


def soft_effective_blocks(energy: torch.Tensor) -> torch.Tensor:
    """Differentiable effective number of blocks: exp(entropy(p_energy))."""
    b, h, qb, kb = energy.shape
    causal = make_causal_block_mask(qb, kb, energy.device)[None, None, :, :]
    p = masked_softmax(energy, causal, dim=-1)
    ent = -(p * torch.log(p + 1e-8)).sum(dim=-1)
    return torch.exp(ent).mean()


def effective_budget_loss(energy: torch.Tensor, budget: int) -> torch.Tensor:
    """Penalize soft effective block count above the current budget."""
    eff = soft_effective_blocks(energy)
    b = max(float(budget), 1.0)
    return F.relu(eff - b).pow(2) / (b * b)


def soft_mass_recall_from_energy(
    energy: torch.Tensor,
    block_scores: torch.Tensor,
    key_mask: Optional[torch.Tensor] = None,
    q_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable proxy of mass recall using a soft sigmoid selection mask.
    This lets guard_p10/tail/far enter the loss, while hard selected-block
    statistics are still computed with the actual hard mask.
    """
    b, h, qb, kb = block_scores.shape
    causal = make_causal_block_mask(qb, kb, energy.device)[None, None, :, :]
    centered = energy.float() - energy.float().mean(dim=-1, keepdim=True)
    soft_mask = torch.sigmoid(centered / max(float(temperature), 1e-6)) * causal.float()

    bs = block_scores.float().clamp_min(0.0)
    if key_mask is not None:
        bs = bs * key_mask[None, None, None, :].float()
        soft_mask = soft_mask * key_mask[None, None, None, :].float()
    if q_mask is not None:
        bs = bs * q_mask[None, None, :, None].float()
        soft_mask = soft_mask * q_mask[None, None, :, None].float()

    denom = bs.sum(dim=-1) + 1e-8
    rec = (bs * soft_mask).sum(dim=-1) / denom
    valid = denom > 1e-7
    if valid.any():
        return rec[valid].mean()
    return rec.mean()


def soft_recall_guard_loss(
    energy: torch.Tensor,
    block_scores: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Differentiable p10/tail/far guard using soft selection mask."""
    _, _, qb, kb = block_scores.shape
    device = block_scores.device

    q_tail = torch.zeros(qb, dtype=torch.bool, device=device)
    q_tail[max(0, qb - args.target_query_tail_blocks):] = True

    p10_mask = torch.zeros(kb, dtype=torch.bool, device=device)
    p10_mask[: max(1, int(math.ceil(kb * 0.10)))] = True

    q_ids = torch.arange(qb, device=device)[:, None]
    k_ids = torch.arange(kb, device=device)[None, :]
    far_matrix = (q_ids - k_ids) >= args.far_distance_blocks
    bs_far = block_scores.float().clamp_min(0.0) * far_matrix[None, None, :, :].float()

    soft_tail = soft_mass_recall_from_energy(energy, block_scores, q_mask=q_tail, temperature=args.temperature)
    soft_p10 = soft_mass_recall_from_energy(energy, block_scores, key_mask=p10_mask, q_mask=q_tail, temperature=args.temperature)

    # Far recall uses the same helper by passing an already-filtered block_scores tensor.
    denom_far = bs_far.sum(dim=-1) + 1e-8
    centered = energy.float() - energy.float().mean(dim=-1, keepdim=True)
    causal = make_causal_block_mask(qb, kb, device)[None, None, :, :]
    soft_mask = torch.sigmoid(centered / max(float(args.temperature), 1e-6)) * causal.float()
    far_rec = (bs_far * soft_mask).sum(dim=-1) / denom_far
    valid_far = denom_far > 1e-7
    soft_far = far_rec[valid_far].mean() if valid_far.any() else far_rec.mean()

    loss = (
        F.relu(float(args.guard_tail_min) - soft_tail).pow(2)
        + F.relu(float(args.guard_p10_min) - soft_p10).pow(2)
        + F.relu(float(args.guard_far_min) - soft_far).pow(2)
    )
    return loss, {
        "soft_tail_recall": soft_tail.detach(),
        "soft_p10_recall": soft_p10.detach(),
        "soft_far_recall": soft_far.detach(),
        "soft_recall_guard_loss": loss.detach(),
    }


@dataclass
class ForwardResult:
    loss: torch.Tensor
    stats: Dict[str, torch.Tensor]


def guarded_forward_layer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_weight: torch.Tensor,
    target_groups: List[List[int]],
    aggregation_groups: List[List[int]],
    loss_type: str,
    args,
    step: int,
) -> ForwardResult:
    q_heads = q.shape[1]
    k = repeat_kv_to_q_heads(k, q_heads)
    v = repeat_kv_to_q_heads(v, q_heads)

    with torch.no_grad():
        dense_out, dense_probs, dense_scores = dense_attention_teacher(q, k, v)
        del dense_probs

        # Inference-matched training path:
        # Full no-spaced-sampling block score on the FULL padded block map.
        # Then trainable 7x7 conv is applied on the full map and find_blocks_chunked
        # is used exactly like Conv_prefill.
        block_scores_full, score_meta = kernels_conv_block_scores_infer_full(
            q,
            k,
            block_size=args.block_size,
            stride=args.score_stride,
            norm=args.score_norm,
            chunk_size=args.score_chunk_size,
            sample_kernel_size=args.score_sample_kernel_size,
            causal=True,
        )

    q_real_blocks = score_meta["q_real_blocks"]
    k_real_blocks = score_meta["k_real_blocks"]

    # Apply conv on the FULL padded map, matching inference.
    energy_full = apply_conv_energy(block_scores_full.detach(), layer_weight)

    # Crop real block region for differentiable loss/statistics and soft surrogate.
    block_scores = block_scores_full[:, :, :q_real_blocks, :k_real_blocks].contiguous()
    energy = energy_full[:, :, :q_real_blocks, :k_real_blocks].contiguous()

    with torch.no_grad():
        hard_mask = make_inference_chunked_block_mask(
            energy_full.detach(),
            threshold=args.threshold,
            q_real_blocks=q_real_blocks,
            k_real_blocks=k_real_blocks,
            num_blocks_per_chunk=score_meta["num_blocks_per_chunk"],
            causal=True,
        )
        hard_out = hard_block_sparse_attention(q, k, v, hard_mask, block_size=args.block_size)

    soft_out = soft_surrogate_attention(
        q,
        k,
        v,
        dense_scores.detach(),
        energy,
        block_size=args.block_size,
        temperature=args.temperature,
    )
    pred = hard_out + (soft_out - soft_out.detach())

    dense_mse = F.mse_loss(pred.float(), dense_out.float())
    block_kl = block_distribution_kl(energy, block_scores.detach())

    rank = energy.new_tensor(0.0)
    agg = energy.new_tensor(0.0)
    if loss_type == "ranking":
        rank = ranking_loss(
            energy,
            target_groups,
            margin=args.target_margin,
            joint_weight=args.target_joint_weight,
            target_query_tail_blocks=args.target_query_tail_blocks,
        )
    elif loss_type == "aggregation":
        agg = aggregation_loss(
            energy,
            aggregation_groups,
            target_query_tail_blocks=args.target_query_tail_blocks,
            cover_mass_target=args.aggregation_cover_mass,
        )
    elif loss_type == "dense":
        pass
    else:
        # mixed/unknown: use whatever supervision is present
        if target_groups:
            rank = ranking_loss(energy, target_groups, args.target_margin, args.target_joint_weight, args.target_query_tail_blocks)
        if aggregation_groups:
            agg = aggregation_loss(energy, aggregation_groups, args.target_query_tail_blocks, args.aggregation_cover_mass)

    ent = entropy_loss(energy)
    comp_w = linear_weight(args.compression_loss_weight, args.compression_loss_weight_final, step, args.steps)
    budget = current_budget(args._budget_schedule, step)
    budget_loss = effective_budget_loss(energy, budget)
    soft_guard, soft_guard_stats = soft_recall_guard_loss(energy, block_scores.detach(), args)

    # Dense guard is scale-normalized. The old relu(dense_mse - max)^2 is too
    # weak when dense_mse has already become very large, e.g. 0.2 vs max 0.03.
    dense_guard = F.relu(dense_mse / max(float(args.guard_mse_max), 1e-8) - 1.0).pow(2)

    loss = (
        args.dense_mse_weight * dense_mse
        + args.dense_block_kl_weight * block_kl
        + args.target_loss_weight * rank
        + args.aggregation_loss_weight * agg
        + comp_w * ent
        + args.budget_loss_weight * budget_loss
        + args.guard_loss_weight * (dense_guard + args.soft_recall_guard_weight * soft_guard)
    )

    stats = compute_proxy_stats(
        block_scores=block_scores,
        hard_mask=hard_mask,
        target_groups=target_groups or aggregation_groups,
        target_query_tail_blocks=args.target_query_tail_blocks,
        far_distance_blocks=args.far_distance_blocks,
    )
    stats.update(
        {
            "dense_mse": dense_mse.detach(),
            "block_kl": block_kl.detach(),
            "rank_loss": rank.detach(),
            "aggregation_loss": agg.detach(),
            "entropy_loss": ent.detach(),
            "compression_weight": torch.tensor(comp_w, device=energy.device),
            "budget_loss": budget_loss.detach(),
            "soft_effective_blocks": soft_effective_blocks(energy).detach(),
            "dense_guard_loss": dense_guard.detach(),
            "selector_budget": torch.tensor(float(budget), device=energy.device),
            **soft_guard_stats,
            "energy_mean": energy.mean().detach(),
            "energy_std": energy.std().detach(),
        }
    )
    return ForwardResult(loss=loss, stats=stats)


# ----------------------------- EMA -----------------------------


class EMA:
    def __init__(self, module: BoundedResidualConvKernel, decay: float):
        self.decay = float(decay)
        self.shadow = module.effective_weight()

    @torch.no_grad()
    def update(self, module: BoundedResidualConvKernel):
        w = module.effective_weight()
        self.shadow.mul_(self.decay).add_(w, alpha=1.0 - self.decay)

    @torch.no_grad()
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.shadow.float().cpu().contiguous(), path)


# ----------------------------- main -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init_path", required=True)

    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--layer_sample_size", type=int, default=4)
    parser.add_argument("--layer_idx", type=int, default=-1)

    parser.add_argument("--min_seq_length", type=int, default=4096)
    parser.add_argument("--max_seq_length", type=int, default=9448)
    parser.add_argument("--skip_length_filter", action="store_true")
    parser.add_argument("--max_dataset_samples", type=int, default=-1)

    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--fallback_topk", type=int, default=16)
    parser.add_argument("--selector_mode", choices=["topp", "top_p", "fixed_topk", "topk"], default="topp")
    parser.add_argument("--min_topk", type=int, default=1)
    parser.add_argument("--use_budget_cap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Block-score estimator used during training. These defaults keep your existing
    # launch command unchanged while switching block_scores to kernels_conv.
    parser.add_argument("--score_stride", type=int, default=16)
    parser.add_argument(
        "--score_chunk_size",
        type=int,
        default=0,
        help="0 selects Conv_prefill's per-sequence automatic chunk size.",
    )
    parser.add_argument(
        "--score_sample_kernel_size",
        type=int,
        default=7,
        help="Compatibility only; Conv.py's no-spaced estimator ignores it.",
    )
    parser.add_argument("--score_norm", type=float, default=1.0)

    parser.add_argument("--dense_mse_weight", type=float, default=1.0)
    parser.add_argument("--dense_block_kl_weight", type=float, default=0.05)
    parser.add_argument("--target_loss_weight", type=float, default=0.04)
    parser.add_argument("--target_margin", type=float, default=0.2)
    parser.add_argument("--target_joint_weight", type=float, default=0.8)
    parser.add_argument("--target_query_tail_blocks", type=int, default=20)
    parser.add_argument("--aggregation_loss_weight", type=float, default=0.03)
    parser.add_argument("--aggregation_cover_mass", type=float, default=0.20)
    parser.add_argument("--compression_loss_weight", type=float, default=0.001)
    parser.add_argument("--compression_loss_weight_final", type=float, default=0.01)
    parser.add_argument("--budget_loss_weight", type=float, default=0.01)
    parser.add_argument("--target_blocks_schedule", default="0:16,2000:14,4000:12,6000:10")

    parser.add_argument("--guard_p10_min", type=float, default=0.90)
    parser.add_argument("--guard_tail_min", type=float, default=0.92)
    parser.add_argument("--guard_far_min", type=float, default=0.92)
    parser.add_argument("--guard_mse_max", type=float, default=0.03)
    parser.add_argument("--guard_loss_weight", type=float, default=10.0)
    parser.add_argument("--soft_recall_guard_weight", type=float, default=1.0)
    parser.add_argument("--reject_bad_updates", action="store_true")
    parser.add_argument("--hard_reject_p10", type=float, default=0.50)
    parser.add_argument("--hard_reject_tail", type=float, default=0.60)
    parser.add_argument("--hard_reject_mse", type=float, default=0.20)

    parser.add_argument("--bounded_delta_alpha", type=float, default=0.12)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--max_grad_norm", type=float, default=0.05)
    parser.add_argument("--far_distance_blocks", type=int, default=16)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--proxy_eval_steps", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required.")

    print("=" * 80)
    print("RULER-Mix guarded conv-kernel training")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.max_dataset_samples > 0:
        ds = ds.select(range(min(args.max_dataset_samples, len(ds))))
    if not args.skip_length_filter:
        ds = filter_dataset_by_truncated_length(ds, tokenizer, args.min_seq_length, args.max_seq_length)
    print(f"Final dataset size: {len(ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    conv = BoundedResidualConvKernel(
        init_path=args.init_path,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kernel_size=args.kernel_size,
        alpha=args.bounded_delta_alpha,
    ).to(device)
    conv.train()
    optimizer = torch.optim.AdamW([conv.delta], lr=args.lr, weight_decay=0.0)
    ema = EMA(conv, decay=args.ema_decay)
    schedule = parse_target_blocks_schedule(args.target_blocks_schedule)
    args._budget_schedule = schedule
    print(f"[import] conv_sparse_ops = {_conv_sparse_ops_module.__file__}")
    print(f"[import] conv_kernel     = {_conv_kernel_module.__file__}")

    indices = list(range(len(ds)))
    random.shuffle(indices)
    ptr = 0
    running: Dict[str, float] = {}
    running_count = 0
    rejected = 0

    def add_stats(loss_val: float, stats: Dict[str, torch.Tensor]):
        nonlocal running_count
        running_count += 1
        running["loss"] = running.get("loss", 0.0) + loss_val
        for name, val in stats.items():
            if torch.is_tensor(val):
                running[name] = running.get(name, 0.0) + float(val.detach().float().cpu().item())

    for step in range(1, args.steps + 1):
        if ptr >= len(indices):
            random.shuffle(indices)
            ptr = 0
        example = ds[indices[ptr]]
        ptr += 1

        text = format_example_to_text(example, tokenizer)
        tok = tokenize_one(tokenizer, text, args.min_seq_length, args.max_seq_length, device)
        if tok is None:
            continue
        input_ids, attention_mask, position_ids, offsets = tok
        seq_len = input_ids.shape[1]
        meta = example.get("meta", {}) or {}
        task_type = str(meta.get("task_type", "unknown"))
        loss_type = str(meta.get("loss_type", "ranking"))

        target_groups = resolve_block_groups(text, offsets, meta, "target_record_spans", args.block_size, seq_len)
        aggregation_groups = resolve_block_groups(text, offsets, meta, "aggregation_spans", args.block_size, seq_len)
        if loss_type == "ranking" and not target_groups:
            # fall back to dense-only if target markers were truncated
            loss_type = "dense"
        if loss_type == "aggregation" and not aggregation_groups:
            loss_type = "dense"

        with torch.no_grad():
            hidden_states = get_all_hidden_inputs(model, input_ids, attention_mask)

        if args.layer_idx >= 0:
            layers = [args.layer_idx]
        else:
            k = min(max(1, args.layer_sample_size), args.num_layers)
            layers = random.sample(range(args.num_layers), k=k)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        merged_stats: Dict[str, torch.Tensor] = {}
        for layer_idx in layers:
            q, k_, v = extract_llama_qkv(model, hidden_states[layer_idx], position_ids, layer_idx)
            layer_weight = conv.get_layer_weight(layer_idx)
            res = guarded_forward_layer(
                q=q,
                k=k_,
                v=v,
                layer_weight=layer_weight,
                target_groups=target_groups,
                aggregation_groups=aggregation_groups,
                loss_type=loss_type,
                args=args,
                step=step,
            )
            total_loss = total_loss + res.loss / len(layers)
            for name, val in res.stats.items():
                merged_stats[name] = merged_stats.get(name, torch.zeros_like(val)) + val.detach() / len(layers)

        # Reject before optimizer.step so bad batches do not pollute either the
        # trainable delta or the EMA shadow. This uses hard catastrophic
        # thresholds; the softer guard thresholds are already in the loss through
        # soft_recall_guard_loss.
        bad = False
        if args.reject_bad_updates:
            p10 = float(merged_stats.get("p10_mass_recall", torch.tensor(1.0)).float().cpu())
            tail = float(merged_stats.get("tail_mass_recall", torch.tensor(1.0)).float().cpu())
            mse = float(merged_stats.get("dense_mse", torch.tensor(0.0)).float().cpu())
            if (p10 < args.hard_reject_p10 and tail < args.hard_reject_tail) or (mse > args.hard_reject_mse):
                bad = True
                rejected += 1

        if bad:
            grad_norm = torch.tensor(0.0, device=device)
            optimizer.zero_grad(set_to_none=True)
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([conv.delta], args.max_grad_norm)
            optimizer.step()
            ema.update(conv)

        merged_stats["grad_norm"] = torch.tensor(float(grad_norm), device=device)
        merged_stats["current_block_budget"] = torch.tensor(float(current_budget(schedule, step)), device=device)
        merged_stats["rejected"] = torch.tensor(float(1 if bad else 0), device=device)
        add_stats(float(total_loss.detach().cpu()), merged_stats)

        if step % args.log_steps == 0:
            denom = max(1, running_count)
            avg_parts = {k: v / denom for k, v in running.items()}
            keys = [
                "loss", "causal_density", "selected_blocks", "mass_recall", "tail_mass_recall",
                "p10_mass_recall", "far_mass_recall", "target_hit_mean", "target_hit_any_head",
                "same_head_all_targets_hit", "dense_mse", "block_kl", "rank_loss",
                "aggregation_loss", "entropy_loss", "compression_weight", "budget_loss",
                "soft_effective_blocks", "dense_guard_loss", "soft_recall_guard_loss",
                "soft_tail_recall", "soft_p10_recall", "soft_far_recall", "selector_budget",
                "energy_mean", "energy_std", "grad_norm", "current_block_budget", "rejected",
            ]
            metric_str = " ".join([f"{k}={avg_parts[k]:.6f}" for k in keys if k in avg_parts])
            layer_desc = ",".join(str(x) for x in layers)
            print(
                f"step={step:05d} layers={layer_desc} seq_len={seq_len} task={task_type} "
                f"loss_type={loss_type} target_groups={target_groups[:6]} agg_groups={len(aggregation_groups)} "
                f"rejected_total={rejected} {metric_str}"
            )
            running.clear()
            running_count = 0

        if step % args.save_steps == 0:
            conv.save(args.out)
            step_path = args.out.replace(".pt", f"_step{step}.pt")
            ema_path = args.out.replace(".pt", f"_ema_step{step}.pt")
            conv.save(step_path)
            ema.save(ema_path)
            print(f"[save latest] {args.out}")
            print(f"[save step] {step_path}")
            print(f"[save ema]  {ema_path}")
            conv.print_kernel(layer_idx=0, head_idx=0)

    conv.save(args.out)
    ema.save(args.out.replace(".pt", "_ema.pt"))
    print(f"[final save] {args.out}")
    print(f"[final ema]  {args.out.replace('.pt', '_ema.pt')}")


if __name__ == "__main__":
    main()
