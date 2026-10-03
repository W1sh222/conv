"""Cumulative selection for signed Conv scores, independent of XAttention."""
import math
import os
import torch


CONV_TOPP_VERSION = "positive_mass_v1"
_LOGGED_POLICIES = set()


def get_conv_topp_policy():
    policy = os.environ.get("CONV_TOPP_SELECTOR", "positive").strip().lower()
    if policy not in {"positive", "legacy"}:
        raise ValueError("CONV_TOPP_SELECTOR must be 'positive' or 'legacy'")
    if policy not in _LOGGED_POLICIES:
        version = CONV_TOPP_VERSION if policy == "positive" else "legacy_signed_sum"
        print(f"[Conv Top-p] selector={version}; scores=max(score,0)" if policy == "positive"
              else f"[Conv Top-p] selector={version}; reproducing original selection", flush=True)
        _LOGGED_POLICIES.add(policy)
    return policy


def select_conv_topp_blocks(scores, threshold, offset=0, causal=True,
                            force_sink=True, force_diagonal=True):
    """Select >= p of causal positive score mass, including forced blocks.

    The caller supplies only real query/key blocks, never chunk padding.
    Positive mass is a score convention, not a claim about true attention
    recall. Zero-mass rows conservatively retain all valid blocks. p=1 also
    retains all valid blocks, giving a dense-attention control.
    """
    if scores.ndim != 4 or scores.shape[-1] < 1:
        raise ValueError("scores must have shape [batch, heads, query_blocks, key_blocks]")
    b, h, q, k = scores.shape
    if torch.is_tensor(threshold):
        if threshold.ndim == 0:
            p = threshold.to(device=scores.device, dtype=torch.float32)
        elif threshold.ndim == 1 and threshold.numel() == h:
            p = threshold.to(device=scores.device, dtype=torch.float32)[None, :, None, None]
        else:
            raise ValueError("threshold tensor must be scalar or contain one value per head")
        if not bool(((p > 0) & (p <= 1) & torch.isfinite(p)).all()):
            raise ValueError("Top-p thresholds must be finite and in (0, 1]")
    else:
        p = float(threshold)
        if not math.isfinite(p) or not 0 < p <= 1:
            raise ValueError("Top-p threshold must be finite and in (0, 1]")

    qi = torch.arange(q, device=scores.device)[:, None]+int(offset)
    ki = torch.arange(k, device=scores.device)[None, :]
    valid = ki <= qi if causal else torch.ones((q, k), dtype=torch.bool, device=scores.device)
    valid4 = valid[None, None]
    mass = torch.nan_to_num(scores.float(), nan=0., posinf=1e4, neginf=-1e4)
    mass = mass.clamp_min(0).masked_fill(~valid4, 0)
    forced = torch.zeros((q, k), dtype=torch.bool, device=scores.device)
    if force_sink:
        forced[:, 0] = True
    if force_diagonal:
        forced = forced | (ki == qi)
    forced = forced & valid
    forced4 = forced[None, None]
    total = mass.sum(-1, keepdim=True)
    remaining = (total*p-(mass*forced4).sum(-1, keepdim=True)).clamp_min(0)

    candidates = mass.masked_fill(forced4, 0)
    values, indices = torch.sort(candidates, dim=-1, descending=True, stable=True)
    before = values.cumsum(-1)-values
    # Include the first candidate whose addition crosses the threshold.
    take = (before < remaining) & (values > 0)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected.scatter_(-1, indices, take)
    selected = (selected | forced4) & valid4
    return torch.where((total <= 0) | (p >= 1), valid4, selected).contiguous()


def select_conv_topp_mask(scores, threshold, q_real_blocks, k_real_blocks,
                         num_blocks_per_chunk, causal=True,
                         force_sink=True, force_diagonal=True):
    """Return a padded-size mask, but select only over real block dimensions."""
    q_real_blocks, k_real_blocks = int(q_real_blocks), int(k_real_blocks)
    num_blocks_per_chunk = int(num_blocks_per_chunk)
    if not (0 < q_real_blocks <= scores.shape[-2] and
            0 < k_real_blocks <= scores.shape[-1] and num_blocks_per_chunk > 0):
        raise ValueError("Real block dimensions/chunk size are incompatible with scores")
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for start in range(0, q_real_blocks, num_blocks_per_chunk):
        end = min(start+num_blocks_per_chunk, q_real_blocks)
        mask[:, :, start:end, :k_real_blocks] = select_conv_topp_blocks(
            scores[:, :, start:end, :k_real_blocks], threshold,
            offset=k_real_blocks-q_real_blocks+start, causal=causal,
            force_sink=force_sink, force_diagonal=force_diagonal,
        )
    return mask
