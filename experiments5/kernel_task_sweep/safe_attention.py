"""Bounded-workspace reference attention for diagnostic block masks.

Uses ordinary PyTorch SDPA rather than the custom block-sparse CUDA extension.
No full N x N token mask is allocated. This is for NLL analysis, not timing.
"""

def masked_sdpa(q, k, v, block_mask, torch, block_size=128, head_chunk=1):
    # Inputs use the extension's [tokens, heads, dim] layout.
    n, heads, dim = q.shape
    nk = k.shape[0]
    if n != nk or block_mask.shape != (1, heads, (n+block_size-1)//block_size,
                                       (nk+block_size-1)//block_size):
        raise ValueError('Reference attention requires matching self-attention lengths/mask')
    if head_chunk < 1:
        raise ValueError('head_chunk must be positive')
    out = torch.empty_like(q)
    for start in range(0, n, block_size):
        end = min(start+block_size, n)
        # All keys after end are causally invisible, so omit them entirely.
        keys = torch.arange(end, device=q.device)
        queries = torch.arange(start, end, device=q.device)
        causal = keys[None, :] <= queries[:, None]
        for h in range(0, heads, head_chunk):
            hh = min(h+head_chunk, heads)
            allowed = block_mask[0, h:hh, start//block_size, keys//block_size]
            allowed = allowed[:, None, :] & causal[None, :, :]
            if not allowed.any(dim=-1).all():
                raise ValueError('Reference sparse attention has an empty token row')
            values = torch.nn.functional.scaled_dot_product_attention(
                q[start:end, h:hh].transpose(0, 1).unsqueeze(0),
                k[:end, h:hh].transpose(0, 1).unsqueeze(0),
                v[:end, h:hh].transpose(0, 1).unsqueeze(0),
                attn_mask=allowed.unsqueeze(0), dropout_p=0.0, is_causal=False)
            out[start:end, h:hh] = values.squeeze(0).transpose(0, 1)
    return out
