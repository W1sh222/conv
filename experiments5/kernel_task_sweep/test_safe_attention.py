import unittest

try:
    import torch
except ImportError:
    torch = None

from safe_attention import masked_sdpa


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class ReferenceAttentionTests(unittest.TestCase):
    def test_matches_full_mask_oracle_with_partial_last_block(self):
        torch.manual_seed(42)
        n, heads, dim = 259, 3, 8
        q, k, v = [torch.randn(n, heads, dim) for _ in range(3)]
        mask = torch.ones(1, heads, 3, 3, dtype=torch.bool).tril()
        mask[0, 0, 2, 1] = False
        mask[0, 1, 1, 0] = False
        indices = torch.arange(n)
        token_mask = mask[0][:, indices//128][:, :, indices//128]
        token_mask &= indices[None, None, :] <= indices[None, :, None]
        logits = torch.einsum('qhd,khd->hqk', q, k) / dim**0.5
        expected = torch.einsum('hqk,khd->qhd', logits.masked_fill(~token_mask, float('-inf')).softmax(-1), v)
        for chunk in (1, 2, 3):
            actual = masked_sdpa(q, k, v, mask, torch, head_chunk=chunk)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_rejects_empty_rows(self):
        q = torch.randn(5, 1, 8)
        with self.assertRaisesRegex(ValueError, 'empty token row'):
            masked_sdpa(q, q, q, torch.zeros(1, 1, 1, 1, dtype=torch.bool), torch)


if __name__ == '__main__':
    unittest.main()
