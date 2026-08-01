import os
import sys
import unittest

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import flash_mla
import quant


class FlashMLASM100SparseDecodeRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")

        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)
        torch.set_default_device(cls.device)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_float32_matmul_precision("high")
        cls.cc = torch.cuda.get_device_capability(cls.device)

    def _require_sm100(self):
        if self.cc[0] != 10:
            self.skipTest("This regression test exercises the SM100 sparse decode path")

    def _make_v32_fp8_kv_cache(self, num_blocks: int, block_size: int) -> torch.Tensor:
        device = self.device
        torch.manual_seed(20260510)

        k_cache = torch.empty(
            (num_blocks, block_size, 1, 656),
            dtype=torch.float8_e4m3fn,
            device=device,
        )

        nope = (torch.randn((num_blocks, block_size, 1, 512), device=device) * 0.5).clamp_(
            -2.0,
            2.0,
        )
        k_cache[..., :512] = nope.to(torch.float8_e4m3fn)

        token_ids = torch.arange(num_blocks * block_size, dtype=torch.float32, device=device).view(
            num_blocks,
            block_size,
            1,
            1,
        )
        tile_ids = torch.arange(4, dtype=torch.float32, device=device).view(1, 1, 1, 4)
        scales = k_cache[..., 512:528].view(torch.float32)
        scales.copy_(0.0137 + (token_ids % 17) * 0.00031 + tile_ids * 0.00103)

        rope = k_cache[..., 528:].view(torch.bfloat16)
        rope.copy_((torch.randn_like(rope.float()) * 0.03).to(torch.bfloat16))
        return k_cache

    def _reference_v32_sparse_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        sm_scale: float,
    ):
        batch, seqlen_q, num_heads, head_dim = q.shape
        head_dim_v = 512
        topk = indices.shape[-1]

        kv = quant.dequantize_k_cache(
            k_cache,
            quant.FP8KVCacheLayout.V32_FP8Sparse,
        ).view(-1, head_dim)
        physical_tokens = kv.shape[0]

        invalid = (indices < 0) | (indices >= physical_tokens)
        invalid |= torch.arange(topk, device=q.device).view(1, 1, topk) >= topk_length.view(
            batch,
            1,
            1,
        )

        fixed_indices = indices.clamp(0, physical_tokens - 1)
        gathered = kv.index_select(0, fixed_indices.reshape(-1)).view(
            batch,
            seqlen_q,
            topk,
            head_dim,
        ).float()

        logits = torch.einsum("bshd,bstd->bsht", q.float(), gathered)
        logits *= sm_scale
        logits.masked_fill_(invalid.view(batch, seqlen_q, 1, topk), float("-inf"))

        lse = torch.logsumexp(logits, dim=-1)
        probs = torch.exp(logits - lse.unsqueeze(-1))
        out = torch.einsum("bsht,bstd->bshd", probs, gathered[..., :head_dim_v])

        no_valid_token = lse == float("-inf")
        out[no_valid_token.unsqueeze(-1).expand_as(out)] = 0.0
        lse[no_valid_token] = float("inf")
        return out.to(torch.bfloat16), lse.transpose(1, 2)

    def test_v32_fp8_scales_and_physical_oob_indices(self):
        self._require_sm100()

        batch = 1
        seqlen_q = 2
        num_heads = 64
        head_dim = 576
        head_dim_v = 512
        topk = 64
        num_blocks = 2
        block_size = 64
        physical_tokens = num_blocks * block_size

        torch.manual_seed(129103)
        q = (torch.randn((batch, seqlen_q, num_heads, head_dim), device=self.device) * 0.08).to(
            torch.bfloat16
        )
        k_cache = self._make_v32_fp8_kv_cache(num_blocks, block_size)

        valid = torch.arange(0, 48, dtype=torch.int32, device=self.device)
        physical_oob = torch.arange(
            physical_tokens,
            physical_tokens + 12,
            dtype=torch.int32,
            device=self.device,
        )
        masked_tail = torch.tensor([7, -1, physical_tokens + 31, 19], dtype=torch.int32, device=self.device)
        row0 = torch.cat([valid, physical_oob, masked_tail])
        row1 = torch.cat([valid.flip(0), physical_oob.flip(0), masked_tail.flip(0)])
        indices = torch.stack([row0, row1]).view(batch, seqlen_q, topk)
        topk_length = torch.tensor([60], dtype=torch.int32, device=self.device)

        sched_meta, num_splits = flash_mla.get_mla_metadata()
        out, lse = flash_mla.flash_mla_with_kvcache(
            q,
            k_cache,
            None,
            None,
            head_dim_v,
            sched_meta,
            num_splits,
            head_dim ** -0.55,
            False,
            True,
            indices,
            None,
            None,
            None,
            topk_length,
            None,
        )
        ref_out, ref_lse = self._reference_v32_sparse_decode(
            q,
            k_cache,
            indices,
            topk_length,
            head_dim ** -0.55,
        )

        torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=2.01 / 128)
        torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=8.01 / 65536)


if __name__ == "__main__":
    unittest.main(verbosity=2)
