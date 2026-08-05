"""Focused SM100 regression for the persistent sparse-attention extension."""

import sys
from pathlib import Path

import torch

TEST_DIR = Path(__file__).resolve().parent
TESTS_ROOT = TEST_DIR.parent
REPO_ROOT = TESTS_ROOT.parent
sys.path.insert(0, str(TESTS_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import flash_mla
import quant


@torch.inference_mode()
def check_stock_indexer_topk() -> None:
    """The nv_dev-only fourth output must stay on the stock extension."""
    torch.manual_seed(19)
    device = torch.device("cuda")
    s_q, s_kv, h_q, d, topk = 32, 1024, 128, 512, 1024
    q = torch.randn((s_q, h_q, d), device=device, dtype=torch.bfloat16)
    kv = torch.randn((s_kv, 1, d), device=device, dtype=torch.bfloat16)
    indices = torch.randint(
        s_kv, (s_q, 1, topk), device=device, dtype=torch.int32
    )
    baseline = flash_mla.flash_mla_sparse_fwd(q, kv, indices, d ** -0.5)
    for indexer_topk in (512, 1024):
        actual = flash_mla.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            d ** -0.5,
            indexer_topk=indexer_topk,
        )
        assert len(actual) == 4
        for got, expected in zip(actual[:3], baseline):
            torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
        assert torch.isfinite(actual[3]).all()
        print("stock indexer_topk", indexer_topk, "ok")


@torch.inference_mode()
def check_prefill() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    s_q, s_kv, h_q, d, topk = 256, 1024, 128, 512, 128
    q = torch.randn((s_q, h_q, d), device=device, dtype=torch.bfloat16)
    kv = torch.randn((s_kv, 1, d), device=device, dtype=torch.bfloat16)
    indices = torch.randint(
        s_kv, (s_q, 1, topk), device=device, dtype=torch.int32
    )
    scale = d ** -0.5

    stock = flash_mla.flash_mla_sparse_fwd(q, kv, indices, scale)
    private_out = torch.empty_like(stock[0])
    private = flash_mla.flash_mla_sparse_fwd(
        q, kv, indices, scale, out=private_out, num_sms=116
    )
    psa_out = torch.empty_like(stock[0])
    psa = flash_mla.flash_mla_sparse_fwd(
        q, kv, indices, scale, out=psa_out, num_sms=116, psa=True
    )
    assert private[0].data_ptr() == private_out.data_ptr()
    assert psa[0].data_ptr() == psa_out.data_ptr()
    for label, got in (("private", private), ("psa", psa)):
        for name, actual, expected in zip(("out", "max", "lse"), got, stock):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            print(
                label,
                name,
                float((actual.float() - expected.float()).abs().max()),
            )


@torch.inference_mode()
def check_decode_ready() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    b, s_q, h_q, d, topk, block_size = 8, 1, 128, 512, 128, 64
    num_blocks = 16
    q = torch.randn(
        (b, s_q, h_q, d), device=device, dtype=torch.bfloat16
    ).clamp_(-1, 1)
    k = torch.randn(
        (num_blocks, block_size, 1, d),
        device=device,
        dtype=torch.bfloat16,
    ).clamp_(-1, 1)
    kq = quant.quantize_k_cache(
        k, quant.FP8KVCacheLayout.MODEL1_FP8Sparse
    )
    indices = torch.randint(
        num_blocks * block_size,
        (b, s_q, topk),
        device=device,
        dtype=torch.int32,
    )
    scale = d ** -0.55
    metadata, _ = flash_mla.get_mla_metadata(num_sm_parts=16)
    stock = flash_mla.flash_mla_with_kvcache(
        q,
        kq,
        None,
        None,
        512,
        metadata,
        None,
        scale,
        False,
        True,
        indices,
    )
    row_ready = torch.ones((b,), device=device, dtype=torch.int32)
    private_out = torch.empty_like(stock[0])
    ready = flash_mla.flash_mla_sparse_decode_ready(
        q,
        kq,
        indices,
        num_sm_parts=16,
        row_ready=row_ready,
        head_dim_v=512,
        softmax_scale=scale,
        out=private_out,
    )
    assert ready[0].data_ptr() == private_out.data_ptr()
    for name, actual, expected in zip(("out", "lse"), ready, stock):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        print(
            "decode",
            name,
            float((actual.float() - expected.float()).abs().max()),
        )


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("this regression requires an SM100 GPU")
    check_stock_indexer_topk()
    check_prefill()
    check_decode_ready()
    torch.cuda.synchronize()
    print("PERSISTENT_SPARSE_ATTENTION_PRIVATE_EXTENSION_OK")


if __name__ == "__main__":
    main()
