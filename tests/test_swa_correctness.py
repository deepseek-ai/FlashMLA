"""Numerical correctness for the causal sliding-window (SWA) dense-MLA path.

Reuses the verified fwd+bwd checker in test_fmha_sm100.py (compares flash_mla's
flash_attn_varlen_func against an SDPA reference whose attn_bias already encodes
the (left,right) window via get_attn_bias). We drive it with window>0, causal,
for both the MLA shape (d=192, dv=128) and the generic shape (128/128), fixed
and varlen, fwd+bwd. window==0 is the regression check (must equal plain causal).

Run with PYTHONPATH=<repo root> and cwd=<repo>/tests.
"""
import torch
import test_fmha_sm100 as T


def run(h, h_k, d, dv, causal, window, varlen):
    # sdpa() in test_fmha_sm100 reads MODULE globals h/h_k for GQA repeat.
    T.h = h
    T.h_k = h_k
    has_bwd = (h == h_k)  # the kernel path rejects GQA backward
    T.test_flash_attention(
        b=2, mean_sq=2048, mean_sk=2048, varlen=varlen,
        h=h, h_k=h_k, d=d, dv=dv, causal=causal, window=window,
        has_bwd=has_bwd, check_correctness=True,
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    dev = torch.device("cuda:0")
    torch.set_default_device(dev)
    torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")

    print("===== regression: window=0 must match plain causal =====", flush=True)
    run(128, 128, 192, 128, causal=True, window=0, varlen=False)
    run(128, 128, 128, 128, causal=True, window=0, varlen=False)

    print("===== causal sliding window (W in {128, 384}) =====", flush=True)
    for (d, dv) in [(192, 128), (128, 128)]:
        for window in [128, 384]:
            for varlen in [False, True]:
                run(128, 128, d, dv, causal=True, window=window, varlen=varlen)

    print("ALL_SWA_TESTS_PASSED", flush=True)
