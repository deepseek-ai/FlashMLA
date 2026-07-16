"""Context-length scaling: SWA (windowed) vs full causal, dense-MLA prefill (SM100).

Measures the forward-skip payoff as context grows (full causal is O(L^2); a causal
window W is ~O(W*L), so the speedup should grow with L). MLA shape h=128, d=192,
dv=128, b=1. fwd (skip-optimized) + bwd (bwd is NOT skip-optimized yet -> baseline,
shown for reference). Uses flash_attn_varlen_func with the (left,right) window API.
"""
import torch
import triton
from flash_mla import flash_attn_varlen_func

DEV = "cuda"
DT = torch.bfloat16
B, H, D, DV = 1, 128, 192, 128


def _mk(L, requires_grad):
    q = (torch.randn(B * L, H, D, device=DEV, dtype=DT) / 10).requires_grad_(requires_grad)
    k = (torch.randn(B * L, H, D, device=DEV, dtype=DT) / 10).requires_grad_(requires_grad)
    v = (torch.randn(B * L, H, DV, device=DEV, dtype=DT) / 10).requires_grad_(requires_grad)
    cu = torch.arange(0, (B + 1) * L, L, device=DEV, dtype=torch.int32)
    return q, k, v, cu


def bench(L, window, do_bwd):
    q, k, v, cu = _mk(L, do_bwd)
    ws = (window - 1, 0) if window > 0 else (-1, -1)
    scale = (D + 100) ** (-0.5)

    def fwd():
        return flash_attn_varlen_func(q, k, v, cu, cu, L, L, softmax_scale=scale,
                                      causal=True, is_varlen=False, window_size=ws)

    t_fwd = triton.testing.do_bench(lambda: fwd()[0], warmup=2, rep=3)
    t_bwd = float("nan")
    if do_bwd:
        out, _lse = fwd()   # flash_attn_varlen_func returns (out, lse)
        g = torch.randn_like(out)

        def bwd():
            q.grad = k.grad = v.grad = None
            out.backward(g, retain_graph=True)

        t_bwd = triton.testing.do_bench(bwd, warmup=2, rep=3)
    return t_fwd, t_bwd


if __name__ == "__main__":
    torch.cuda.set_device(0)
    lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    windows = [128, 512]
    print(f"# dense-MLA prefill, b={B} h={H} d={D} dv={DV}, fwd+bwd ms (bwd NOT skip-optimized)")
    print(f"{'L':>8} | {'full.fwd':>9} {'full.bwd':>9} | " +
          " | ".join(f"W{w}.fwd  spd  W{w}.bwd" for w in windows))
    for L in lengths:
        do_bwd = L <= 65536  # cap bwd memory at the longest length
        f_full, b_full = bench(L, 0, do_bwd)
        cells = []
        for w in windows:
            f_w, b_w = bench(L, w, do_bwd)
            spd = f_full / f_w if f_w > 0 else float("nan")
            cells.append(f"{f_w:7.3f} {spd:4.1f}x {b_w:7.3f}")
        print(f"{L:>8} | {f_full:9.3f} {b_full:9.3f} | " + " | ".join(cells), flush=True)
    print("BENCH_DONE", flush=True)
