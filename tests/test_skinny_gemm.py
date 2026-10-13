"""Tests of the weight-streaming skinny GEMM (csrc/fused/skinny_gemm.cu, flash_mla/skinny_gemm.py) on the decode-layer shapes:
accuracy vs an fp32 reference, every (n_tile, splits) path, strided views, run-to-run determinism, CUDA-graph replay."""
import pytest
import torch

from flash_mla import skinny_gemm as sg

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and sg.available()), reason="needs CUDA + flash_mla_fused_cuda")

H, HID, QH, NOPE, LORA, VH = 16, 2048, 192, 128, 512, 128
NQKV = H * QH + LORA + 64  # 3648


def cos_diff(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (1 - 2 * (a * b).sum() / ((a * a).sum() + (b * b).sum()).clamp_min(1e-30)).item()


def ref_bmm(x, w):  # fp32 reference: [T, G, K] x [G, N, K] -> [T, G, N]
    return torch.einsum("tgk,gnk->tgn", x.float(), w.float())


def rnd(*s, dt, std=0.05, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return (torch.randn(*s, generator=g, device="cuda") * std).to(dt)


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("T", [1, 3, 8, 13, 16, 31, 40, 64])
@pytest.mark.parametrize("op", ["qkv_a", "w_uk", "w_uv", "o_proj"])
def test_layer_shapes(op, T, dt):
    """Default plan on the four layer GEMMs, with the strided operands the layer uses."""
    if op == "qkv_a":
        x, w = rnd(T, HID, dt=dt), rnd(NQKV, HID, dt=dt, std=0.02, seed=1)
        y = sg.skinny_linear(x, w)
        ref = x.float() @ w.float().T
    elif op == "o_proj":
        x, w = rnd(T, H * VH, dt=dt), rnd(HID, H * VH, dt=dt, std=0.02, seed=1)
        y = sg.skinny_linear(x, w)
        ref = x.float() @ w.float().T
    elif op == "w_uk":  # x = q_nope, a strided view into the qkv_a output
        qkv = rnd(T, NQKV, dt=dt)
        x = qkv[:, :H * QH].view(T, H, QH)[..., :NOPE]
        w = rnd(H, LORA, NOPE, dt=dt, seed=1)  # pre-transposed W_UK [H, 512, 128]
        y = sg.skinny_bmm(x, w)
        ref = ref_bmm(x, w)
    else:  # w_uv writes straight into the [T, H * 128] o_proj input
        x = rnd(T, H, LORA, dt=dt)
        w = rnd(H, VH, LORA, dt=dt, seed=1)
        out = torch.full((T, H * VH), float("nan"), device="cuda", dtype=dt)
        y = sg.skinny_bmm(x, w, out=out.view(T, H, VH))
        assert y.data_ptr() == out.data_ptr()
        y, ref = out, ref_bmm(x, w).reshape(T, H * VH)
    assert y.dtype == dt and not torch.isnan(y).any()
    cd = cos_diff(y, ref)
    assert cd < 1e-5, cd
    tol = 1e-2 if dt == torch.bfloat16 else 2e-3
    assert (y.float() - ref).abs().max().item() <= tol * ref.abs().max().item() + 1e-3


@pytest.mark.parametrize("n_tile", [16, 32, 64])
@pytest.mark.parametrize("splits", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("T", [1, 8, 17, 64])
def test_tiles_and_splits(n_tile, splits, T):
    """Every kernel instance (n_tile x token bucket) and split-K path, incl. uneven chunk ranges (8 chunks / 3 or 5 splits)."""
    dt = torch.bfloat16
    G, N, K = 2, 128, 512
    x, w = rnd(T, G, K, dt=dt), rnd(G, N, K, dt=dt, seed=1)
    y = sg.skinny_bmm(x, w, n_tile=n_tile, splits=splits)
    assert cos_diff(y, ref_bmm(x, w)) < 1e-5
    y2 = sg.skinny_bmm(x, w, n_tile=n_tile, splits=splits)
    assert torch.equal(y, y2)
    # counters were returned to zero: a third run on other data is still right
    x3 = rnd(T, G, K, dt=dt, seed=7)
    assert cos_diff(sg.skinny_bmm(x3, w, n_tile=n_tile, splits=splits), ref_bmm(x3, w)) < 1e-5
    if splits > 1:
        assert int(sg._WS[torch.cuda.current_device()][1].abs().sum()) == 0


def test_split_invariance_of_results():
    """Different split counts give (nearly) the same answer; the same split count is bitwise reproducible."""
    dt = torch.bfloat16
    x, w = rnd(8, HID, dt=dt), rnd(HID, H * VH, dt=dt, std=0.02, seed=3)
    ys = [sg.skinny_linear(x, w, n_tile=16, splits=s) for s in (1, 2, 4, 8)]
    for y in ys[1:]:
        assert cos_diff(y, ys[0]) < 1e-5
    for s in (1, 4):
        a = sg.skinny_linear(x, w, n_tile=16, splits=s)
        b = sg.skinny_linear(x, w, n_tile=16, splits=s)
        assert torch.equal(a, b)


def test_cuda_graph_replay():
    """Captured split-K GEMMs replay correctly with new inputs (counters stay consistent across replays)."""
    dt = torch.bfloat16
    T = 8
    x = rnd(T, H * VH, dt=dt)
    w = rnd(HID, H * VH, dt=dt, std=0.02, seed=1)
    xb = rnd(T, H, LORA, dt=dt, seed=2)
    wb = rnd(H, VH, LORA, dt=dt, seed=3)
    out = torch.empty(T, HID, device="cuda", dtype=dt)
    outb = torch.empty(T, H * VH, device="cuda", dtype=dt)
    sg.skinny_linear(x, w, out=out, splits=4)   # allocate the scratch before capture
    sg.skinny_bmm(xb, wb, out=outb.view(T, H, VH), splits=2)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sg.skinny_linear(x, w, out=out, splits=4)
        sg.skinny_bmm(xb, wb, out=outb.view(T, H, VH), splits=2)
    for seed in range(3):
        x.copy_(rnd(T, H * VH, dt=dt, seed=10 + seed))
        xb.copy_(rnd(T, H, LORA, dt=dt, seed=20 + seed))
        g.replay()
        torch.cuda.synchronize()
        assert cos_diff(out, x.float() @ w.float().T) < 1e-5
        assert cos_diff(outb, ref_bmm(xb, wb).reshape(T, -1)) < 1e-5


def test_auto_plan_falls_back_to_cublas():
    """Single-matrix projections beyond 32 tokens go to cuBLAS (R-051); per-head bmm stays on the kernel up to 64."""
    assert sg.plan(33, 1, 2048, 2048) is None and sg.plan(32, 1, 2048, 2048) == (32, 1)
    assert sg.plan(64, 16, 128, 512) == (32, 1) and sg.plan(8, 16, 512, 128) == (16, 1)
    dt = torch.bfloat16
    x, w = rnd(48, HID, dt=dt), rnd(HID, H * VH, dt=dt, std=0.02, seed=1)
    out = torch.empty(48, HID, device="cuda", dtype=dt)
    y = sg.skinny_linear(x, w, out=out)
    assert y.data_ptr() == out.data_ptr() and cos_diff(out, x.float() @ w.float().T) < 1e-5


def test_rejects_bad_inputs():
    dt = torch.bfloat16
    w = rnd(64, 128, dt=dt)
    with pytest.raises(RuntimeError):
        sg.skinny_linear(rnd(65, 128, dt=dt), w, n_tile=16, splits=1)  # T > 64 (forced kernel; the auto plan uses cuBLAS)
    with pytest.raises(RuntimeError):
        sg.skinny_linear(rnd(4, 128, dt=dt), rnd(64, 96, dt=dt))  # K % 64
    with pytest.raises(RuntimeError):
        sg.skinny_linear(rnd(4, 128, dt=dt), w, n_tile=16, splits=3)  # splits > K / 64
    with pytest.raises(RuntimeError):
        sg.skinny_bmm(rnd(4, 3, 132, dt=dt)[..., :128], rnd(3, 64, 128, dt=dt))  # row stride not 16-byte aligned
