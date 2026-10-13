"""Python front end of the weight-streaming skinny GEMM (csrc/fused/skinny_gemm.cu) used by the decode layer.

    y = skinny_linear(x, w)           x [T, K], w [N, K]            -> x @ w.T            (nn.Linear without bias)
    y = skinny_bmm(x, w_t, out=...)   x [T, G, K], w_t [G, N, K]    -> y[t, g] = x[t, g] @ w_t[g].T

T (tokens) must be <= 64; x / out may be strided views (last dim contiguous).  The kernel streams the weight once with a
deep cp.async ring per CTA and keeps the 3090's DRAM at ~870 GB/s while streaming, where cuBLAS launches 32-64 CTAs and
reaches 20-70 % of the read bandwidth at decode batch sizes (R-045 / R-047).  The default plan (n_tile, splits) comes from
the sweep in R-051; with it the four DeepSeek-V2-Lite layer GEMMs take 41.5 us at b=8 (cuBLAS 61.7 us).  For the
single-matrix projections cuBLAS is faster beyond 32 tokens, so skinny_linear falls back to it there (unless a tile
configuration is forced).  Results are deterministic (fixed summation order) and CUDA-graph safe; with splits > 1 the
scratch and counters are allocated once per device on first use (do that before graph capture) and the counters are
always left at zero, so GEMMs sharing one device must run stream-ordered.
"""
import torch
import torch.nn.functional as F

try:
    import flash_mla_fused_cuda as _ext
except ImportError:  # extension not built
    _ext = None

MAX_T = 64
_WS = {}       # device index -> (fp32 scratch, int32 counters)
_KEEP = []     # outgrown scratch buffers stay alive: a captured CUDA graph may still point at them
_NSM = {}


def available():
    return _ext is not None and hasattr(_ext, "skinny_gemm")


def supports(x):
    return available() and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and 1 <= x.shape[0] <= MAX_T


def _num_sms(dev):
    if dev not in _NSM:
        _NSM[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _NSM[dev]


def plan(T, G, N, K, num_sms=82):
    """(n_tile, splits), or None when cuBLAS is the better choice (single matrix, T > 32).
    R-051 (RTX 3090, T = 1..64): split-K never paid off (one wave of deep-ring CTAs already streams at ~870 GB/s, extra
    CTAs only add fixed cost), n_tile = 32 was best or within 2 %, 16 for the per-head matrices at T <= 16."""
    if G == 1 and T > 32:
        return None
    n_tile = 16 if (G > 1 and T <= 16) else 32
    while N % n_tile:
        n_tile //= 2
    return n_tile, 1


def _scratch(dev, floats, ctrs):
    ws, ct = _WS.get(dev, (None, None))
    if ws is None or ws.numel() < floats:
        if ws is not None:
            _KEEP.append(ws)
        ws = torch.empty(max(floats, 1 << 20), dtype=torch.float32, device=dev)
    if ct is None or ct.numel() < ctrs:
        if ct is not None:
            _KEEP.append(ct)
        ct = torch.zeros(max(ctrs, 4096), dtype=torch.int32, device=dev)
    _WS[dev] = (ws, ct)
    return ws, ct


def skinny_bmm(x, w, out=None, n_tile=None, splits=None):
    """out[t, g, :] = x[t, g, :] @ w[g].T   x [T, G, K] (any 16-byte aligned t/g strides), w [G, N, K] contiguous."""
    T, G, K = x.shape
    N = w.shape[1]
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if n_tile is None or splits is None:
        pl = plan(T, G, N, K, _num_sms(dev))
        if pl is None:  # cuBLAS
            res = torch.bmm(x.transpose(0, 1), w.transpose(1, 2)).transpose(0, 1)
            return out.copy_(res) if out is not None else res
        n_tile, splits = n_tile or pl[0], splits or pl[1]
    if out is None:
        out = torch.empty(T, G, N, dtype=x.dtype, device=x.device)
    ws = ct = None
    if splits > 1:
        ws, ct = _scratch(dev, G * N * splits * T, G * (N // n_tile))
    _ext.skinny_gemm(x, w, out, n_tile, splits, ws, ct)
    return out


def skinny_linear(x, w, out=None, n_tile=None, splits=None):
    """x [T, K] @ w[N, K].T -> [T, N] (out: optional [T, N] view with contiguous last dim)."""
    if (n_tile is None or splits is None) and plan(x.shape[0], 1, w.shape[0], w.shape[1]) is None:  # cuBLAS
        return F.linear(x, w) if out is None else out.copy_(F.linear(x, w))
    y = skinny_bmm(x.unsqueeze(1), w.unsqueeze(0), None if out is None else out.unsqueeze(1), n_tile, splits)
    return y.squeeze(1)
