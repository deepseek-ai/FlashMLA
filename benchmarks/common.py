"""Shared pieces of the FlashMLA decode benchmarks (pure torch, so bench_flashinfer.py can import it from the vllm env).

* shape / accounting conventions: MLA decode, h_kv = 1, head dim 576 (512 latent + 64 rope), V = first 512 dims.
* bytes  = KV tokens read (sum of seqlens, bf16: 1152 B/token, quantized: 644 B/token) + Q + O.
* FLOPs  = 2 * s_q * h_q * sum(seqlens) * (576 + 512).
* peaks  = measured on the RTX 3090 (docs/perf_log_sys.md): read-only 873.8 GB/s, bf16 70.0 / fp16 66.4 TFLOPS.
"""
import csv
import math
import os
import random
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict

import torch

D, DV = 576, 512
PEAK_GBPS = float(os.environ.get("FLASHMLA_PEAK_GBPS", "873.8"))
PEAK_TFLOPS = {
    "bfloat16": float(os.environ.get("FLASHMLA_PEAK_TF_BF16", "70.0")),
    "float16": float(os.environ.get("FLASHMLA_PEAK_TF_FP16", "66.4")),
}
LOG_DIR = os.environ.get("FLASHMLA_LOG_DIR", "/data2/LiuJun/tmp/flashmla/logs")
QUANT_BYTES_PER_TOKEN = 512 + 64 * 2 + 4  # docs/kv_quant_format.md


# ----------------------------------------------------------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    h_q: int
    s_q: int
    b: int
    s_k: int          # mean (or fixed) cache length
    page: int
    varlen: bool = False
    dtype: str = "bfloat16"
    kv: str = "native"  # native | fp8 | int8

    @property
    def causal(self):
        return self.s_q > 1

    def tag(self):
        return (f"h{self.h_q}_sq{self.s_q}_b{self.b}_sk{self.s_k}_p{self.page}" + ("_var" if self.varlen else "")
                + ("" if self.kv == "native" else f"_{self.kv}"))


def make_seqlens(case: Case, seed: int = 0):
    """Fixed length, or the fork test's normal(mean, mean/2) varlen (clamped to [s_q, 2*mean])."""
    if not case.varlen:
        return [case.s_k] * case.b
    rng = random.Random(seed * 1000003 + case.b * 131 + case.s_k)
    return [min(max(int(rng.normalvariate(case.s_k, case.s_k / 2)), case.s_q), 2 * case.s_k) for _ in range(case.b)]


def dtype_of(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def case_stats(case: Case, seqlens):
    elt = 2
    tot = int(sum(seqlens))
    kv_bytes = tot * (D * elt if case.kv == "native" else QUANT_BYTES_PER_TOKEN)
    q_bytes = case.b * case.s_q * case.h_q * D * elt
    o_bytes = case.b * case.s_q * case.h_q * DV * elt
    flops = 2 * case.s_q * case.h_q * tot * (D + DV)
    return dict(s_k_total=tot, bytes=kv_bytes + q_bytes + o_bytes, kv_bytes=kv_bytes, flops=flops)


def preset(name):
    """Named shape sets (plan section 5). Returns a list of Case."""
    cs = []
    if name == "smoke":
        for h in (16, 128):
            cs.append(Case(h, 1, 4, 1024, 64))
    elif name == "main":  # the Phase-0 legacy table
        for h in (16, 128):
            for b in (1, 64, 128):
                for sk in (1024, 4096, 8192):
                    cs.append(Case(h, 1, b, sk, 64))
    elif name == "matrix":  # plan section 5: heads x s_q x page
        for h in (8, 16, 32, 64, 128):
            bs = (1, 16, 128) if h <= 32 else (1, 16, 64)
            sks = (1024, 4096, 8192) if h <= 32 else (1024, 4096)
            for sq in (1, 2):
                for page in (16, 32, 64):
                    for b in bs:
                        for sk in sks:
                            cs.append(Case(h, sq, b, sk, page))
    elif name == "full":  # integration matrix: h_q x s_q x b x s_k (+ b=1 at 32k), one page size (set with --page)
        for h in (8, 16, 32, 64, 128):
            for sq in (1, 2):
                for b in (1, 4, 16, 64, 128):
                    for sk in (1024, 4096, 8192):
                        cs.append(Case(h, sq, b, sk, 64))
                cs.append(Case(h, sq, 1, 32768, 64))
    elif name == "core":  # reduced integration matrix for the fp16 / fp8 / int8 runs: the two regimes + h_q=64
        for h in (16, 64, 128):
            for sq in (1, 2):
                for b in (1, 4, 16, 64, 128):
                    for sk in (1024, 4096, 8192):
                        cs.append(Case(h, sq, b, sk, 64))
                cs.append(Case(h, sq, 1, 32768, 64))
    elif name == "spot":  # page-size spot checks (run with --page 16 / 32)
        for h in (16, 128):
            for b, sk in ((1, 4096), (4, 4096), (64, 4096), (128, 8192)):
                cs.append(Case(h, 1, b, sk, 64))
        cs.append(Case(16, 2, 16, 4096, 64))
    elif name == "varlen":
        for h in (16, 64, 128):
            for sq in (1, 2):
                cs.append(Case(h, sq, 64, 4096, 64, varlen=True))
    elif name == "longk":  # b=1, long context (many splits)
        for h in (16, 128):
            for sk in (16384, 32768, 65536):
                cs.append(Case(h, 1, 1, sk, 64))
    elif name == "layer":  # DeepSeek-V2-Lite decode (h_q = 16): typical serving batches
        for b in (1, 8, 32, 64):
            for sk in (1024, 4096):
                cs.append(Case(16, 1, b, sk, 64))
    elif name == "mtp":
        for h in (16, 128):
            for b in (1, 16, 64):
                cs.append(Case(h, 2, b, 4096, 64))
    else:
        raise ValueError(f"unknown preset {name}")
    return cs


def parse_int_list(s):
    return [int(x) for x in str(s).split(",") if x != ""]


# ----------------------------------------------------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------------------------------------------------
_FLUSH = None


def flush_buffer(mb=256):
    """A buffer much larger than the 6 MB L2; zero_() before each rep evicts the KV cache from L2."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(mb * 1024 * 1024, dtype=torch.int8, device="cuda")
    return _FLUSH


def _stats(ts):
    ts = sorted(ts)
    n = len(ts)
    return dict(median=statistics.median(ts), p10=ts[int(0.1 * (n - 1))], p90=ts[int(0.9 * (n - 1))], min=ts[0], reps=n)


def time_events(fn, warmup=10, target_ms=250.0, min_reps=40, max_reps=500, flush=True):
    """Median wall time (us) of fn() measured with CUDA events on the GPU timeline, L2 flushed before every rep.
    The event pair brackets the *launches* too, so an eager Python call that is launch-bound reports its launch-bound time."""
    fb = flush_buffer() if flush else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    def run(n):
        st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            if fb is not None:
                fb.zero_()
            st[i].record()
            fn()
            en[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) * 1e3 for s, e in zip(st, en)]

    est = statistics.median(run(8))
    reps = int(max(min_reps, min(max_reps, target_ms * 1e3 / (est + 350.0))))
    return _stats(run(reps))


def time_graph_replay(graph, n_layers, warmup=3, target_ms=250.0, min_reps=20, max_reps=200, flush=True):
    """Per-layer time (us) of one replay of a graph holding n_layers attention calls (each on its own KV cache)."""
    fb = flush_buffer() if flush else None
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    def run(n):
        st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            if fb is not None:
                fb.zero_()
            st[i].record()
            graph.replay()
            en[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) * 1e3 / n_layers for s, e in zip(st, en)]

    est = statistics.median(run(5)) * n_layers
    reps = int(max(min_reps, min(max_reps, target_ms * 1e3 / (est + 350.0))))
    return _stats(run(reps))


def smi_sample():
    """(sm_clock_mhz, power_w, temp_c, throttle_reasons) of the first visible GPU, or None."""
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", idx, "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        v = [x.strip() for x in out.split(",")]
        return float(v[0]), float(v[1]), float(v[2]), v[3]
    except Exception:
        return None


COOL_HI = float(os.environ.get("FLASHMLA_COOL_HI", "68"))
COOL_LO = float(os.environ.get("FLASHMLA_COOL_LO", "56"))
COOL_MAX_S = float(os.environ.get("FLASHMLA_COOL_MAX_S", "60"))
_COOL_ENABLED = False


def enable_cooldown(on=True):
    global _COOL_ENABLED
    _COOL_ENABLED = on


def cooldown():
    """Thermal control (no sudo -> clocks cannot be locked): the RTX 3090 here sustains only ~1.2-1.5 GHz once it reaches its 83 C
    target under continuous load, but boosts to 1.8-1.9 GHz when cool, and it cools from 91 C to <50 C within ~20 s when idle.
    When enabled (`--cool`), block while the first visible GPU is hotter than COOL_HI until it is below COOL_LO (at most COOL_MAX_S)."""
    if not _COOL_ENABLED:
        return 0.0
    t0 = time.time()
    s = smi_sample()
    if s is None or s[2] <= COOL_HI:
        return 0.0
    while time.time() - t0 < COOL_MAX_S:
        time.sleep(1.0)
        s = smi_sample()
        if s is None or s[2] <= COOL_LO:
            break
    return time.time() - t0


# ----------------------------------------------------------------------------------------------------------------------
# reference + metrics
# ----------------------------------------------------------------------------------------------------------------------
def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def ref_attention(q, kv, causal, scale=None):
    """fp32 reference for one sequence. q: [s_q, h_q, 576]; kv: [s_k, 576] (K = all dims, V = first 512).
    Returns out [s_q, h_q, 512], lse [h_q, s_q] (natural log of the scaled scores' normalizer)."""
    q, kv = q.float(), kv.float()
    s_q, s_k = q.shape[0], kv.shape[0]
    scale = D ** -0.5 if scale is None else scale
    s = torch.einsum("qhd,kd->hqk", q, kv) * scale
    if causal:
        mask = torch.ones(s_q, s_k, dtype=torch.bool, device=q.device).tril(diagonal=s_k - s_q)
        s = s.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,kd->qhd", p, kv[:, :DV]), lse


def metrics(case, stats, time_us):
    t = time_us * 1e-6
    gbps = stats["bytes"] / t / 1e9
    tf = stats["flops"] / t / 1e12
    return dict(time_us=time_us, gbps=gbps, tflops=tf, pct_bw=100 * gbps / PEAK_GBPS,
                pct_tf=100 * tf / PEAK_TFLOPS[case.dtype])


# ----------------------------------------------------------------------------------------------------------------------
# csv
# ----------------------------------------------------------------------------------------------------------------------
FIELDS = ["impl", "mode", "dtype", "kv", "h_q", "s_q", "b", "s_k", "s_k_total", "varlen", "page", "n_layers",
          "time_us", "p10_us", "p90_us", "reps", "gbps", "tflops", "pct_bw", "pct_tf", "meta_us",
          "cos_out", "cos_lse", "extra", "sm_mhz", "power_w", "status"]


class CsvOut:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS)
        self.w.writeheader()
        self.f.flush()

    def write(self, **row):
        base = {k: "" for k in FIELDS}
        base.update({k: v for k, v in row.items() if k in FIELDS})
        for k, v in list(base.items()):
            if isinstance(v, float):
                base[k] = f"{v:.4g}" if k in ("cos_out", "cos_lse") else f"{v:.2f}"
        self.w.writerow(base)
        self.f.flush()

    def close(self):
        self.f.close()


def case_row(case: Case, stats):
    return dict(dtype=case.dtype, kv=case.kv, h_q=case.h_q, s_q=case.s_q, b=case.b, s_k=case.s_k,
                s_k_total=stats["s_k_total"], varlen=int(case.varlen), page=case.page)


def default_out(tag):
    return os.path.join(LOG_DIR, f"bench_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv")


# ----------------------------------------------------------------------------------------------------------------------
# data for one case
# ----------------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, case: Case, n_layers: int, scatter: bool, check: bool, seed: int):
        dev, dt = "cuda", dtype_of(case.dtype)
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        self.case, self.n_layers = case, n_layers
        self.sl_list = make_seqlens(case, seed)
        self.stats = case_stats(case, self.sl_list)
        self.sl = torch.tensor(self.sl_list, dtype=torch.int32, device=dev)
        max_pages = -(-max(self.sl_list) // case.page)
        nblk = case.b * max_pages
        perm = torch.randperm(nblk, device=dev, generator=g) if scatter else torch.arange(nblk, device=dev)
        self.bt = perm.view(case.b, max_pages).to(torch.int32).contiguous()
        self.nblk = nblk
        self.kc, self.q, self.kq, self.kscale = [], [], [], []
        for _ in range(n_layers):
            kc = torch.randn(nblk, case.page, 1, D, device=dev, dtype=dt, generator=g)
            if check:  # NaN-pad everything past the sequence end (page tables are in bt order)
                kf = kc.view(nblk * case.page, D)
                for i, s in enumerate(self.sl_list):
                    idx = (self.bt[i].long()[:, None] * case.page + torch.arange(case.page, device=dev)[None]).reshape(-1)
                    kf[idx[s:]] = float("nan")
            self.q.append(torch.randn(case.b, case.s_q, case.h_q, D, device=dev, dtype=dt, generator=g))
            if case.kv == "native":
                self.kc.append(kc)
            else:
                from flash_mla.kv_quant import quantize_kv, dequantize_kv  # reference implementation (SA)
                kvq, sc = quantize_kv(torch.nan_to_num(kc.view(nblk, case.page, D)), case.kv)
                self.kq.append(kvq)  # uint8 [nblk, page, 640], as in docs/kv_quant_format.md
                self.kscale.append(sc)
                # the dequantized reference copy is only needed for layer 0 (the correctness check): keeps the memory of
                # b=128, s_k=8192 caches affordable
                self.kc.append(dequantize_kv(kvq, sc, case.kv, dt).view(nblk, case.page, 1, D) if (check and len(self.kq) == 1) else None)
            del kc

    def kv_of(self, layer, i):
        """[s_k, 576] reference KV (dequantized for quantized caches) of sequence i."""
        kc = self.kc[layer]
        return kc[self.bt[i].long()].reshape(-1, D)[: self.sl_list[i]]
