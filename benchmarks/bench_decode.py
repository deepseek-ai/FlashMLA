#!/usr/bin/env python
"""MLA decode benchmark harness (RTX 3090 / sm_86).

Benchmarks *whichever* `flash_mla` package is first on PYTHONPATH, e.g. the kernel worktree:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    PYTHONPATH=/data2/LiuJun/infra/flashmla-wt/kernel \\
    python /data2/LiuJun/infra/flashmla-wt/sys/benchmarks/bench_decode.py --preset main --impls pkg,legacy,triton

Implementations (--impls, comma separated):
    pkg     flash_mla.flash_mla_with_kvcache with the package's default impl
    legacy  same call with impl="legacy" (fork's sm80 kernel; page size 32 only)
    v2      same call with impl="v2" (new sm86 kernel)
    triton  vendored vLLM/SGLang grouped Triton decode attention (benchmarks/baselines), best-of num_kv_splits
    (FlashInfer runs in the vllm env: benchmarks/bench_flashinfer.py, same CSV format)

Modes (--modes):
    eager   one attention call, L2 flushed, CUDA-event time incl. launch overhead (median of many reps)
    graph   CUDA Graph with N layers (default 27 = DeepSeek-V2-Lite), every layer with its OWN KV cache; time/layer

Interface assumed from the kernel package (unsupported kwargs are detected with inspect and dropped):
    get_mla_metadata(cache_seqlens, num_q_tokens_per_head_k, num_heads_k)
    flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata, num_splits,
                           softmax_scale=None, causal=False, out=None, lse=None, workspace=None, impl="v2", kv_scale=None)

Output: one CSV row per (case, impl, mode) in /data2/LiuJun/tmp/flashmla/logs (see common.FIELDS), plus a pivot on stdout.
"""
import argparse
import inspect
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from common import (D, DV, Case, CsvOut, Data, case_row, case_stats, cos_diff, default_out, dtype_of, make_seqlens, metrics,
                    parse_int_list, preset, ref_attention, smi_sample, time_events, time_graph_replay)


# ----------------------------------------------------------------------------------------------------------------------
# implementations
# ----------------------------------------------------------------------------------------------------------------------
class FlashMLAImpl:
    def __init__(self, name, impl_kw):
        self.name, self.impl_kw = name, impl_kw
        import flash_mla as fm
        self.fm = fm
        params = inspect.signature(fm.flash_mla_with_kvcache).parameters
        self.has_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        self.params = params
        self.prealloc = True
        self.use_ws = True

    def accepts(self, k):
        return self.has_kw or k in self.params

    def unsupported(self, case):
        # A package without workspace= (legacy interface) only has the fork's kernel: page 32, bf16. Its FLASH_ASSERT would
        # exit() the whole process, so screen for it here.
        legacy_pkg = self.impl_kw in ("legacy", "legacy_ws") or (self.impl_kw is None and not self.accepts("workspace"))
        if legacy_pkg and (case.page != 32 or case.dtype != "bfloat16" or case.kv != "native"):
            return "legacy kernel: page_size==32, bf16, native KV only"
        if self.impl_kw is not None and not self.accepts("impl"):
            return f"package has no impl= kwarg (impl={self.impl_kw})"
        if case.kv != "native" and not self.accepts("kv_scale"):
            return "package has no kv_scale= kwarg"
        return None

    def prepare(self, case, data):
        meta, nsplit = self.fm.get_mla_metadata(data.sl, case.s_q * case.h_q, 1)
        st = dict(meta=meta, nsplit=nsplit, ws=None)
        # persistent workspace (partial O / LSE / combine counters) -> the call allocates nothing and is graph safe
        if self.use_ws and self.accepts("workspace") and hasattr(self.fm, "alloc_mla_workspace") and self.impl_kw in (None, "v2"):
            st["ws"] = self.fm.alloc_mla_workspace(case.b, case.s_q * case.h_q, 1, meta.shape[0])
        return st

    def meta_fn(self, case, data):
        return lambda: self.fm.get_mla_metadata(data.sl, case.s_q * case.h_q, 1)

    def runner(self, case, data, state, layer):
        kw = {}
        if self.impl_kw is not None:
            kw["impl"] = self.impl_kw
        if self.prealloc and self.accepts("out") and self.accepts("lse"):
            kw["out"] = torch.empty(case.b, case.s_q, case.h_q, DV, device="cuda", dtype=dtype_of(case.dtype))
            kw["lse"] = torch.empty(case.b, case.h_q, case.s_q, device="cuda", dtype=torch.float32)
        if state.get("ws") is not None:
            kw["workspace"] = state["ws"]
        if case.kv != "native":
            kw["kv_scale"] = data.kscale[layer]
            kc = data.kq[layer]
        else:
            kc = data.kc[layer]
        q, bt, sl, meta, nsplit, causal = data.q[layer], data.bt, data.sl, state["meta"], state["nsplit"], case.causal
        f = self.fm.flash_mla_with_kvcache

        def fn():
            return f(q, kc, bt, sl, DV, meta, nsplit, causal=causal, **kw)

        return fn

    def normalize(self, case, res):
        out, lse = res
        return out, lse  # [b, s_q, h_q, DV], [b, h_q, s_q]


class TritonImpl:
    """vLLM/SGLang grouped decode attention (stage 1 + stage 2). s_q > 1 is folded into the head dim (shares the KV loads
    like FlashMLA does, but the causal mask on the last key is ignored: timing only, no correctness check for s_q > 1)."""
    name = "triton"
    CANDIDATES = (2, 4, 8, 16, 32, 64, 128)
    MAX_LOGITS_BYTES = 1.2e9

    def __init__(self):
        from baselines.triton_decode_attention import decode_attention_fwd
        self.f = decode_attention_fwd

    def unsupported(self, case):
        if case.kv != "native":
            return "triton baseline has no quantized KV"
        return None

    def _make(self, case, data, layer, splits):
        b, hh = case.b, case.s_q * case.h_q
        dt = dtype_of(case.dtype)
        q = data.q[layer].reshape(b, hh, D)
        kc = data.kc[layer]
        vb = kc[..., :DV]
        o = torch.empty(b, hh, DV, device="cuda", dtype=dt)
        lse = torch.empty(b, hh, device="cuda", dtype=torch.float32)
        logits = torch.empty(b, hh, splits, DV + 1, device="cuda", dtype=torch.float32)
        bt, sl, page, scale = data.bt, data.sl, case.page, D ** -0.5
        ks = torch.ones((), dtype=torch.float32, device="cuda")  # created outside: a per-call torch.tensor() is not graph-capturable
        f = self.f

        def fn():
            f(q, kc, vb, o, lse, bt, sl, logits, splits, scale, page, k_scale=ks, v_scale=ks)
            return o, lse

        return fn

    def prepare(self, case, data):
        b, hh = case.b, case.s_q * case.h_q
        best = None
        for s in self.CANDIDATES:
            if b * hh * s * (DV + 1) * 4 > self.MAX_LOGITS_BYTES:
                continue
            fn = self._make(case, data, 0, s)
            fn()
            torch.cuda.synchronize()
            t = time_events(fn, warmup=3, target_ms=30, min_reps=8, max_reps=30)["median"]
            if best is None or t < best[0]:
                best = (t, s)
        return dict(splits=best[1])

    def meta_fn(self, case, data):
        return None

    def runner(self, case, data, state, layer):
        return self._make(case, data, layer, state["splits"])

    def extra(self, state):
        return f"num_kv_splits={state['splits']}"

    def normalize(self, case, res):
        o, lse = res
        b = case.b
        return o.view(b, case.s_q, case.h_q, DV), lse.view(b, case.s_q, case.h_q).permute(0, 2, 1)


def make_impl(name):
    if name == "pkg":
        return FlashMLAImpl("pkg", None)
    if name in ("legacy", "v2", "legacy_ws"):
        return FlashMLAImpl(name, name)
    if name == "triton":
        return TritonImpl()
    raise ValueError(f"unknown impl {name}")


# ----------------------------------------------------------------------------------------------------------------------
def capture_graph(runners):
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            for fn in runners:
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for fn in runners:
            fn()
    torch.cuda.synchronize()
    return g


def check_correctness(case, data, impl, fn):
    if isinstance(impl, TritonImpl) and case.s_q > 1:
        return None, None
    out, lse = impl.normalize(case, fn())
    co, cl = [], []
    for i in range(min(case.b, 2)):
        ref_o, ref_l = ref_attention(data.q[0][i], data.kv_of(0, i), case.causal)
        co.append(cos_diff(out[i].float(), ref_o))
        cl.append(cos_diff(lse[i].float(), ref_l))
    return max(co), max(cl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impls", default="pkg,triton")
    ap.add_argument("--modes", default="eager", help="eager,graph")
    ap.add_argument("--preset", default="smoke", help="smoke|main|matrix|varlen|longk|layer|mtp (ignored when explicit axes given)")
    ap.add_argument("--hq", default=None); ap.add_argument("--sq", default=None); ap.add_argument("--b", default=None)
    ap.add_argument("--sk", default=None); ap.add_argument("--page", default=None)
    ap.add_argument("--varlen", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--kv", default="native", choices=["native", "fp8", "int8"])
    ap.add_argument("--layers", type=int, default=27, help="layers captured per CUDA graph")
    ap.add_argument("--mem-gb", type=float, default=12.0, help="KV memory budget for the graph layers")
    ap.add_argument("--contiguous", action="store_true", help="contiguous block table instead of a random permutation")
    ap.add_argument("--no-prealloc", action="store_true", help="let the kernel allocate out/lse per call")
    ap.add_argument("--no-workspace", action="store_true", help="do not pass a persistent workspace")
    ap.add_argument("--check", action="store_true", help="cos_diff vs an fp32 reference (NaN-pads the cache tail)")
    ap.add_argument("--no-smi", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="decode")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if any(x is not None for x in (a.hq, a.sq, a.b, a.sk)):
        cases = [Case(h, sq, b, sk, p, a.varlen, a.dtype, a.kv)
                 for h in parse_int_list(a.hq or "16") for sq in parse_int_list(a.sq or "1")
                 for b in parse_int_list(a.b or "64") for sk in parse_int_list(a.sk or "4096")
                 for p in parse_int_list(a.page or "64")]
    else:
        pages = parse_int_list(a.page) if a.page else [None]  # --page alone overrides the preset's page size(s)
        cases = [Case(c.h_q, c.s_q, c.b, c.s_k, pg or c.page, c.varlen or a.varlen, a.dtype, a.kv)
                 for c in preset(a.preset) for pg in pages]
    modes = a.modes.split(",")
    torch.manual_seed(a.seed)

    import flash_mla
    print(f"[bench] flash_mla from {flash_mla.__file__}; torch {torch.__version__}; {torch.cuda.get_device_name(0)}; "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    impls = []
    for n in a.impls.split(","):
        try:
            impl = make_impl(n.strip())
            if a.no_prealloc and hasattr(impl, "prealloc"):
                impl.prealloc = False
            if a.no_workspace and hasattr(impl, "use_ws"):
                impl.use_ws = False
            impls.append(impl)
        except Exception as ex:
            print(f"[bench] impl {n} unavailable: {ex!r}", flush=True)
    path = a.out or default_out(a.tag)
    out = CsvOut(path)
    print(f"[bench] writing {path}; {len(cases)} cases x {len(impls)} impls x modes {modes}", flush=True)
    s0 = smi_sample() if not a.no_smi else None
    print(f"[bench] gpu before: {s0}", flush=True)

    for case in cases:
        n_layers = 1
        if "graph" in modes:
            max_pages = -(-int(max(make_seqlens(case, a.seed))) // case.page)
            per_layer = case.b * max_pages * case.page * (D * 2 if case.kv == "native" else 644)
            n_layers = int(max(1, min(a.layers, a.mem_gb * 1e9 // per_layer)))
        try:
            data = Data(case, n_layers, not a.contiguous, a.check, a.seed)
        except Exception as ex:
            print(f"[bench] {case.tag()}: data setup failed: {ex!r}", flush=True)
            torch.cuda.empty_cache()
            continue
        base = case_row(case, data.stats)
        for impl in impls:
            reason = impl.unsupported(case)
            if reason:
                out.write(impl=impl.name, mode="-", status="skip: " + reason, **base)
                continue
            try:
                state = impl.prepare(case, data)
                fn = impl.runner(case, data, state, 0)
                try:
                    fn()
                except (TypeError, RuntimeError) as ex:  # pre-allocated out/lse not accepted -> retry without
                    if getattr(impl, "prealloc", False):
                        impl.prealloc = False
                        fn = impl.runner(case, data, state, 0)
                        fn()
                    else:
                        raise
                torch.cuda.synchronize()
                cod, cl = check_correctness(case, data, impl, fn) if a.check else (None, None)
                meta_us = None
                mf = impl.meta_fn(case, data)
                if mf is not None:
                    meta_us = time_events(mf, warmup=3, target_ms=30, min_reps=10, max_reps=60)["median"]
                extra = impl.extra(state) if hasattr(impl, "extra") else ""
                try:  # compiled-in instance of the sm86 kernel, if the package reports it (see sweep.py)
                    import flash_mla_cuda
                    if hasattr(flash_mla_cuda, "sm86_config") and isinstance(impl, FlashMLAImpl):
                        extra = (extra + " " + str(flash_mla_cuda.sm86_config())).strip()
                except Exception:
                    pass
                for mode in modes:
                    if mode == "eager":
                        st = time_events(fn)
                        nl = 1
                    else:
                        runners = [impl.runner(case, data, state, l) for l in range(data.n_layers)]
                        g = capture_graph(runners)
                        st = time_graph_replay(g, data.n_layers)
                        nl = data.n_layers
                        del g, runners
                    smi = smi_sample() if not a.no_smi else None
                    m = metrics(case, data.stats, st["median"])
                    out.write(impl=impl.name, mode=mode, n_layers=nl, p10_us=st["p10"], p90_us=st["p90"], reps=st["reps"],
                              meta_us=meta_us if meta_us is not None else "", cos_out=cod if cod is not None else "",
                              cos_lse=cl if cl is not None else "", extra=extra,
                              sm_mhz=smi[0] if smi else "", power_w=smi[1] if smi else "", status="ok", **base, **m)
                    print(f"{case.tag():34s} {impl.name:7s} {mode:5s} {m['time_us']:9.1f} us {m['gbps']:7.1f} GB/s "
                          f"({m['pct_bw']:4.0f}%) {m['tflops']:6.2f} TF ({m['pct_tf']:3.0f}%)"
                          + (f" cos={cod:.1e}/{cl:.1e}" if cod is not None else "") + (f" {extra}" if extra else ""), flush=True)
            except Exception as ex:
                traceback.print_exc(limit=3)
                out.write(impl=impl.name, mode="-", status=f"error: {type(ex).__name__}: {str(ex)[:150]}", **base)
                print(f"{case.tag():34s} {impl.name:7s} ERROR {type(ex).__name__}: {str(ex)[:120]}", flush=True)
                try:
                    torch.cuda.synchronize()
                except Exception:
                    print("[bench] CUDA context is broken, aborting", flush=True)
                    out.close()
                    return 1
        del data
        torch.cuda.empty_cache()
    out.close()
    print(f"[bench] gpu after: {smi_sample() if not a.no_smi else None}")
    print(f"[bench] done -> {path}")
    try:
        import summarize
        summarize.pivot(path)
    except Exception as ex:
        print(f"[bench] (summary skipped: {ex!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
