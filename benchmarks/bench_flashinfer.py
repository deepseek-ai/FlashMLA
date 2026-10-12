#!/usr/bin/env python
"""FlashInfer fa2 MLA baseline, same shapes / CSV format as bench_decode.py.  Runs in the *vllm* conda env (nothing installed).

    source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data2/LiuJun/conda/envs/vllm
    export CUDA_HOME=/data2/LiuJun/cuda/cuda-12.4 PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST=8.6 \\
           FLASHINFER_WORKSPACE_BASE=/data2/LiuJun/tmp/flashmla/flashinfer_cache
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_flashinfer.py --preset main

flashinfer.mla.BatchMLAPagedAttentionWrapper(backend="fa2"): JIT-compiles on first use (several minutes, cached afterwards).
If the JIT or the kernel fails on sm_86, the failure is recorded as a CSV row and the script exits cleanly.
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from common import (D, DV, Case, CsvOut, Data, case_row, case_stats, cos_diff, default_out, make_seqlens, metrics,
                    parse_int_list, preset, ref_attention, smi_sample, time_events, time_graph_replay)


class FlashInferMLA:
    name = "flashinfer"

    def __init__(self, case: Case, data: Data, use_graph: bool):
        import flashinfer
        self.fi = flashinfer
        b, page = case.b, case.page
        dev = "cuda"
        npages = [-(-s // page) for s in data.sl_list]
        kv_indptr = torch.zeros(b + 1, dtype=torch.int32)
        kv_indptr[1:] = torch.tensor(npages).cumsum(0)
        self.kv_indptr = kv_indptr.to(dev)
        self.kv_indices = torch.cat([data.bt[i, :n] for i, n in enumerate(npages)]).to(torch.int32).contiguous()
        self.qo_indptr = (torch.arange(b + 1, dtype=torch.int32) * case.s_q).to(dev)
        self.kv_len = data.sl.clone()
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=dev)
        kw = dict(use_cuda_graph=True, qo_indptr=self.qo_indptr, kv_indptr=self.kv_indptr, kv_indices=self.kv_indices,
                  kv_len_arr=self.kv_len) if use_graph else {}
        self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(ws, backend="fa2", **kw)
        self.case, self.data = case, data
        self.plan()

    def plan(self):
        c = self.case
        dt = self.data.q[0].dtype
        self.wrapper.plan(self.qo_indptr, self.kv_indptr, self.kv_indices, self.kv_len, c.h_q, DV, D - DV, c.page,
                          c.causal, D ** -0.5, dt, dt)

    def runner(self, layer):
        c, d = self.case, self.data
        q = d.q[layer].reshape(c.b * c.s_q, c.h_q, D)
        q_nope, q_pe = q[..., :DV].contiguous(), q[..., DV:].contiguous()
        kc = d.kc[layer].view(d.nblk, c.page, D)
        ckv, kpe = kc[..., :DV], kc[..., DV:]
        o = torch.empty(c.b * c.s_q, c.h_q, DV, dtype=q.dtype, device="cuda")
        lse = torch.empty(c.b * c.s_q, c.h_q, dtype=torch.float32, device="cuda")
        w = self.wrapper

        def fn():
            w.run(q_nope, q_pe, ckv, kpe, out=o, lse=lse, return_lse=True)
            return o, lse

        return fn

    def normalize(self, res):
        c = self.case
        o, lse = res
        # FlashInfer returns the LSE in base 2; the reference / FlashMLA use the natural log
        return o.view(c.b, c.s_q, c.h_q, DV), (lse * 0.6931471805599453).view(c.b, c.s_q, c.h_q).permute(0, 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="eager")
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--hq", default=None); ap.add_argument("--sq", default=None); ap.add_argument("--b", default=None)
    ap.add_argument("--sk", default=None); ap.add_argument("--page", default=None)
    ap.add_argument("--varlen", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--mem-gb", type=float, default=12.0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-smi", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="flashinfer")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if any(x is not None for x in (a.hq, a.sq, a.b, a.sk)):
        cases = [Case(h, sq, b, sk, p, a.varlen, a.dtype)
                 for h in parse_int_list(a.hq or "16") for sq in parse_int_list(a.sq or "1")
                 for b in parse_int_list(a.b or "64") for sk in parse_int_list(a.sk or "4096")
                 for p in parse_int_list(a.page or "64")]
    else:
        pages = parse_int_list(a.page) if a.page else [None]
        cases = [Case(c.h_q, c.s_q, c.b, c.s_k, pg or c.page, c.varlen or a.varlen, a.dtype)
                 for c in preset(a.preset) for pg in pages]
    modes = a.modes.split(",")
    import flashinfer
    print(f"[flashinfer] {flashinfer.__version__}; torch {torch.__version__}; {torch.cuda.get_device_name(0)}", flush=True)
    out = CsvOut(a.out or default_out(a.tag))
    failures = 0
    for case in cases:
        n_layers = 1
        if "graph" in modes:
            max_pages = -(-int(max(make_seqlens(case, a.seed))) // case.page)
            per_layer = case.b * max_pages * case.page * D * 2
            n_layers = int(max(1, min(a.layers, a.mem_gb * 1e9 // per_layer)))
        data = Data(case, n_layers, True, a.check, a.seed)
        base = case_row(case, data.stats)
        try:
            fi = FlashInferMLA(case, data, use_graph=("graph" in modes))
            fn = fi.runner(0)
            fn()
            torch.cuda.synchronize()
            cod = cl = None
            if a.check:
                o, lse = fi.normalize(fn())
                co, cll = [], []
                for i in range(min(case.b, 2)):
                    ro, rl = ref_attention(data.q[0][i], data.kv_of(0, i), case.causal)
                    co.append(cos_diff(o[i].float(), ro)); cll.append(cos_diff(lse[i].float(), rl))
                cod, cl = max(co), max(cll)
            meta_us = time_events(fi.plan, warmup=2, target_ms=30, min_reps=5, max_reps=30, flush=False)["median"]
            for mode in modes:
                if mode == "eager":
                    st, nl = time_events(fn), 1
                else:
                    runners = [fi.runner(l) for l in range(n_layers)]
                    g = torch.cuda.CUDAGraph()
                    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(2):
                            [r() for r in runners]
                    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
                    with torch.cuda.graph(g):
                        [r() for r in runners]
                    st, nl = time_graph_replay(g, n_layers), n_layers
                smi = smi_sample() if not a.no_smi else None
                m = metrics(case, data.stats, st["median"])
                out.write(impl="flashinfer", mode=mode, n_layers=nl, p10_us=st["p10"], p90_us=st["p90"], reps=st["reps"],
                          meta_us=meta_us, cos_out=cod if cod is not None else "", cos_lse=cl if cl is not None else "",
                          sm_mhz=smi[0] if smi else "", power_w=smi[1] if smi else "", status="ok", **base, **m)
                print(f"{case.tag():34s} flashinfer {mode:5s} {m['time_us']:9.1f} us {m['gbps']:7.1f} GB/s ({m['pct_bw']:4.0f}%)"
                      f" {m['tflops']:6.2f} TF" + (f" cos={cod:.1e}/{cl:.1e}" if cod is not None else ""), flush=True)
        except Exception as ex:
            traceback.print_exc(limit=4)
            msg = f"error: {type(ex).__name__}: {str(ex)[:150]}"
            out.write(impl="flashinfer", mode="-", status=msg, **base)
            print(f"{case.tag():34s} flashinfer FAILED {msg}", flush=True)
            failures += 1
            if failures >= 3 and failures == len(cases[: failures]):
                print("[flashinfer] first cases all failed; giving up (skip FlashInfer baseline)", flush=True)
                break
            try:
                torch.cuda.synchronize()
            except Exception:
                break
        del data
        torch.cuda.empty_cache()
    out.close()
    print(f"[flashinfer] done -> {out.path}")


if __name__ == "__main__":
    main()
