#!/usr/bin/env python
"""Integration matrix -> markdown tables (docs/bench_matrix.md) + scorecard numbers.

    python benchmarks/summarize_matrix.py [--logs /data2/LiuJun/tmp/flashmla/logs] [--out docs/bench_matrix.md]

Reads the newest CSV of each tag written by bench_decode.py / bench_flashinfer.py (see /data2/LiuJun/tmp/flashmla/si_matrix.sh):
    si_head_p32_bf16 + si_head_fi_p32_bf16   headline (main shapes, page 32: v2 / legacy / triton / flashinfer)
    si_p64_bf16 (+ si_fi_p64_bf16)           full matrix, page 64: v2 / triton (/ flashinfer)
    si_p32_bf16                              full matrix, page 32: legacy / v2
    si_p64_fp16, si_p64_fp8, si_p64_int8     core matrix
    si_p16_spot (+ si_fi_p16_spot)           page-16 spot checks
Missing files are skipped.
"""
import argparse
import glob
import math
import os

import pandas as pd

PEAK_BW, PEAK_TF = 873.8, {"bfloat16": 70.0, "float16": 66.4}
KEY = ["dtype", "kv", "h_q", "s_q", "b", "s_k", "page"]


def load(logs, tag):
    fs = sorted(glob.glob(os.path.join(logs, f"bench_{tag}_2*.csv")))
    if not fs:
        return None
    df = pd.read_csv(fs[-1])
    df = df[df["status"] == "ok"].copy()
    df["src"] = os.path.basename(fs[-1])
    return df


def cat(*dfs):
    dfs = [d for d in dfs if d is not None]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def piv(df, col="time_us", mode="graph"):
    d = df[df["mode"] == mode]
    return d.pivot_table(index=KEY, columns="impl", values=col, aggfunc="median")


def geomean(x):
    x = [v for v in x if v and v > 0 and not math.isnan(v)]
    return math.exp(sum(math.log(v) for v in x) / len(x)) if x else float("nan")


def fmt(v, nd=1):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def shape(idx):
    dtype, kv, h, sq, b, sk, page = idx
    return f"h{h} sq{sq} b{b} sk{sk}"


def compare_table(df, impls, base, modes=("eager", "graph"), title="", detail=True):
    """Rows = shapes, per mode: time of each impl (us) and speedup of `base` over the others."""
    out = [f"#### {title}", ""]
    for mode in modes:
        p = piv(df, "time_us", mode)
        if p.empty:
            continue
        cols = [i for i in impls if i in p.columns]
        out.append(f"**{mode}** (median us per call{' per layer' if mode == 'graph' else ''}; `x` = other / {base})")
        out.append("")
        hdr = ["shape"] + [f"{c} us" for c in cols] + [f"{c}/{base}" for c in cols if c != base]
        rows = []
        for idx, r in p.iterrows():
            rows.append([shape(idx)] + [fmt(r.get(c)) for c in cols] +
                        [fmt(r.get(c) / r[base], 2) if base in r and r.get(base) else "-" for c in cols if c != base])
        if detail:
            out.append(md_table(hdr, rows))
            out.append("")
        # summary by h_q
        srows = []
        for h, g in p.reset_index().groupby("h_q"):
            srow = [h]
            for c in cols:
                if c == base:
                    continue
                ratios = (g[c] / g[base]).tolist() if c in g and base in g else []
                srow.append(f"{geomean(ratios):.2f} ({min(ratios):.2f}-{max(ratios):.2f})" if ratios else "-")
            srows.append(srow)
        out.append(f"geomean (min-max) of `impl / {base}` per h_q, {mode}:")
        out.append("")
        out.append(md_table(["h_q"] + [f"{c}/{base}" for c in cols if c != base], srows))
        out.append("")
    return "\n".join(out)


def metric_table(df, impl, mode, title):
    d = df[(df["impl"] == impl) & (df["mode"] == mode)]
    if d.empty:
        return ""
    rows = []
    for _, r in d.sort_values(["h_q", "s_q", "b", "s_k"]).iterrows():
        rows.append([f"h{r.h_q} sq{r.s_q} b{r.b} sk{r.s_k}", fmt(r.time_us), fmt(r.gbps, 0), fmt(r.pct_bw, 0), fmt(r.tflops, 1), fmt(r.pct_tf, 0),
                     r.get("cos_out", ""), fmt(r.get("sm_mhz"), 0), fmt(r.get("power_w"), 0)])
    return "\n".join([f"#### {title}", "", md_table(["shape", "us", "GB/s", "% of 874", "TFLOPS", "% of peak", "cos_diff(out)", "SM MHz", "W"], rows), ""])


def quant_table(bf, q8, i8, mode):
    out = []
    pb = piv(bf, "time_us", mode)
    rows = []
    for name, qd in (("fp8", q8), ("int8", i8)):
        if qd is None:
            continue
        pq = piv(qd, "time_us", mode)
        for idx in pq.index:
            ib = (idx[0], "native") + idx[2:]
            if ib in pb.index and "v2" in pb.columns and "v2" in pq.columns:
                rows.append((name, idx, pb.loc[ib, "v2"], pq.loc[idx, "v2"]))
    if not rows:
        return ""
    tab = []
    for name, idx, tb, tq in sorted(rows, key=lambda r: (r[0], r[1][2], r[1][3], r[1][4], r[1][5])):
        tab.append([name, shape(idx), fmt(tb), fmt(tq), fmt(tb / tq, 2)])
    out.append(f"#### 8-bit KV vs bf16 (v2, page 64, {mode}, per layer us)")
    out.append("")
    out.append(md_table(["kv", "shape", "bf16 us", "8-bit us", "speedup"], tab))
    out.append("")
    # summary per h_q for the memory-bound batch sizes
    srows = []
    for name in ("fp8", "int8"):
        for h in sorted({r[1][2] for r in rows}):
            for (lo, hi, lab) in ((1, 4, "b<=4"), (16, 16, "b=16"), (64, 128, "b>=64")):
                sp = [r[2] / r[3] for r in rows if r[0] == name and r[1][2] == h and lo <= r[1][4] <= hi and r[1][3] == 1]
                if sp:
                    srows.append([name, f"h{h}", lab, f"{geomean(sp):.2f}", f"{min(sp):.2f}-{max(sp):.2f}"])
    out.append(f"speedup (s_q=1) geomean / range by h_q and batch ({mode}):")
    out.append("")
    out.append(md_table(["kv", "h_q", "batch", "geomean", "range"], srows))
    out.append("")
    return "\n".join(out)


def scorecard(v2):
    """Numbers used by the plan section 5 scorecard (graph mode, page 64, s_q = 1)."""
    out = ["#### Scorecard inputs (v2, bf16, page 64)", ""]
    g = v2[(v2["impl"] == "v2") & (v2["kv"] == "native") & (v2["dtype"] == "bfloat16") & (v2["s_q"] == 1)]
    rows = []
    d = g[(g["h_q"] == 16) & (g["b"] >= 64) & (g["s_k"] >= 4096)]
    for mode in ("eager", "graph"):
        dm = d[d["mode"] == mode]
        if not dm.empty:
            rows.append([f"h16 b>=64 sk>=4096 {mode}", f"{dm.gbps.min():.0f}-{dm.gbps.max():.0f} GB/s (median {dm.gbps.median():.0f})",
                         f"{dm.pct_bw.min():.0f}-{dm.pct_bw.max():.0f}% of {PEAK_BW}", ">= 80% (700 GB/s)"])
    for lab, sel in (("h128 b>=16 sk>=4096", (g["h_q"] == 128) & (g["b"] >= 16) & (g["s_k"] >= 4096)),
                     ("h128 b>=4 (all s_k)", (g["h_q"] == 128) & (g["b"] >= 4))):
        d = g[sel]
        for mode in ("eager", "graph"):
            dm = d[d["mode"] == mode]
            if not dm.empty:
                rows.append([f"{lab} {mode}", f"{dm.tflops.min():.1f}-{dm.tflops.max():.1f} TFLOPS (median {dm.tflops.median():.1f})",
                             f"{dm.pct_tf.min():.0f}-{dm.pct_tf.max():.0f}% of {PEAK_TF['bfloat16']}", ">= 60% (42 TFLOPS)"])
    out.append(md_table(["case", "measured", "% of measured peak", "target"], rows))
    out.append("")
    # small-batch latency
    rows = []
    d = g[(g["h_q"] == 16) & (g["b"] <= 4)]
    for mode in ("graph", "eager"):
        for _, r in d[d["mode"] == mode].sort_values(["b", "s_k"]).iterrows():
            kvb = r.s_k_total * 1152
            ideal = kvb / (PEAK_BW * 1e9) * 1e6
            tgt = 1.3 * ideal + 5.0
            rows.append([mode, f"b{r.b} sk{r.s_k}", fmt(kvb / 1e6, 1), fmt(ideal, 1), fmt(tgt, 1), fmt(r.time_us, 1), fmt(r.time_us - tgt, 1), "met" if r.time_us <= tgt else "MISS"])
    out.append("b <= 4 latency (h_q=16, s_q=1): target 1.3 x (KV bytes / 873.8 GB/s) + 5 us")
    out.append("")
    out.append(md_table(["mode", "shape", "KV MB", "KV/BW us", "target us", "measured us", "excess us", "verdict"], rows))
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/data2/LiuJun/tmp/flashmla/logs")
    ap.add_argument("--out", default="docs/bench_matrix.md")
    a = ap.parse_args()
    L = lambda t: load(a.logs, t)
    head, head_fi = cat(L("si_head_p32_bf16")), L("si_head_fi_p32_bf16")
    p64, fi64 = L("si_p64_bf16"), cat(L("si_fi_p64_bf16"), L("si_fi_p64_bf16_rest"), L("si_fi_p64_bf16_rest2"))
    p32 = L("si_p32_bf16")
    fp16 = L("si_p64_fp16")
    fi16 = cat(L("si_fi_p64_fp16_core"), L("si_fi_p64_fp16_core_rest"), L("si_fi_p64_fp16_core_rest2"))
    q8 = cat(L("si_p64_fp8"), L("si_p64_fp8_big"), L("si_p64_fp8_big2"))
    i8 = cat(L("si_p64_int8"), L("si_p64_int8_big"), L("si_p64_int8_big2"))
    p16, fi16s = L("si_p16_spot"), L("si_fi_p16_spot")

    out = ["# Integration matrix (RTX 3090, GPU1, nvcc 12.4 / torch 2.7.1+cu126)", "",
           "Generated by `benchmarks/summarize_matrix.py` from the CSVs in `/data2/LiuJun/tmp/flashmla/logs` (newest file per tag). Times are medians; "
           "eager = one call with the 256 MB L2 flush before every rep (includes launch overhead), graph = per-layer time of an N-layer CUDA graph "
           "(each layer with its own KV cache, N<=27). GB/s counts KV + Q + O bytes against the measured read-only peak 873.8 GB/s, TFLOPS = "
           "2*s_q*h_q*sum(s_k)*(576+512) against 70.0 (bf16) / 66.4 (fp16) TFLOPS. `--cool` was on: the benchmark waits until the GPU is <= 66 C "
           "whenever it exceeds 78 C (the RTX 3090 here throttles to ~1.2-1.4 GHz at its 83 C target under continuous load, clocks cannot be locked); "
           "the SM clock and power of every row are in the CSV columns `sm_mhz`, `power_w`.", ""]
    if head is not None:
        h = cat(head, head_fi)
        out += ["## 1. Headline: main shapes, page 32, bf16 (all four implementations)", "",
                compare_table(h, ["v2", "legacy", "triton", "flashinfer"], "v2", title="v2 vs legacy vs Triton vs FlashInfer"), ""]
    if p64 is not None:
        d = cat(p64, fi64)
        out += ["## 2. Full matrix, page 64, bf16 (v2 / Triton / FlashInfer)", "",
                compare_table(d, ["v2", "triton", "flashinfer"], "v2", title="page 64, bf16"), ""]
        out += [scorecard(p64), ""]
        out += [metric_table(p64, "v2", "graph", "v2 absolute numbers (graph)"), metric_table(p64, "v2", "eager", "v2 absolute numbers (eager)")]
    if p32 is not None:
        out += ["## 3. Full matrix, page 32, bf16 (legacy vs v2)", "",
                compare_table(p32, ["v2", "legacy"], "v2", title="page 32, bf16 (legacy only supports page 32)"), ""]
    if fp16 is not None:
        d = cat(fp16, fi16)
        out += ["## 4. fp16 (core matrix, page 64)", "", compare_table(d, ["v2", "triton", "flashinfer"], "v2", title="fp16"), ""]
    if q8 is not None or i8 is not None:
        out += ["## 5. FP8 / INT8 KV (core matrix, page 64)", ""]
        base = p64
        if base is not None:
            for mode in ("graph", "eager"):
                out += [quant_table(base, q8, i8, mode)]
    if p16 is not None:
        d = cat(p16, fi16s)
        out += ["## 6. Page 16 spot checks", "", compare_table(d, ["v2", "triton", "flashinfer"], "v2", title="page 16, bf16"), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", newline="\n").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
