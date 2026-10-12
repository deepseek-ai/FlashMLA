#!/usr/bin/env python
"""Pivot one or more bench CSVs (bench_decode.py / bench_flashinfer.py) into a comparison table.

    python benchmarks/summarize.py a.csv [b.csv ...] [--md out.md] [--value time_us|gbps|tflops|pct_bw] [--mode eager]

Rows = case (h_q, s_q, b, s_k, page, varlen, kv), columns = impl, cell = chosen value; adds a `speedup` column
(best of the first impl vs. every other impl) when --ref IMPL is given.
"""
import argparse
import sys

import pandas as pd

KEYS = ["dtype", "kv", "h_q", "s_q", "b", "s_k", "varlen", "page"]


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df[df["status"] == "ok"].copy()


def pivot(paths, value="time_us", mode="eager", ref=None, md=None):
    paths = [paths] if isinstance(paths, str) else paths
    df = load(paths)
    if df.empty:
        print("(no ok rows)")
        return None
    df = df[df["mode"] == mode] if mode in set(df["mode"]) else df
    pv = df.pivot_table(index=KEYS, columns="impl", values=value, aggfunc="median")
    if ref and ref in pv.columns:
        for c in list(pv.columns):
            if c != ref:
                pv[f"{c}/{ref}"] = pv[c] / pv[ref] if value == "time_us" else pv[ref] / pv[c]
    pv = pv.round(2)
    txt = pv.to_string()
    print(f"\n== {value} ({mode}) ==\n{txt}")
    if md:
        with open(md, "w") as f:
            f.write(f"### {value} ({mode})\n\n" + to_md(pv.reset_index()) + "\n")
    return pv


def to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--value", default="time_us")
    ap.add_argument("--mode", default="eager")
    ap.add_argument("--ref", default=None, help="impl to compare the others against (ratio column)")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    pivot(a.csv, a.value, a.mode, a.ref, a.md)
    sys.exit(0)
