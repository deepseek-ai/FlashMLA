"""Fig. 1 - one decode layer: kernel pipeline, where the time goes, and which optimization touches which kernel."""
from sketch import FILL, INK, STROKE, Sketch, text_width

W, H = 1980, 1392
s = Sketch(W, H, seed=11)
BLUE, GREEN, ORANGE, RED, GRAY, PURPLE, TEAL = (STROKE[k] for k in ("blue", "green", "orange", "red", "gray", "purple", "teal"))

X0, CW, GAP = 186, 240, 14


def cx(i):
    return X0 + i * (CW + GAP)


# ----------------------------------------------------------------------------------------------- title
s.text(28, 48, "Fig. 1  One DeepSeek-V2-Lite decode layer on an RTX 3090 (sm_86): kernel pipeline and optimization map", 29, bold=True)
s.text(30, 80, "batch 1, 4096 cached tokens, bf16. Per-kernel times: nsys trace of the 27-layer CUDA graph (phase-2 build, includes ~0.4 us of "
               "tracing overhead per kernel).", 16, GRAY)

# ----------------------------------------------------------------------------------------------- DRAM lane
s.group(X0 - 14, 108, 7 * CW + 6 * GAP + 28, 102, "DRAM  -  each layer has to stream 32.2 MB of weights + KV;  read peak 873.8 GB/s  =>  byte floor 37.0 us",
        stroke=PURPLE, size=16)
mem = [(0, 1, "W_qkv_a", "14.9 MB"), (1, 1, "W_UK", "2.1 MB"), (2, 2, "paged KV cache", "read 4.7 MB / write 1 row per token"),
       (5, 1, "W_UV", "2.1 MB"), (6, 1, "W_o", "8.4 MB")]
for c, span, name, sz in mem:
    x, w = cx(c) + 18, span * CW + (span - 1) * GAP - 36
    s.rect(x, 132, w, 62, PURPLE, FILL["purple"], r=8, hatch="#d0bfff", hatch_gap=9)
    s.out.append(f'<rect x="{x + w / 2 - text_width(name, 17, True) / 2 - 8:.1f}" y="140" width="{text_width(name, 17, True) + 16:.1f}" height="46" fill="{FILL["purple"]}" opacity="0.92"/>')
    s.text(x + w / 2, 160, name, 17, PURPLE, "middle", bold=True)
    s.text(x + w / 2, 181, sz, 14, INK, "middle", maxw=w - 8)

# ----------------------------------------------------------------------------------------------- kernel row
KY, KH = 262, 150
GY, GH = 236, 346
s.group(18, GY, W - 36, GH, None, stroke=GRAY)
kern = [
    ("qkv_a", ["q_proj + kv_a_proj, one GEMM", "W: 3648 x 2048", "grid 114 x 1 x 1", "-> q 16x192, kv_c 512, k_pe 64"], "blue"),
    ("W_UK bmm", ["absorb: q_nope 128 -> 512", "16 heads x [512 x 128]", "grid 32 x 1 x 16", "-> q_abs [T, 16, 512]"], "blue"),
    ("prologue (fused)", ["RoPE q_pe/k_pe, RMSNorm kv_c", "concat q_full [T, 16, 576]", "write KV row (+ quantize)", "grid (H+1) x T, latency-bound"], "green"),
    ("decode (v2)", ["82 persistent CTAs, split-KV", "QK -> softmax -> PV per tile", "streams 4.7 MB of KV", "-> partial O + LSE (fp32)"], "orange"),
    ("combine", ["sum_s exp(lse_s - lse) * O_s", "reads 2.6 MB partials from L2", "wide / warp variants", "-> o [T, 16, 512]"], "orange"),
    ("W_UV bmm", ["16 heads x [128 x 512]", "grid 8 x 1 x 16", "writes [T, 16*128] in place", "-> y [T, 2048]"], "blue"),
    ("o_proj", ["W: 2048 x 2048", "grid 64 x 1 x 1", "-> hidden' [T, 2048]", "-> next layer's qkv_a"], "blue"),
]
for i, (name, ls, col) in enumerate(kern):
    s.box(cx(i), KY, CW, KH, name, ls, fill=FILL[col], stroke=STROKE[col], tsize=20, size=14.5, lh=1.42, pad=12)
    if i < 6:
        s.arrow([(cx(i) + CW + 1, KY + KH / 2), (cx(i + 1) - 2, KY + KH / 2)], width=2.2, head=7, rough=0.4)
s.text(cx(6) + CW + 6, KY + KH / 2 - 10, "", 12)
# DRAM -> kernel arrows
for c, tgt in ((0, 0), (1, 1), (5, 5), (6, 6)):
    s.arrow([(cx(c) + CW / 2, 196), (cx(tgt) + CW / 2, KY - 3)], PURPLE, 2.2, head=9)
s.arrow([(cx(3) + CW / 2, 196), (cx(3) + CW / 2, KY - 3)], PURPLE, 2.2, head=9)
s.arrow([(cx(2) + CW / 2, KY - 3), (cx(2) + CW / 2, 198)], PURPLE, 2.0, head=9, dash="6 5")
s.text(cx(2) + CW / 2 + 10, 226, "write 1 row", 13.5, PURPLE, bold=True)
s.text(cx(3) + CW / 2 + 10, 226, "read", 13.5, PURPLE, bold=True)
for c in (0, 1, 5, 6):
    s.text(cx(c) + CW / 2 + 10, 226, "stream", 13.5, PURPLE, bold=True)
# per-step box (left column)
s.box(26, KY, 146, KH, "once per step", ["metadata kernel", "4.2 us: seqlens", "-> split schedule", "(parallel, on GPU)", "+ ~13 tiny", "bookkeeping", "kernels"],
      fill=FILL["gray"], stroke=GRAY, tsize=16, size=12.5, lh=1.2, pad=7)

# ----------------------------------------------------------------------------------------------- time bars
TY = KY + KH + 18
times = [(20.80, 17.1), (4.77, 2.4), (2.40, 0.0), (12.64, 5.4), (4.29, 0.0), (4.87, 2.4), (14.46, 9.6)]
SC = 11.0
for i, (t, fl) in enumerate(times):
    x = cx(i)
    if fl > 0:
        s.rect(x, TY, fl * SC, 22, STROKE["teal"], FILL["teal"], 1.5, r=0, rough=0.6)
    s.rect(x + fl * SC, TY, (t - fl) * SC, 22, RED, FILL["red"], 1.5, r=0, rough=0.6, hatch="#fa5252", hatch_gap=6)
    s.text(x, TY + 44, f"{t:.1f} us", 17, INK, bold=True)
    s.text(x + text_width(f"{t:.1f} us", 17, True) + 8, TY + 44, f"floor {fl:.1f}  +  {t - fl:.1f}" if fl else "no DRAM bytes: all latency", 14, GRAY)
s.lines(28, TY + 12, ["time = byte floor", "+ excess (hatched)", "total 64.2 = 37.0 + 27"], 13, GRAY, lh=1.3)

# ----------------------------------------------------------------------------------------------- L2 lane
LY = TY + 66
s.text(28, LY + 30, "L2 cache (6 MB)", 15, TEAL, bold=True)
wx, ww = cx(3) + 10, 2 * CW + GAP - 20
s.rect(wx, LY, ww, 50, TEAL, FILL["teal"], r=8)
s.text(wx + ww / 2, LY + 21, "workspace: partial O (fp32, 2.6 MB) + LSE", 15, TEAL, "middle", bold=True, maxw=ww - 10)
s.text(wx + ww / 2, LY + 41, "written by decode, read once by combine -> left dirty in L2", 13.5, INK, "middle", maxw=ww - 10)
s.arrow([(cx(3) + CW / 2 - 30, TY + 50), (cx(3) + CW / 2 - 30, LY - 2)], TEAL, 2, head=8)
s.arrow([(cx(4) + CW / 2 + 60, LY - 2), (cx(4) + CW / 2 + 60, TY + 50)], TEAL, 2, head=8)
nx = cx(5) + 6
s.arrow([(wx + ww + 2, LY + 25), (nx + 16, LY + 25)], RED, 2.2, head=9, dash="7 5")
s.lines(nx + 24, LY + 12, [("written back to DRAM while W_UV / o_proj stream weights:", RED, True),
                           ("both fight for bandwidth  ->  fixed by P1 (discard after the read)", RED)], 14, lh=1.45,
        maxw=2 * CW + GAP - 30)
s.lines(cx(0) + 4, LY + 14, ["activations (q, kv_c, k_pe, q_full, o, y) are < 0.1 MB and stay in L2;",
                             "only weights, KV and the partial-O write-back touch DRAM"], 13.5, GRAY, lh=1.4, maxw=3 * CW + 2 * GAP - 8)

glab = "one CUDA graph = 27 layers x 7 kernels   (before phase 1: 40-43 kernels per layer)"
s.out.append(f'<rect x="{W / 2 - text_width(glab, 16, True) / 2 - 12:.1f}" y="{GY + GH - 13}" width="{text_width(glab, 16, True) + 24:.1f}" height="26" fill="{FILL["paper"]}"/>')
s.text(W / 2, GY + GH + 6, glab, 16, GRAY, "middle", bold=True)

# ----------------------------------------------------------------------------------------------- optimization matrix
MY = 636
s.text(28, MY - 4, "Which optimization touches which kernel", 21, bold=True)
ROW_H, BAN_H, CELL_H = 178, 34, 132


def cell(x, y, w, h, items, stroke, fill, dash=None, hatch=None):
    s.rect(x, y, w, h, stroke, fill, 1.6, r=8, dash=dash, hatch=hatch, rough=0.8)
    ty = y + 24
    for kind, t in items:
        if kind == "b":      # bold title in the row colour
            s.text(x + 11, ty, t, 15, stroke, bold=True, maxw=w - 18)
            ty += 21
        elif kind == "x":    # rejected
            s.text(x + 11, ty, t, 14, RED, bold=True, maxw=w - 18)
            ty += 20
        elif kind == "g":    # greyed
            s.text(x + 11, ty, t, 14.5, GRAY, bold=True, maxw=w - 18)
            ty += 21
        else:
            s.text(x + 11, ty, t, 13.5, INK if kind == "n" else GRAY, maxw=w - 18)
            ty += 19


rows = [
    ("Phase 1", ["implemented", "v2 kernel, graph,", "fused prologue,", "8-bit KV"], "blue", None,
     "Whole-layer CUDA graph + graph-safe API (pre-allocated out / lse / workspace, no host sync): ~40 launches per layer vanish.   b=1: 820 -> 115 us/layer;  with fused prologue + v2 kernel -> 65.8",
     [[("g", "cuBLAS (unchanged)"), ("d", "only 57 CTAs at T <= 64"), ("d", "61-63% of the DRAM peak")],
      [("g", "cuBLAS (unchanged)"), ("d", "256 small CTAs"), ("d", "19-39% of the DRAM peak")],
      [("b", "Fused prologue"), ("n", "RoPE + RMSNorm + concat + paged"), ("n", "KV write (+ quantize): 1 kernel"), ("n", "59-90 us (torch ops) -> 1.8-14 us")],
      [("b", "v2 decode kernel (Fig. 2)"), ("n", "h_q=16: 94-99% of DRAM peak"), ("n", "h_q=128: 73-75% of bf16 peak"), ("b", "8-bit KV cache: 1.47-1.71x")],
      [("b", "Row-count-based combine"), ("n", "wide16 / wide8 / wide4 / warp"), ("n", "b=1: 9.2 -> 5.5 us")],
      [("g", "cuBLAS (unchanged)"), ("d", "32 CTAs, 22-38% of the peak"), ("d", "+ 2.3 us reshape copy")],
      [("g", "cuBLAS (unchanged)"), ("d", "64 CTAs"), ("d", "51-63% of the DRAM peak")]]),
    ("Phase 2", ["implemented", "skinny GEMM,", "two KV groups", "for 8-bit KV"], "green", None,
     "Weight-streaming \"skinny\" GEMM replaces cuBLAS for the four projections (Fig. 3A).   b=8: the four GEMMs 61.7 -> 41.5 us, the layer 115.4 -> 102.0 us (1.13x);  8 -> 7 kernels per layer",
     [[("b", "Skinny GEMM"), ("n", "114 CTAs, 48 KB cp.async ring"), ("n", "streams at ~870 GB/s")],
      [("b", "Skinny bmm"), ("n", "512 CTAs (16 weight rows each)"), ("n", "all 16 heads in one launch")],
      [("g", "(unchanged)")],
      [("b", "8-bit: 2 KV groups per CTA"), ("n", "8 warps hide the HMMA latency"), ("n", "one choice per launch (load >= 64)"), ("n", "long contexts +4-11%"), ("n", "FP8 vs bf16: 1.55-1.70x")],
      [("x", "x  single-round partial read"), ("d", "4.28 -> 4.26 us: no gain, dropped")],
      [("b", "Skinny bmm"), ("n", "writes the o_proj input layout"), ("n", "directly: the reshape copy is gone")],
      [("b", "Skinny GEMM"), ("n", "64 CTAs, 48 KB cp.async ring")]]),
    ("Phase 3", ["planned,", "not implemented", "", "P5: lean per-step", "bookkeeping", "-0.4 .. -0.6 us"], "orange", "8 6",
     "Plan P1-P6. Numbers are research measurements on the real 27-layer graph (A/B runs or emulation).   P1+P2+P3 stacked, measured: b=1 61.1 -> 56.2 us/layer;  all six, projected: 51-53.5",
     [[("b", "P3  prefetch chain (target)"), ("n", "the previous o_proj prefetches"), ("n", "the first 2 MB of this weight;"), ("n", "weights loaded evict_first")],
      [("b", "P3  <- prefetched by qkv_a"), ("b", "P4  one launch with prologue"), ("n", "7 -> 6 kernels per layer")],
      [("b", "P4  horizontal fusion"), ("n", "W_UK writes q_full[..., :512];"), ("n", "prologue's latency chain overlaps"), ("n", "the weight stream: -0.8 .. -2.4 us")],
      [("b", "P2  balanced part count"), ("n", "Pe = ceil(T / ceil(T / P)): 82 -> 65"), ("n", "parts at b=1;  -0.2 .. -3.1 us"), ("x", "x KV prefetch: slower in-layer")],
      [("b", "P1  discard.global.L2"), ("n", "drop the dirty partials after the"), ("n", "read: -2.6 .. -3.3 us (b = 1..8)"), ("b", "P6  prefetch W_UV"), ("d", "(synthetic evidence only so far)")],
      [("b", "P3  -> prefetch W_o"), ("n", "and, thanks to P1, no write-back"), ("n", "competes for DRAM any more")],
      [("b", "P3  -> prefetch next qkv_a"), ("n", "with P1 in place:"), ("n", "b=1 -1.7 .. -1.9,  b=8 -1.4 us")]]),
]
for r, (name, sub, col, dash, banner, cells) in enumerate(rows):
    y = MY + 14 + r * (ROW_H + 14)
    st, fl = STROKE[col], FILL[col]
    s.rect(26, y, 146, ROW_H, st, fl, 2.0, r=10, dash=dash)
    s.text(99, y + 34, name, 23, st, "middle", bold=True)
    s.lines(99, y + 60, sub, 13.5, INK, "middle", lh=1.32, maxw=140)
    s.rect(X0, y, 7 * CW + 6 * GAP, BAN_H, st, fl, 1.6, r=8, dash=dash, rough=0.7)
    s.text(X0 + 14, y + 23, banner, 14.5, INK, maxw=7 * CW + 6 * GAP - 24)
    for i, items in enumerate(cells):
        grey = items[0][0] == "g" or items[0][0] == "x"
        cell(cx(i), y + BAN_H + 12, CW, CELL_H, items, GRAY if grey else st, FILL["gray"] if grey else "#ffffff", dash=dash if not grey else "3 5",
             hatch=None)
# prefetch arcs in the phase-3 row
y3 = MY + 14 + 2 * (ROW_H + 14) + BAN_H + 12 + CELL_H
for a, b in ((0, 1), (5, 6)):
    s.arrow([(cx(a) + CW - 60, y3 + 4), ((cx(a) + cx(b) + CW) / 2, y3 + 26), (cx(b) + 60, y3 + 4)], ORANGE, 2.2, dash="7 5", smooth=True, head=9)
s.arrow([(cx(4) + CW - 40, y3 + 4), ((cx(4) + cx(5) + CW) / 2, y3 + 26), (cx(5) + 40, y3 + 4)], ORANGE, 2.0, dash="2 6", smooth=True, head=9)
s.arrow([(cx(6) + CW - 50, y3 + 4), (cx(6) + CW + 4, y3 + 24)], ORANGE, 2.2, dash="7 5", head=9)
s.text(cx(0) + CW + 7, y3 + 44, "prefetch (P3)", 13.5, ORANGE, "middle", bold=True)
s.text(cx(4) + CW + 7, y3 + 44, "prefetch (P6)", 13.5, ORANGE, "middle", bold=True)
s.text(cx(5) + CW + 7, y3 + 44, "prefetch (P3)", 13.5, ORANGE, "middle", bold=True)
s.text(cx(6) + CW - 4, y3 + 44, "to the next layer", 13.5, ORANGE, "end", bold=True)

# ----------------------------------------------------------------------------------------------- bottom strip
BY = y3 + 70
s.line(26, BY - 12, W - 26, BY - 12, GRAY, 1.2, dash="2 7", passes=1)


def stepper(x, y, label, steps, colours):
    s.text(x, y + 6, label, 16, INK, bold=True)
    x += text_width(label, 16, True) + 14
    for i, ((top, bot), c) in enumerate(zip(steps, colours)):
        wbox = max(text_width(top, 19, True), text_width(bot, 12.5)) + 26
        st = STROKE[c]
        s.rect(x, y - 26, wbox, 54, st, FILL[c], 1.7, r=9, dash="7 5" if c == "orange" else None, rough=0.7)
        s.text(x + wbox / 2, y - 3, top, 19, st, "middle", bold=True)
        s.text(x + wbox / 2, y + 17, bot, 12.5, INK, "middle")
        x += wbox
        if i < len(steps) - 1:
            s.arrow([(x + 5, y), (x + 27, y)], width=2, head=8, rough=0.4)
            x += 32
    return x


stepper(28, BY + 30, "kernels per layer:", [("40-43", "unfused PyTorch"), ("8", "phase 1"), ("7", "phase 2"), ("6", "phase 3 plan")],
        ["gray", "blue", "green", "orange"])
stepper(760, BY + 30, "us per layer (b=1, s_k=4096, bf16):",
        [("820", "unfused, eager"), ("115", "unfused, graph"), ("65.8", "phase 1"), ("61.5", "phase 2"), ("56.2", "P1+P2+P3 measured"),
         ("51-53.5", "all six, projected")], ["gray", "gray", "blue", "green", "orange", "orange"])
# legend
ly = BY + 84
s.text(28, ly + 5, "legend:", 15, INK, bold=True)
lx = 100
for c, t, dash in (("blue", "phase 1 (implemented)", None), ("green", "phase 2 (implemented)", None), ("orange", "phase 3 (planned)", "7 5")):
    s.rect(lx, ly - 11, 30, 20, STROKE[c], FILL[c], 1.5, r=4, dash=dash, rough=0.5)
    s.text(lx + 38, ly + 5, t, 14)
    lx += 60 + text_width(t, 14)
s.rect(lx, ly - 11, 30, 20, STROKE["teal"], FILL["teal"], 1.5, r=0, rough=0.5)
s.text(lx + 38, ly + 5, "time explained by bytes / 873.8 GB/s", 14)
lx += 60 + text_width("time explained by bytes / 873.8 GB/s", 14)
s.rect(lx, ly - 11, 30, 20, RED, FILL["red"], 1.5, r=0, rough=0.5, hatch="#fa5252", hatch_gap=6)
s.text(lx + 38, ly + 5, "time above the byte floor", 14)
lx += 60 + text_width("time above the byte floor", 14)
s.rect(lx, ly - 11, 30, 20, PURPLE, FILL["purple"], 1.5, r=4, rough=0.5, hatch="#d0bfff", hatch_gap=7)
s.text(lx + 38, ly + 5, "data that lives in DRAM", 14)

s.save("fig1_layer_pipeline.svg")
print("fig1 ok, bottom =", ly + 20, "H =", H)
