"""Fig. 4 - results on the RTX 3090: whole decode layer, the decode kernel against other implementations, hardware ceilings."""
from sketch import FILL, INK, STROKE, Sketch, text_width

W, H = 1980, 1226
s = Sketch(W, H, seed=51)
BLUE, GREEN, ORANGE, RED, GRAY, PURPLE, TEAL, YELLOW = (STROKE[k] for k in ("blue", "green", "orange", "red", "gray", "purple", "teal", "yellow"))

s.text(28, 48, "Fig. 4  Results on one RTX 3090: from the baseline to the optimized decode layer", 28, bold=True)
s.text(30, 80, "All timings: GPU otherwise idle, CUDA graph replay unless noted, KV page size 32 or 64, medians. Lower is better in A and B.", 16, GRAY)

# =============================================================================================== panel A
AY, AH = 112, 548
s.group(24, AY, 1932, AH, "A  One MLA decode layer (DeepSeek-V2-Lite shape, h_q = 16, 4096 cached tokens): microseconds per layer", stroke=BLUE, size=17)
series = [("unfused PyTorch, eager", "gray", None), ("unfused PyTorch, CUDA graph", "gray", "#adb5bd"), ("fused layer + Triton attention", "purple", None),
          ("fused layer + legacy kernel", "yellow", None), ("phase 1: fused layer + v2 kernel", "blue", None),
          ("phase 2: + skinny GEMM (bf16 KV)", "green", None), ("phase 2: 8-bit KV cache (FP8)", "teal", None),
          ("phase 3 plan: P1+P2+P3 (prototype, measured)", "orange", "#ffa94d")]
data = {  # batch: eager, graph, triton, legacy, v2 (phase 1), phase 2 bf16, phase 2 fp8, phase-3 stack (bf16) or None
    1: [820.1, 114.7, 76.4, 71.1, 65.8, 61.5, 61.6, 56.2],
    8: [895.1, 184.9, 133.8, 115.7, 117.9, 102.3, 86.3, 97.4],
    32: [903.1, 330.0, 355.7, 258.2, 259.3, 239.5, 172.0, None],
    64: [899.2, 515.0, 599.6, 440.9, 440.0, 431.4, 291.2, None]}
CH = 330               # plot height
CY = AY + 86           # top of the plots
for gi, (b, vals) in enumerate(data.items()):
    gx = 80 + gi * 474
    gw = 440
    ymax = max(v for v in vals[1:] if v) * 1.12
    base = CY + CH
    s.text(gx + gw / 2, CY - 22, f"batch {b}", 18, INK, "middle", bold=True)
    s.line(gx - 6, base, gx + gw, base, INK, 1.8, rough=0.6)
    s.line(gx - 6, base, gx - 6, CY - 6, INK, 1.6, rough=0.6)
    step = {1: 25, 8: 50, 32: 100, 64: 150}[b]
    t = step
    while t < ymax:
        ty = base - CH * t / ymax
        s.line(gx - 11, ty, gx + gw, ty, "#ced4da", 1.0, rough=0.3, passes=1, dash="3 6")
        s.text(gx - 14, ty + 5, str(t), 12.5, GRAY, "end")
        t += step
    bw = 44
    for i, v in enumerate(vals):
        name, col, hatch = series[i]
        x = gx + 10 + i * (bw + 9)
        if v is None:
            s.text(x + bw / 2, base - 26, "not", 12, GRAY, "middle")
            s.text(x + bw / 2, base - 11, "run", 12, GRAY, "middle")
            continue
        if i == 0:   # broken bar
            h = CH * 1.04
            s.rect(x, base - h, bw, h, STROKE[col], "#f1f3f5", 1.5, r=2, rough=0.5)
            by = base - CH * 0.78
            s.out.append(f'<rect x="{x - 4}" y="{by - 7}" width="{bw + 8}" height="14" fill="{FILL["paper"]}"/>')
            s.line(x - 6, by + 5, x + bw + 6, by - 9, STROKE[col], 1.6, rough=0.4)
            s.line(x - 6, by + 13, x + bw + 6, by - 1, STROKE[col], 1.6, rough=0.4)
            s.text(x + bw / 2, base - h - 8, f"{v:.0f}", 14, STROKE[col], "middle", bold=True)
            continue
        h = CH * v / ymax
        dash = "7 5" if i == 7 else None
        s.rect(x, base - h, bw, h, STROKE[col], FILL[col], 1.6, r=2, rough=0.5, hatch=hatch, hatch_gap=7, dash=dash)
        s.text(x + bw / 2, base - h - 8, f"{v:.1f}" if v < 200 else f"{v:.0f}", 13.5, STROKE[col], "middle", bold=True)
    # speed-up strip under the plot
    sp_graph = vals[1] / vals[5]
    sp_fp8 = vals[1] / vals[6]
    s.text(gx + gw / 2, base + 26, f"vs unfused graph: {sp_graph:.2f}x (bf16), {sp_fp8:.2f}x (FP8 KV)", 13.5, GREEN, "middle", bold=True)
    s.text(gx + gw / 2, base + 46, f"vs unfused eager: {vals[0] / vals[5]:.1f}x", 13.5, GRAY, "middle")
# legend
lx, ly = 80, AY + AH - 62
for i, (name, col, hatch) in enumerate(series):
    x = lx + (i % 4) * 470
    y = ly + (i // 4) * 26
    s.rect(x, y, 26, 16, STROKE[col], "#f1f3f5" if i == 0 else FILL[col], 1.4, r=2, rough=0.4, hatch=hatch, hatch_gap=6, dash="6 4" if i == 7 else None)
    s.text(x + 34, y + 13, name, 13.5, INK)

# =============================================================================================== panel B
BY2, BH2 = AY + AH + 26, H - (AY + AH + 26) - 22
s.group(24, BY2, 1130, BH2, "B  The decode kernel alone: time of other implementations relative to v2 (= 1.0)", stroke=ORANGE, size=17)
shapes = [("h16", "b1 s1k", 9.4, 10.3, 15.4, 25.5), ("h16", "b1 s4k", 15.9, 21.0, 27.0, 62.8), ("h16", "b1 s8k", 22.9, 27.9, 37.0, 83.8),
          ("h16", "b64 s4k", 370.9, 371.3, 529.6, 1405.2), ("h16", "b128 s8k", 1404.4, 1410.8, 1928.5, 5144.6),
          ("h128", "b1 s8k", 58.8, 69.5, 128.8, 90.2), ("h128", "b64 s4k", 1499.5, 1779.9, 3742.6, 2631.0),
          ("h128", "b64 s8k", 2885.2, 3509.2, 7733.4, 5134.6), ("h128", "b128 s8k", 5820.5, 6991.5, 15777.2, 10254.9)]
PX, PYT, PH = 86, BY2 + 78, 250
pbase = PYT + PH
ymax = 4.3
s.line(PX - 6, pbase, PX + 1030, pbase, INK, 1.8, rough=0.6)
s.line(PX - 6, pbase, PX - 6, PYT - 8, INK, 1.6, rough=0.6)
for t in (1, 2, 3, 4):
    ty = pbase - PH * t / ymax
    s.line(PX - 11, ty, PX + 1030, ty, RED if t == 1 else "#ced4da", 1.6 if t == 1 else 1.0, rough=0.3, passes=1, dash=None if t == 1 else "3 6")
    s.text(PX - 14, ty + 5, f"{t}x", 12.5, RED if t == 1 else GRAY, "end", bold=(t == 1))
s.text(PX + 1034, pbase - PH / ymax - 6, "v2", 13.5, RED, "end", bold=True)
cols3 = [("legacy kernel (smem-fixed fork)", "yellow"), ("Triton (SGLang / vLLM)", "purple"), ("FlashInfer 0.6.6", "gray")]
gwid = 1030 / 9
for i, (hq, lab, v2, leg, tri, fi) in enumerate(shapes):
    gx = PX + i * gwid + 12
    for k, v in enumerate((leg, tri, fi)):
        r = v / v2
        col = cols3[k][1]
        h = PH * r / ymax
        s.rect(gx + k * 30, pbase - h, 26, h, STROKE[col], FILL[col], 1.5, r=2, rough=0.5, hatch="#adb5bd" if k == 2 else None, hatch_gap=6)
        s.text(gx + k * 30 + 13, pbase - h - 6, f"{r:.2f}", 11.5, STROKE[col], "middle", bold=True)
    s.text(gx + 43, pbase + 20, lab, 13, INK, "middle", bold=True)
    s.text(gx + 43, pbase + 37, f"v2: {v2:.1f} us" if v2 < 100 else f"v2: {v2:.0f} us", 12, GRAY, "middle")
xm = PX + 5 * gwid + 2
s.line(xm, PYT - 10, xm, pbase + 62, GRAY, 1.2, rough=0.4, passes=1, dash="5 5")
s.text(PX + 2.5 * gwid, pbase + 60, "h_q = 16: memory-bound (bandwidth)", 14, BLUE, "middle", bold=True)
s.text(PX + 7 * gwid, pbase + 60, "h_q = 128: compute-bound (tensor cores)", 14, ORANGE, "middle", bold=True)
for k, (name, col) in enumerate(cols3):
    x = PX + 8 + k * 300
    s.rect(x, BY2 + 34, 26, 16, STROKE[col], FILL[col], 1.4, r=2, rough=0.4, hatch="#adb5bd" if k == 2 else None, hatch_gap=6)
    s.text(x + 34, BY2 + 47, name, 13.5, INK)
s.lines(PX - 30, pbase + 92, [("geometric mean over the full matrix (graph):", INK, True),
                              "h_q = 16:   legacy 1.07x (1.00-1.32),  Triton 1.47x (1.34-1.70),  FlashInfer 3.43x (2.70-4.18)",
                              "h_q = 128:  legacy 1.17x (1.02-1.23),  Triton 2.23x (1.17-2.71),  FlashInfer 1.77x (1.54-2.18)",
                              ("the original fork does not launch on a 3090 at all: it asks for 113,280 B of shared memory, the limit is 101,376 B", RED)],
        13.5, lh=1.45, maxw=1090)

# =============================================================================================== panel C
CX2 = 24 + 1130 + 16
CW2 = 1956 - CX2
s.group(CX2, BY2, CW2, BH2, "C  How close to the hardware", stroke=GREEN, size=17)
gx0, gwd = CX2 + 26, CW2 - 150
gauges = [("h_q=16, b >= 64: bytes per second", "94-99% of the 873.8 GB/s read peak", 0.94, 0.99, "blue", "target was 80%"),
          ("h_q=128: tensor throughput", "73-75% of the 70 TFLOPS bf16 peak", 0.73, 0.75, "orange", "target was 60%"),
          ("h_q=128, FP16 inputs", "77% of the FP16 peak (66.4 TFLOPS)", 0.77, 0.77, "orange", ""),
          ("original kernel on an A100 (author's numbers)", "23% of the bandwidth, 19% of the tensor peak", 0.19, 0.23, "red", "")]
for i, (t, sub, lo, hi, col, tgt) in enumerate(gauges):
    y = BY2 + 46 + i * 66
    s.text(gx0, y, t, 14, INK, bold=True)
    s.rect(gx0, y + 9, gwd, 22, GRAY, "#f1f3f5", 1.3, r=3, rough=0.4)
    s.rect(gx0, y + 9, gwd * hi, 22, STROKE[col], FILL[col], 1.6, r=3, rough=0.5, hatch={"blue": "#74c0fc", "orange": "#ffa94d", "red": "#ffa8a8"}[col], hatch_gap=7)
    s.out.append(f'<rect x="{gx0 + 8}" y="{y + 12}" width="{text_width(sub, 13) + 10:.1f}" height="17" fill="{FILL[col]}" opacity="0.93"/>')
    s.text(gx0 + 13, y + 25, sub, 13, INK)
    s.text(gx0 + gwd + 8, y + 25, "100%", 12.5, GRAY)
    if tgt:
        txx = gx0 + gwd * (0.80 if i == 0 else 0.60)
        s.line(txx, y + 4, txx, y + 36, RED, 2.0, rough=0.4)
        s.text(txx - 4, y + 48, tgt, 11.5, RED, "middle")
y2 = BY2 + 46 + 4 * 66 + 8
s.text(gx0, y2, "8-bit KV cache vs bf16 (h_q = 16, b >= 32, s_k >= 4096)", 14, INK, bold=True, maxw=CW2 - 40)
s.lines(gx0, y2 + 22, ["bytes per token: 644 B vs 1152 B = 0.559x  ->  1.79x would be ideal",
                       ("measured: FP8 1.55-1.70x,  INT8 1.54-1.60x", GREEN, True),
                       "accuracy (cos_diff): FP8 5e-5..2.4e-3, INT8 6e-6..2.5e-4"], 13.5, lh=1.45, maxw=CW2 - 40)
y3 = y2 + 98
s.text(gx0, y3, "What is still open: small batches", 14, INK, bold=True)
s.lines(gx0, y3 + 22, ["at b = 1 a layer moves 32 MB = 37 us of bandwidth, but takes 61.5 us:",
                       "27 us are launches, cold starts, idle DRAM and a write-back",
                       ("phase 3 (planned) takes back 8-10 us of it: 61.1 -> 51-53.5 us", ORANGE, True)], 13.5, lh=1.45, maxw=CW2 - 40)

s.save("fig4_results.svg")
print("fig4 ok; panel C text bottom =", y3 + 22 + 3 * 20, "group bottom =", BY2 + BH2)
