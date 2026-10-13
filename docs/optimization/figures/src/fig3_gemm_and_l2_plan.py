"""Fig. 3 - phase 2: the weight-streaming skinny GEMM; phase 3 (planned): what DRAM and L2 do between the kernels."""
from sketch import FILL, INK, STROKE, Sketch, text_width

W, H = 1980, 1476
s = Sketch(W, H, seed=37)
BLUE, GREEN, ORANGE, RED, GRAY, PURPLE, TEAL, YELLOW = (STROKE[k] for k in ("blue", "green", "orange", "red", "gray", "purple", "teal", "yellow"))

s.text(28, 48, "Fig. 3  Phase 2: the weight-streaming skinny GEMM.   Phase 3 (planned): what DRAM and L2 do between the kernels", 27, bold=True)
s.text(30, 80, "The four projections of a decode layer multiply 1..64 tokens by 2-15 MB of weights: they are weight-streaming problems, not matrix-multiply "
               "problems.", 16, GRAY)

# =============================================================================================== panel A
s.group(24, 112, 1932, 438, "A  Skinny GEMM (phase 2, implemented):  y[t, g, :] = x[t, g, :] @ W[g]^T  for T <= 64 tokens", stroke=GREEN, size=17)
MX, MY, MW, MH = 78, 186, 500, 250
NCOL, NBAND = 20, 5
cw, bh = MW / NCOL, MH / NBAND
s.text(MX + MW / 2, MY - 12, "K   (2048 for qkv_a and o_proj;  128 or 512 for the per-head matrices)", 13.5, INK, "middle")
s.text(MX - 14, MY + MH / 2, "N weight rows", 13.5, INK, "middle", rot=-90)
s.rect(MX, MY, MW, MH, PURPLE, FILL["purple"], 1.8, r=0, rough=0.7)
for b in range(1, NBAND):
    s.line(MX, MY + b * bh, MX + MW, MY + b * bh, PURPLE, 1.0, rough=0.4, passes=1, opacity=0.6)
for c in range(1, NCOL):
    s.line(MX + c * cw, MY, MX + c * cw, MY + MH, PURPLE, 0.8, rough=0.3, passes=1, opacity=0.35)
band = 2
yb = MY + band * bh
for c in range(NCOL):
    xc = MX + c * cw
    if c < 6:
        s.rect(xc + 1, yb + 1, cw - 2, bh - 2, GRAY, "#dee2e6", 1.0, r=0, rough=0.3, passes=1)
    elif c == 6:
        s.rect(xc + 1, yb + 1, cw - 2, bh - 2, GREEN, "#8ce99a", 1.6, r=0, rough=0.3)
    elif c <= 13:
        s.rect(xc + 1, yb + 1, cw - 2, bh - 2, TEAL, FILL["teal"], 1.2, r=0, rough=0.3, passes=1, hatch="#3bc9db", hatch_gap=5)
s.rect(MX - 3, yb - 3, MW + 6, bh + 6, BLUE, None, 2.6, r=0, rough=0.8)
s.arrow([(MX + 8, yb + bh + 16), (MX + MW - 6, yb + bh + 16)], BLUE, 2.2, head=10)
s.text(MX + MW / 2, yb + bh + 36, "the CTA sweeps along K: 64 elements = one 128-B line per row and stage", 13, BLUE, "middle", bold=True, maxw=MW)
# callouts
s.text(MX + 3 * cw, yb - 12, "done", 13, GRAY, "middle", bold=True)
s.text(MX + 6.5 * cw, yb - 12, "in use", 13, GREEN, "middle", bold=True)
s.text(MX + 10.6 * cw, yb - 12, "in flight (cp.async ring)", 13, TEAL, "middle", bold=True)
s.text(MX + 17 * cw, yb + 30, "not yet", 13, PURPLE, "middle")
s.text(MX + 17 * cw, yb + 47, "requested", 13, PURPLE, "middle")
s.lines(MX, MY + MH + 26, [("blue band = one CTA = n_tile weight rows of one head", BLUE, True),
                           "n_tile = 32 (16 for the per-head matrices at T <= 16)",
                           "grid = (N / n_tile, 1, heads), 128 threads per CTA",
                           "64-512 CTAs for 82 SMs, each with up to 48 KB in flight"], 13.5, lh=1.42, maxw=540)

# CTA pipeline
PX, PW = 640, 520
steps = [("1  cp.async ring: 48 KB of shared memory", ["one stage = 64 K-elements of 32 weight rows + the tokens: 40 x 128 B",
                                                        "3..8 stages in flight, rows XOR-swizzled for ldmatrix"], "teal"),
         ("2  mma.sync m16n8k16 with A and B swapped", ["A = 16 weight rows (m16), B = tokens (n8): 8 tokens cost one n8",
                                                         "bf16 or fp16 in, fp32 accumulate"], "blue"),
         ("3  the 4 warps split the stage along K", ["64 elements = 4 x k16, one k16 step per warp:",
                                                      "no warp waits for another one inside a stage"], "blue"),
         ("4  epilogue: one deterministic rounding", ["fixed-order sum of the 4 warps in smem, then bf16 / fp16",
                                                       "strided store: W_UV writes the o_proj input in place"], "orange")]
for i, (t, ls, col) in enumerate(steps):
    y = 140 + i * 100
    s.box(PX, y, PW, 86, t, ls, fill=FILL[col], stroke=STROKE[col], tsize=15.5, size=13.5, align="left", lh=1.36, pad=10)
    if i < 3:
        s.arrow([(PX + PW / 2, y + 87), (PX + PW / 2, y + 99)], INK, 2, head=7, rough=0.3)
s.arrow([(MX + MW + 6, yb + bh / 2), (PX - 6, 183)], TEAL, 2.2, head=9)

# comparison table
TX = 1200
s.text(TX, 152, "cuBLAS vs skinny GEMM at 8 tokens", 16, GREEN, bold=True)
hdr = ["GEMM", "weights", "cuBLAS: CTAs, share of DRAM peak", "skinny: CTAs"]
rowsT = [("qkv_a", "14.9 MB", "57 CTAs,   61-63%", "114"), ("W_UK (16 heads)", "2.1 MB", "256 small CTAs,   19-39%", "512"),
         ("W_UV (16 heads)", "2.1 MB", "32 CTAs,   22-38%", "128"), ("o_proj", "8.4 MB", "64 CTAs,   51-63%", "64, deep ring")]
colx = [TX, TX + 150, TX + 240, TX + 560]
for c, t in enumerate(hdr):
    s.text(colx[c], 182, t, 13.5, INK, bold=True)
s.line(TX - 4, 190, TX + 720, 190, GRAY, 1.2, rough=0.4, passes=1)
for r, row in enumerate(rowsT):
    for c, t in enumerate(row):
        s.text(colx[c], 214 + r * 26, t, 13.5, INK if c != 2 else RED)
s.lines(TX, 330, [("cuBLAS opens fewer CTAs than there are SMs (82) and keeps little in flight;", INK),
                  ("the skinny kernel streams at ~870 GB/s once it is running.", INK)], 13.5, lh=1.4, maxw=730)
res = [("the four GEMMs, b = 8:", "61.7 -> 41.5 us   (bytes / peak bandwidth = 31.5 us)"),
       ("layer, b = 8, s_k = 4096:", "115.4 -> 102.0 us bf16 (1.13x),   100.4 -> 86.1 us FP8 (1.17x)"),
       ("layer, b = 64:", "1.01-1.05x: attention dominates there"),
       ("rules from the sweep:", "no split-K;  above 32 tokens qkv_a / o_proj go back to cuBLAS")]
for i, (a, b) in enumerate(res):
    s.text(TX, 396 + i * 27, a, 13.5, GREEN, bold=True)
    s.text(TX + 190, 396 + i * 27, b, 13.5, INK, maxw=545)
s.text(TX, 396 + 4 * 27 + 6, "what is left per GEMM: ~2.5 us of launch + start-up + tail  ->  phase 3 below", 13.5, ORANGE, bold=True, maxw=730)

# =============================================================================================== panel B
BY = 578
s.group(24, BY, 1932, 876, "B  Phase 3 (planned): one layer on the time axis, before and after   (batch 1, 4096 cached tokens, bf16)", stroke=ORANGE, size=17)
X0, SC = 196, 26.6
LBL = 44


def lane_label(y, t, col=INK, sub=None):
    s.text(LBL, y, t, 14.5, col, bold=True)
    if sub:
        s.text(LBL, y + 17, sub, 12, GRAY)


def bars(y, h, segs, small=13.5):
    x = X0
    out = []
    for name, t, col, extra in segs:
        w = t * SC
        s.rect(x, y, w - 3, h, STROKE[col], FILL[col], 1.7, r=5, rough=0.6, dash=extra.get("dash"))
        lab = extra.get("label", name)
        if text_width(lab, small, True) < w - 10:
            s.text(x + (w - 3) / 2, y + h / 2 - 2, lab, small, STROKE[col], "middle", bold=True)
            if "t" in extra:
                s.text(x + (w - 3) / 2, y + h / 2 + 15, extra["t"], 12.5, INK, "middle")
        else:
            s.text(x + (w - 3) / 2, y - 8, lab, 12.5, STROKE[col], "middle", bold=True)
            if "t" in extra:
                s.text(x + (w - 3) / 2, y + h / 2 + 5, extra["t"], 12, INK, "middle")
        out.append((x, w))
        x += w
    return out


# ---------------------------------------------------------------- before
y0 = BY + 52
s.text(LBL, y0 - 8, "BEFORE (phase-2 build): 61.1 us per layer", 17, INK, bold=True)
s.text(LBL + 392, y0 - 8, "bar lengths to scale (nsys: 64.2 us including ~0.4 us of tracing per kernel)", 13, GRAY)
lane_label(y0 + 44, "kernels")
before = [("qkv_a", 20.80, "blue", {"t": "20.8 us"}), ("W_UK", 4.77, "blue", {"t": "4.8"}), ("prologue", 2.40, "green", {"t": "2.4"}),
          ("decode", 12.64, "orange", {"t": "12.6 us"}), ("combine", 4.29, "orange", {"t": "4.3"}), ("W_UV", 4.87, "blue", {"t": "4.9"}),
          ("o_proj", 14.46, "blue", {"t": "14.5 us"})]
kb = bars(y0 + 18, 50, before)
# DRAM lane
lane_label(y0 + 104, "DRAM", PURPLE, "873.8 GB/s peak")
dram_b = [("read W_qkv_a (14.9 MB)", "purple", None), ("W_UK", "purple", None), (None, None, "idle"), ("read KV (4.7 MB)", "purple", None),
          (None, None, "idle"), ("W_UV", "purple", None), ("read W_o (8.4 MB)", "purple", None)]
for (x, w), (lab, col, idle) in zip(kb, dram_b):
    if idle:
        s.rect(x, y0 + 84, w - 3, 30, GRAY, None, 1.2, r=3, dash="4 5", rough=0.4, passes=1)
        s.text(x + (w - 3) / 2, y0 + 104, idle, 12.5, GRAY, "middle")
    else:
        s.rect(x, y0 + 84, w - 3, 30, PURPLE, FILL["purple"], 1.4, r=3, rough=0.5, hatch="#d0bfff", hatch_gap=8)
        if text_width(lab, 12.5) < w - 8:
            s.out.append(f'<rect x="{x + (w - 3) / 2 - text_width(lab, 12.5) / 2 - 4:.1f}" y="{y0 + 90}" width="{text_width(lab, 12.5) + 8:.1f}" height="18" fill="{FILL["purple"]}" opacity="0.9"/>')
            s.text(x + (w - 3) / 2, y0 + 104, lab, 12.5, INK, "middle")
        else:
            s.text(x + (w - 3) / 2, y0 + 104, lab, 11.5, INK, "middle")
xw0, xw1 = kb[5][0], kb[6][0] + kb[6][1] * 0.72
s.rect(xw0, y0 + 118, xw1 - xw0, 22, RED, FILL["red"], 1.5, r=3, rough=0.5, hatch="#fa5252", hatch_gap=6)
s.out.append(f'<rect x="{xw0 + 30}" y="{y0 + 121}" width="{text_width("write-back of the 2.6 MB of partial O", 12.5, True) + 12:.1f}" height="16" fill="{FILL["red"]}" opacity="0.92"/>')
s.text(xw0 + 36, y0 + 134, "write-back of the 2.6 MB of partial O", 12.5, RED, bold=True)
# L2 lane
lane_label(y0 + 174, "L2 cache", TEAL, "6 MB")
yl0 = y0 + 158
xd0, xd1 = kb[3][0] + kb[3][1] * 0.45, kb[4][0] + kb[4][1]
s.rect(xd0, yl0, xd1 - xd0 - 3, 28, TEAL, FILL["teal"], 1.5, r=3, rough=0.5)
s.text((xd0 + xd1) / 2, yl0 + 19, "partial O (2.6 MB, fp32)", 12.5, TEAL, "middle", bold=True)
s.rect(xd1, yl0, xw1 - xd1, 28, RED, FILL["red"], 1.5, r=3, rough=0.5, hatch="#ffa8a8", hatch_gap=7)
_t = "dead after the combine, but still dirty"
s.out.append(f'<rect x="{(xd1 + xw1) / 2 - text_width(_t, 12.5, True) / 2 - 6:.1f}" y="{yl0 + 5}" width="{text_width(_t, 12.5, True) + 12:.1f}" height="18" fill="{FILL["red"]}" opacity="0.92"/>')
s.text((xd1 + xw1) / 2, yl0 + 19, _t, 12.5, RED, "middle", bold=True)
s.arrow([(xw0 + 14, yl0 - 2), (xw0 + 14, y0 + 143)], RED, 2.0, head=7, rough=0.3)
# numbered problems
probs = [(kb[1][0] + 4, y0 + 10, "1"), (kb[2][0] + kb[2][1] / 2 - 2, y0 + 134, "2"), (kb[3][0] + kb[3][1] - 26, y0 + 10, "3"), (xw1 + 18, y0 + 172, "4")]
for (x, y, n) in probs:
    s.ellipse(x, y, 12, 12, RED, "#ffffff", 1.8, rough=0.5)
    s.text(x, y + 5, n, 14, RED, "middle", bold=True)
py = y0 + 216
ptxt = [("1", "every kernel starts cold: its first weight lines are requested only after the launch, so each GEMM runs 2.4-4.9 us over its byte floor"),
        ("2", "prologue and combine are latency-bound (dependent loads, reductions): DRAM is idle for 2.4 + 4.3 us"),
        ("3", "130 work units cut into 82 parts: 48 parts carry 2 units, 34 carry 1  ->  uneven tails and 82 partial rows to merge"),
        ("4", "the partials are dead after the combine, but still dirty: they are written back while W_UV and o_proj need the bandwidth")]
for i, (n, t) in enumerate(ptxt):
    yy = py + i * 24
    s.ellipse(LBL + 10, yy - 5, 10.5, 10.5, RED, "#ffffff", 1.6, rough=0.5)
    s.text(LBL + 10, yy, n, 13, RED, "middle", bold=True)
    s.text(LBL + 30, yy, t, 13.5, INK, maxw=1860)

# ---------------------------------------------------------------- after
y1 = py + 4 * 24 + 34
s.line(LBL, y1 - 26, 1936, y1 - 26, GRAY, 1.2, dash="2 7", passes=1)
s.text(LBL, y1, "AFTER (plan):  56.2 us measured with P1 + P2 + P3,  51-53.5 us projected with all six", 17, ORANGE, bold=True)
s.text(LBL + 800, y1, "bar lengths are illustrative: only layer totals were measured", 13, GRAY)
lane_label(y1 + 56, "kernels")
after = [("qkv_a", 20.2, "blue", {}), ("W_UK + prologue", 5.0, "green", {"label": "W_UK+prologue"}),
         ("decode", 11.4, "orange", {"label": "decode: 65 balanced parts"}), ("combine", 4.3, "orange", {}), ("W_UV", 3.4, "blue", {}),
         ("o_proj", 11.5, "blue", {})]
ka = bars(y1 + 26, 50, after)
s.badge(ka[1][0] + ka[1][1] / 2, y1 + 92, "P4: one launch", FILL["orange"], ORANGE, 12.5)
s.badge(ka[2][0] + ka[2][1] / 2, y1 + 92, "P2: 82 -> 65 parts", FILL["orange"], ORANGE, 12.5)

# DRAM lane: the kernel's own stream; teal = served from L2 because it was prefetched
yd = y1 + 122
lane_label(yd + 16, "DRAM", PURPLE, "the kernel's own stream")


def own(x, w, lab, col, hatch):
    s.rect(x, yd, w, 28, STROKE[col], FILL[col], 1.4, r=3, rough=0.5, hatch=hatch, hatch_gap=8)
    if lab and text_width(lab, 12.5) < w - 6:
        s.out.append(f'<rect x="{x + w / 2 - text_width(lab, 12.5) / 2 - 4:.1f}" y="{yd + 5}" width="{text_width(lab, 12.5) + 8:.1f}" height="18" fill="{FILL[col]}" opacity="0.92"/>')
        s.text(x + w / 2, yd + 19, lab, 12.5, INK, "middle")


def split(k, frac, lab):
    x, w = ka[k][0], ka[k][1] - 3
    own(x, w * frac, "L2", "teal", None)
    own(x + w * frac + 2, w * (1 - frac) - 2, lab, "purple", "#d0bfff")


split(0, 2.0 / 14.9, "W_qkv_a: 12.9 MB from DRAM")
own(ka[1][0], ka[1][1] - 3, "W_UK from L2", "teal", None)
own(ka[2][0], ka[2][1] - 3, "KV: 4.7 MB from DRAM", "purple", "#d0bfff")
s.rect(ka[3][0], yd, ka[3][1] - 3, 28, GRAY, None, 1.2, r=3, dash="4 5", rough=0.4, passes=1)
s.text(ka[3][0] + (ka[3][1] - 3) / 2, yd + 19, "idle", 12.5, GRAY, "middle")
own(ka[4][0], ka[4][1] - 3, "from L2", "teal", None)
split(5, 2.0 / 8.4, "W_o: 6.4 MB from DRAM")

# prefetch lane
yp = yd + 44
lane_label(yp + 15, "prefetch", ORANGE, "for the next kernel")


def pf(x, w, lab, tgt):
    s.rect(x, yp, w, 24, ORANGE, FILL["orange"], 1.5, r=3, rough=0.5, hatch="#ffc078", hatch_gap=6)
    s.out.append(f'<rect x="{x + w / 2 - text_width(lab, 12.5, True) / 2 - 4:.1f}" y="{yp + 3}" width="{text_width(lab, 12.5, True) + 8:.1f}" height="18" fill="{FILL["orange"]}" opacity="0.92"/>')
    s.text(x + w / 2, yp + 17, lab, 12.5, ORANGE, "middle", bold=True)
    if tgt is not None:
        s.arrow([(x + w + 2, yp + 10), (tgt + 12, yp + 6), (tgt + 14, yd + 30)], ORANGE, 2.0, head=7, smooth=True)


e0 = ka[0][0] + ka[0][1] - 3
pf(e0 - 70, 70, "W_UK", ka[1][0])
pf(ka[3][0], ka[3][1] - 3, "W_UV (P6)", ka[4][0])
pf(ka[4][0] + 22, ka[4][1] - 25, "W_o", ka[5][0])
e5 = ka[5][0] + ka[5][1] - 3
pf(e5 - 84, 84, "W_qkv_a", None)
s.arrow([(e5 + 4, yp + 12), (e5 + 44, yp + 12)], ORANGE, 2.0, head=8, rough=0.3)
s.text(e5 + 52, yp + 10, "first 2 MB of the next", 12.5, ORANGE, bold=True)
s.text(e5 + 52, yp + 26, "layer's weights", 12.5, ORANGE, bold=True)
s.text(ka[0][0] + 6, yp + 17, "P3: issued with the last weight load, by all threads; 2 MB budget", 12.5, ORANGE,
       maxw=e0 - 80 - ka[0][0])

# L2 lane
yl = yp + 44
lane_label(yl + 18, "L2 cache", TEAL)
xa0, xa1 = ka[2][0] + ka[2][1] * 0.45, ka[3][0] + ka[3][1] - 3
s.rect(xa0, yl, xa1 - xa0, 28, TEAL, FILL["teal"], 1.5, r=3, rough=0.5)
s.text((xa0 + xa1) / 2, yl + 19, "partial O", 12.5, TEAL, "middle", bold=True)
s.line(xa1 + 6, yl - 4, xa1 + 30, yl + 32, RED, 3, rough=0.6)
s.line(xa1 + 30, yl - 4, xa1 + 6, yl + 32, RED, 3, rough=0.6)
s.text(xa1 + 42, yl + 11, "P1: discard.global.L2 right after the read:", 13, RED, bold=True)
s.text(xa1 + 42, yl + 28, "the lines are dropped, nothing is written back", 13, INK)
s.text(ka[0][0] + 6, yl + 19, "P3: weights are loaded evict_first; prefetched lines are marked evict_last", 12.5, TEAL)

# ---------------------------------------------------------------- bottom boxes
yb2 = yl + 52
bw3 = 616
bx0 = LBL - 6
s.rect(bx0, yb2, bw3, 196, ORANGE, "#ffffff", 1.8, r=10, rough=0.7)
s.text(bx0 + 12, yb2 + 25, "The six items  (us per layer, batch 1 and batch 8)", 15, ORANGE, bold=True)
c6 = [bx0 + 12, bx0 + 312, bx0 + 404, bx0 + 486]
for c, t in enumerate(["", "b = 1", "b = 8", "evidence"]):
    s.text(c6[c], yb2 + 47, t, 12.8, GRAY, bold=True)
rows6 = [("P1  discard the partials after the combine", "-3.0", "-2.8", "in-layer A/B"),
         ("P2  balanced part count", "-1.0 .. -1.7", "-1.0 .. -1.2", "in-layer"),
         ("P3  GEMM -> GEMM prefetch chain", "-1.7 .. -1.9", "-1.4", "in-layer, with P1"),
         ("P4  W_UK + prologue in one launch", "-1.5 .. -2.4", "-0.8 .. -2.3", "emulated"),
         ("P5  no positions / slots on the fused path", "-0.6", "-0.4", "in-layer"),
         ("P6  prefetch W_UV from the combine", "-0.85 .. -1.3", "-", "synthetic only")]
for r, row in enumerate(rows6):
    for c, t in enumerate(row):
        s.text(c6[c], yb2 + 68 + r * 19.5, t, 12.8, INK, maxw=(c6[c + 1] - c6[c] - 4) if c < 3 else 124)
s.text(bx0 + 12, yb2 + 186, "output bit-identical for all but P2 (split boundaries move; within test tolerance)", 12.5, GRAY, maxw=bw3 - 20)
s.box(LBL + bw3 + 10, yb2, bw3, 196, "What prefetch.global.L2 needs on this GPU", [
    "it blocks the issuing thread: spread the lines over all threads of the CTA",
    "(9 of 128 threads issuing = 300-400 GB/s and a false negative)",
    "the issuing kernel does not retire until its prefetches have landed:",
    "a separate prefetch kernel loses 0.6-0.8 us",
    "issue it with the last weight load, not after the epilogue (1.1-1.5 us better)",
    "it pays only inside a latency-bound or DRAM-idle window",
    "without P1 the write-back eats the gain at b = 4 and b = 8"],
      fill="#ffffff", stroke=TEAL, tsize=15, size=12.8, align="left", lh=1.52, pad=10)
s.box(LBL + 2 * (bw3 + 10), yb2, 1936 - (LBL + 2 * (bw3 + 10)) - 4, 196, "Did not survive the real layer", [
    ("x  KV prefetch before the decode", RED, True),
    "    synthetic bench -0.9 .. -1.9 us, inside the layer +0.1 .. +1.7 us",
    ("x  page-number hints, warming metadata / block table", RED, True),
    "    no gain: that hop is not on the critical path",
    ("x  partial O stored as bf16", RED, True),
    "    superseded by P1 and it would change the numerics",
    ("=> every item is judged by an A/B on the real 27-layer graph", INK, True)],
      fill="#ffffff", stroke=RED, tsize=15, size=12.8, align="left", lh=1.52, pad=10)

s.save("fig3_gemm_and_l2_plan.svg")
print("fig3 ok; bottom =", yb2 + 196, "group bottom =", BY + 876, "H =", H)
