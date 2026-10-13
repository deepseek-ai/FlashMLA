"""Fig. 2 - the v2 decode kernel: split-KV scheduler, one CTA from the inside, combine, and the 8-bit KV variant."""
from sketch import FILL, INK, STROKE, Sketch, text_width

W, H = 1980, 1336
s = Sketch(W, H, seed=23)
BLUE, GREEN, ORANGE, RED, GRAY, PURPLE, TEAL, YELLOW = (STROKE[k] for k in ("blue", "green", "orange", "red", "gray", "purple", "teal", "yellow"))

s.text(28, 48, "Fig. 2  The v2 decode kernel: split-KV scheduling, one CTA from the inside, combine, and the 8-bit KV variant", 28, bold=True)
s.text(30, 80, "MLA decode: q [b, s_q, h_q, 576] attends to a paged cache of 576-dim tokens (512 latent + 64 RoPE; K and V share the latent); the output "
               "has 512 dims per head.", 16, GRAY)

# =============================================================================================== panel A: scheduler
AX, AW = 24, 524
s.group(AX, 112, AW, 1206, "A  Scheduler: all requests -> at most 82 equal parts", stroke=BLUE, size=17)
x0 = AX + 22
s.text(x0, 152, "example: 3 requests with 4096 / 700 / 2000 cached tokens", 14.5, GRAY)
# the unit line
units = [("request 0: 128 + 2", 130, "blue"), ("22+2", 24, "green"), ("request 2: 63 + 2", 65, "orange")]
T = sum(u for _, u, _ in units)
bw, bx, by = 480.0, x0, 178
x = bx
for name, u, col in units:
    w = bw * u / T
    s.rect(x, by, w, 30, STROKE[col], FILL[col], 1.6, r=0, rough=0.6)
    s.rect(x, by, 9, 30, STROKE[col], None, 1.2, r=0, rough=0.4, hatch=STROKE[col], hatch_gap=4)
    s.text(x + w / 2 + 4, by + 20, name, 13, STROKE[col], "middle", bold=True)
    x += w
for p in range(83):   # 82 parts
    px = bx + bw * p / 82.0
    s.line(px, by + 33, px, by + 41, INK, 1.0, rough=0.2, passes=1)
s.text(x0, by + 60, "82 parts of 2-3 units each; one unit = a 32-token block; hatched = +2 units", 13.5, INK, maxw=AW - 30)
s.text(x0, by + 78, "of fixed per-request overhead (not to scale)", 13.5, INK)
s.lines(x0, by + 110, [("cost_i = max(1, ceil(seqlen_i / 32)) + 2", BLUE, True),
                       ("T = sum of all costs;   Pe = min(82, T) parts", BLUE, True),
                       ("part p = units [ floor(p*T/Pe), floor((p+1)*T/Pe) )", BLUE, True)], 15, lh=1.5, maxw=AW - 30)
s.lines(x0, by + 184, ["Every part is located in parallel by a binary search on the",
                       "prefix sums, in one small kernel on the GPU: no host sync,",
                       "so the whole decode can be captured in a CUDA graph."], 14, lh=1.38, maxw=AW - 30)
# metadata row
my = by + 258
s.text(x0, my, "metadata row of part p (8 ints, read by its CTA as 2 x int4)", 14.5, INK, bold=True, maxw=AW - 30)
cells = ["begin_idx", "begin_seqlen", "end_idx", "end_seqlen", "first split", "seqlen_k[begin]", "units_per_part", "0"]
for i, c in enumerate(cells):
    cxx, cyy = x0 + (i % 4) * 120, my + 12 + (i // 4) * 40
    hl = c in ("seqlen_k[begin]", "units_per_part")
    s.rect(cxx, cyy, 116, 34, YELLOW if hl else GRAY, FILL["yellow"] if hl else FILL["gray"], 1.4, r=4, rough=0.5)
    s.text(cxx + 58, cyy + 22, c, 13, INK, "middle", maxw=112)
s.lines(x0, my + 112, ["seqlen_k[begin]: saves the CTA one dependent load at start-up",
                       "units_per_part: one launch-wide value; the 8-bit kernel uses it to",
                       "decide for all CTAs at once whether to run two KV groups"], 13.5, lh=1.38, maxw=AW - 30)
# 82 SMs
sy = my + 186
s.text(x0, sy, "part p -> persistent CTA p, one per SM (82 SMs)", 14.5, INK, bold=True)
b0, b1 = 82 * 130 / T, 82 * 154 / T
for p in range(82):
    sx_, sy_ = x0 + (p % 21) * 23, sy + 14 + (p // 21) * 23
    col = "blue" if p + 1 <= b0 else ("green" if p + 1 <= b1 else "orange")
    if p < b0 < p + 1 or p < b1 < p + 1:
        col = "yellow"
    s.rect(sx_, sy_, 19, 19, STROKE[col], FILL[col], 1.2, r=0, rough=0.35, passes=1)
s.lines(x0, sy + 126, ["A CTA walks its token range, possibly across a request boundary",
                       "(yellow). Requests cut into several parts = split-KV: each part",
                       "writes a partial result, merged later by the combine kernel."], 13.5, lh=1.38, maxw=AW - 30)
# what a CTA does
wy = sy + 200
s.text(x0, wy, "start-up chain of every CTA", 14.5, INK, bold=True)
steps = ["metadata row", "block table (page ids)", "first KV tile (cp.async)"]
xx = x0
for i, t in enumerate(steps):
    wbox = text_width(t, 13.5) + 18
    s.rect(xx, wy + 12, wbox, 30, TEAL, FILL["teal"], 1.4, r=6, rough=0.5)
    s.text(xx + wbox / 2, wy + 32, t, 13.5, INK, "middle")
    xx += wbox
    if i < 2:
        s.arrow([(xx + 3, wy + 27), (xx + 19, wy + 27)], width=1.8, head=6, rough=0.3)
        xx += 23
s.lines(x0, wy + 66, ["Three serial memory round trips: ~4 us at b=1 before the first",
                      "tile arrives. After that the CTA streams at the DRAM peak, so",
                      "small batches pay for fixed costs, not for the stream."], 13.5, lh=1.38, maxw=AW - 30)
# BM32
gy = wy + 144
s.box(x0, gy, AW - 44, 104, "more than 16 query rows (BM32)", ["rows are cut into m-blocks of 32; h_q=128 runs",
                                                             "4 m-blocks x 20 parts, and the 4 m-blocks read the",
                                                             "same KV, shared through L2 (74.7% hit rate)"],
      fill=FILL["gray"], stroke=GRAY, tsize=15, size=13.5, align="left", lh=1.36)
# workspace layout
ly = gy + 136
s.text(x0, ly, "workspace (allocated once, reused by all layers and steps)", 14.5, INK, bold=True, maxw=AW - 30)
segs = [("counters", 70, "gray"), ("partial O: fp32, (b + parts) x rows x 512", 300, "teal"), ("LSE", 50, "teal"), ("tmp", 44, "gray")]
xx = x0
for t, wseg, col in segs:
    s.rect(xx, ly + 12, wseg, 32, STROKE[col], FILL[col], 1.4, r=0, rough=0.5)
    s.text(xx + wseg / 2, ly + 33, t, 12.5, INK, "middle", maxw=wseg - 4)
    xx += wseg + 4
# P2 note
s.note(x0 + 6, ly + 70, AW - 70, 108, ["Pe = ceil(T / ceil(T / P)): the fewest parts whose",
                                        "heaviest part is no heavier. b=1, s_k=4096: 82 -> 65",
                                        "parts, 20% fewer partial rows, -1.0 .. -1.7 us/layer"], title="P2 (planned, phase 3)", size=13.5, rot=-0.8)

# =============================================================================================== panel B: one CTA
BX, BW = 566, 838
s.group(BX, 112, BW, 1206, "B  Inside one CTA  (BM16: 16 query rows, 4 warps, 128 threads)", stroke=ORANGE, size=17)
# KV source + ring
s.box(BX + 20, 146, 196, 74, "paged KV cache", ["DRAM; 1 token =", "576 x bf16 = 1152 B"], fill=FILL["purple"], stroke=PURPLE, tsize=15, size=13, lh=1.25, pad=7)
RX, RW = BX + 300, 516
s.text(RX, 144, "smem ring: 3 stages x 16 tokens = 55.3 KB, XOR-swizzled rows", 13.5, TEAL, bold=True, maxw=RW + 4)
slots = ["tile c+2: in flight", "tile c+1: landed", "tile c: in use"]
for i, t in enumerate(slots):
    col = ("gray", "teal", "green")[i]
    s.rect(RX + i * 174, 154, 168, 50, STROKE[col], FILL[col], 1.6, r=6, rough=0.6, hatch="#ced4da" if i == 0 else None, hatch_gap=7)
    s.out.append(f'<rect x="{RX + i * 174 + 16}" y="166" width="136" height="26" fill="{FILL[col]}" opacity="0.9"/>')
    s.text(RX + i * 174 + 84, 185, t, 14, INK, "middle", bold=True)
s.arrow([(BX + 218, 180), (RX - 4, 180)], PURPLE, 2.2, head=9)
s.text(BX + 259, 172, "cp.async", 13.5, PURPLE, "middle", bold=True)
s.text(RX, 224, "issued by all warps; page id looked up one tile ahead; tail rows zero-filled", 12.5, GRAY, maxw=RW - 20)

# lanes
LX0, LW, LG = BX + 200, 144, 8


def lx(j):
    return LX0 + j * (LW + LG)


AXL = BX + 16      # left annotation column
rowsB = [  # (y, h, annotation lines, per-lane lines or None for a full-width bar, colour)
    (246, 70, [("Q lives in registers", ORANGE, True), "as MMA A-fragments,", "never in shared memory"],
     lambda j: [(f"warp {j}", ORANGE, True), f"Q dims {144 * j}..{144 * j + 143}", "36 regs, loaded once"], "orange"),
    (334, 62, [("1  QK: split-K per CTA", INK, True), "each warp owns 144 of", "the 576 dims"],
     lambda j: [("QK", BLUE, True), "9 x mma.sync (k16)", "-> partial S (16 x 16)"], "blue"),
    (414, 50, [("2  exchange + barrier", INK, True), "double-buffered smem,", "bar.sync (named)"], None, "teal"),
    (482, 62, [("3  softmax in every warp", INK, True), "same S everywhere, so", "no second exchange"],
     lambda j: [("online softmax", BLUE, True), "row max m, sum l", "rescale O if m grew"], "blue"),
    (562, 50, [("4  P stays in registers", INK, True), "no trip through smem"],
     lambda j: [("P -> bf16", BLUE, True), "A-fragments"], "blue"),
    (630, 62, [("5  PV: split by columns", INK, True), "V = the same tile, read", "transposed"],
     lambda j: [("PV", BLUE, True), f"V cols {128 * j}..{128 * j + 127}", "ldmatrix.trans"], "blue"),
    (710, 50, [("6  O accumulators", INK, True), "16 x 128 fp32 per warp"],
     lambda j: [("O: 64 regs", ORANGE, True), "fp32, in registers"], "orange"),
]
for (y, h, ann, lane, col) in rowsB:
    s.lines(AXL, y + 18, ann, 13.5, INK, lh=1.3, maxw=184)
    if lane is None:
        s.rect(lx(0), y, 4 * LW + 3 * LG, h, STROKE[col], FILL[col], 1.6, r=8, rough=0.6)
        s.text(lx(0) + (4 * LW + 3 * LG) / 2, y + 21, "S exchange buffer: every warp writes its partial S, then all add the 4 partials", 13.5, INK, "middle",
               maxw=4 * LW + 3 * LG - 10)
        s.text(lx(0) + (4 * LW + 3 * LG) / 2, y + 39, "in the same fixed order  ->  S is bit-identical in all 4 warps", 13.5, INK, "middle")
    else:
        for j in range(4):
            s.rect(lx(j), y, LW, h, STROKE[col], FILL[col], 1.6, r=8, rough=0.6)
            ls = lane(j)
            ty = y + (h - len(ls) * 17.5) / 2 + 13.5
            s.lines(lx(j) + LW / 2, ty, ls, 13, INK, "middle", lh=1.35, maxw=LW - 6)
for j in range(4):   # arrows between the bands
    for (ya, yb) in ((316, 334), (396, 414), (464, 482), (544, 562), (612, 630), (692, 710)):
        s.arrow([(lx(j) + LW / 2, ya + 1), (lx(j) + LW / 2, yb - 1)], INK, 1.7, head=6, rough=0.3, passes=1)
# tile feeds QK and PV
ex = lx(3) + LW
s.arrow([(RX + 2 * 174 + 150, 206), (ex + 10, 250), (ex + 13, 365), (ex + 2, 365)], GREEN, 2.0, head=8, smooth=True)
s.arrow([(ex + 13, 365), (ex + 15, 560), (ex + 12, 661), (ex + 2, 661)], GREEN, 2.0, head=8, smooth=True)
s.text(ex + 29, 512, "the same tile feeds QK and PV", 13, GREEN, "middle", bold=True, rot=-90)
# loop + epilogue
s.arrow([(AXL + 26, 776), (AXL + 34, 790), (AXL + 18, 794), (AXL + 6, 784), (AXL + 12, 772), (AXL + 22, 771)], GRAY, 1.8, head=6, smooth=True, passes=1)
s.text(AXL + 44, 788, "repeat per tile: wait_group, one CTA barrier per stage, refill the freed ring slot", 13.5, GRAY, bold=True, maxw=BW - 70)
s.box(BX + 20, 806, BW - 40, 66, "end of the part's token range", ["write partial O (fp32) + LSE into the workspace slot of this split  -  "
                                                                "or the final O directly when the request has only one split"],
      fill=FILL["teal"], stroke=TEAL, tsize=15, size=13.5, lh=1.3, pad=8)
# resources table
ry = 902
s.text(BX + 20, ry, "Why it fits: 101,376 B of shared memory per block on sm_86, 64 K registers per SM", 15, INK, bold=True, maxw=BW - 30)
tab = [("kernel instance", "regs / thread", "shared memory", "tile", True),
       ("original fork (Q + double-buffered K + P in smem)", "-", "113,280 B: cannot launch", "32 tokens, 2 stages", False),
       ("v2 BM16  (h_q * s_q <= 16)", "187", "59.4 KB", "16 tokens, 3 stages", False),
       ("v2 BM32  (2 M-groups x 4 warps)", "212", "90.0 KB", "32 tokens, 2 stages", False),
       ("v2 8-bit KV  (one / two KV groups)", "218 / 254", "49.4 KB (one group)", "32 tokens, 2 stages", False)]
colx = [BX + 20, BX + 400, BX + 506, BX + 676]
for r, row in enumerate(tab):
    yy = ry + 30 + r * 27
    if r == 1:
        s.out.append(f'<rect x="{BX + 16}" y="{yy - 18}" width="{BW - 32}" height="25" fill="{FILL["red"]}"/>')
    for c in range(4):
        s.text(colx[c], yy, row[c], 13.5, RED if r == 1 else INK, bold=row[4], maxw=(colx[c + 1] - colx[c] - 6) if c < 3 else BW - 690)
    if r == 0:
        s.line(BX + 16, yy + 7, BX + BW - 16, yy + 7, GRAY, 1.2, rough=0.4, passes=1)
s.text(BX + 20, ry + 30 + 5 * 27 + 2, "every instance: 0 register spills, 1 CTA per SM; registers are what bounds the design (BM32 with a producer warp would spill)", 13,
       GRAY, maxw=BW - 30)
# tried and dropped
dy = ry + 200
s.text(BX + 20, dy, "Tried on this kernel and dropped", 15, RED, bold=True)
drops = [("producer warp + mbarrier (warp specialisation)", "correct, not faster: the bandwidth case is already at the peak"),
         ("software-pipelined inner loop", "163 -> 157 us with L2-resident data, nothing on real shapes"),
         ("2 CTAs per SM for bf16", "the split count doubles: more partials, bigger combine"),
         ("fused last-arriver combine", "b=32, s_k=1024: 73.7 -> 81.9 us (serial merge on the critical path)"),
         ("h_q=128: skip identity rescale / ping-pong groups", "2-5% slower: bound by the FP32-accumulate tensor pipe")]
for i, (a, b) in enumerate(drops):
    yy = dy + 28 + i * 25
    s.text(BX + 20, yy, "x", 14, RED, bold=True)
    s.text(BX + 38, yy, a, 13.5, INK, bold=True)
    s.text(BX + 44 + text_width(a, 13.5, True), yy, " -  " + b, 13.5, GRAY, maxw=BW - 60 - text_width(a, 13.5, True))
s.lines(BX + 20, dy + 28 + 5 * 25 + 10, ["Where it stands: the h_q=16 kernel reads at 94-99% of the DRAM peak; what is left there is the fixed cost at small batch.",
                                         "The h_q=128 kernel reaches 76% of what FP32-accumulate HMMA can do on a GeForce card (half rate)."], 13.5, INK, lh=1.4, maxw=BW - 30)

# =============================================================================================== panel C: combine + 8-bit
CX, CWD = 1422, 534
s.group(CX, 112, CWD, 1206, "C  Combine, and the 8-bit KV variant", stroke=GREEN, size=17)
c0 = CX + 22
s.text(c0, 152, "combine kernel:  O = sum_s exp(lse_s - lse) * O_s", 15, GREEN, bold=True, maxw=CWD - 30)
for i in range(4):
    yy = 170 + i * 30
    s.rect(c0, yy, 250, 22, TEAL, FILL["teal"], 1.3, r=0, rough=0.4)
    s.text(c0 + 125, yy + 16, f"partial O of split {i}  (512 x fp32)", 12.5, INK, "middle")
    s.arrow([(c0 + 254, yy + 11), (c0 + 330, 226)], GRAY, 1.4, head=6, rough=0.3, passes=1)
    s.text(c0 + 272, yy + 6, f"w{i}", 12, GRAY)
s.ellipse(c0 + 356, 226, 26, 22, GREEN, FILL["green"])
s.text(c0 + 356, 232, "sum", 14, GREEN, "middle", bold=True)
s.arrow([(c0 + 384, 226), (c0 + 412, 226)], width=1.8, head=7, rough=0.3)
s.box(c0 + 414, 204, 80, 44, None, ["O row", "(bf16)"], fill=FILL["orange"], stroke=ORANGE, size=13, lh=1.2, pad=4)
s.lines(c0, 314, [("wide kernel (<= 96 output rows)", INK, True),
                  "one CTA per (row, quarter of the 512 columns); all the partial",
                  "loads are issued before the LSE weights are known",
                  ("warp kernel (more rows)", INK, True),
                  "one warp per row, 8 rows per CTA",
                  ("chosen by row count", INK, True),
                  "<= 16: wide16,  <= 32: wide8,  <= 96: wide4,  else warp",
                  "b=1: 9.2 -> 5.5 us;  b=32, s_k=1024: 11.7 -> 5.2 us"], 13.5, lh=1.42, maxw=CWD - 30)
s.note(c0 + 4, 478, CWD - 60, 92, ["after the read, discard.global.L2 the partials: they",
                                    "are never read again, so they need no write-back.",
                                    "-2.6 .. -3.3 us/layer at b = 1..8, output bit-identical"], title="P1 (planned, phase 3)", size=13.5, rot=0.7)
# 8-bit layout
ey = 614
s.text(c0, ey, "8-bit KV cache (optional): 0.559x the bytes", 15, GREEN, bold=True)
s.rect(c0, ey + 14, 330, 36, BLUE, FILL["blue"], 1.5, r=0, rough=0.5)
s.text(c0 + 165, ey + 37, "512 x 8-bit latent (FP8 e4m3 or INT8)", 13, INK, "middle", maxw=324)
s.rect(c0 + 334, ey + 14, 100, 36, ORANGE, FILL["orange"], 1.5, r=0, rough=0.5)
s.text(c0 + 384, ey + 37, "64 x bf16 RoPE", 13, INK, "middle", maxw=96)
s.rect(c0 + 438, ey + 14, 56, 36, PURPLE, FILL["purple"], 1.5, r=0, rough=0.5)
s.text(c0 + 466, ey + 37, "scale", 13, INK, "middle")
s.lines(c0, ey + 74, ["640 B row + one fp32 scale per token = 644 B  (bf16: 1152 B)",
                      "dequantised inside the kernel with LOP3 / PRMT bit tricks",
                      "(sm_86 has no FP8 conversion); the MMA is still bf16.",
                      "INT8 is 2.4-13x more accurate than FP8 at the same size."], 13.5, lh=1.42, maxw=CWD - 30)
# two KV groups
ky = ey + 172
s.text(c0, ky, "phase 2: two KV groups per CTA (8 warps)", 15, GREEN, bold=True)
for i in range(8):
    col = "blue" if i % 2 == 0 else "orange"
    s.rect(c0 + i * 61, ky + 14, 56, 30, STROKE[col], FILL[col], 1.4, r=4, rough=0.5)
    s.text(c0 + i * 61 + 28, ky + 34, f"tile {i}", 13, INK, "middle")
s.text(c0, ky + 64, "group 0: tiles 0, 2, 4, ...", 13.5, BLUE, bold=True)
s.text(c0 + 250, ky + 64, "group 1: tiles 1, 3, 5, ...", 13.5, ORANGE, bold=True)
s.lines(c0, ky + 88, ["The 8-bit kernel was not memory-bound: 96% of its math-pipe",
                      "stalls sat on HMMA with one warp per scheduler. A second",
                      "group of 4 warps (own 2-stage ring, own barrier) fills those",
                      "gaps; (m, l, O) of the two groups are merged at the end.",
                      ("One choice for the whole launch: units_per_part >= 64.", INK, True),
                      "A per-part choice starves the CTAs with fewer bytes in flight",
                      "(b=32: 151 vs 122 us), so the gate must be launch-wide."], 13.5, lh=1.42, maxw=CWD - 30)
# results
ry2 = ky + 266
s.box(c0 - 4, ry2, CWD - 36, 226, "Measured on the RTX 3090 (GPU idle, CUDA graph)", [
    ("h_q=16, b >= 64, s_k >= 4096", GREEN, True),
    "821-864 GB/s  =  94-99% of the 873.8 GB/s read peak",
    ("h_q=128, b >= 16, s_k >= 4096", GREEN, True),
    "50.8-52.6 TFLOPS  =  73-75% of the 70 TFLOPS bf16 peak",
    ("time of other kernels / v2  (h_q=16 | h_q=128, geomean)", GREEN, True),
    "legacy 1.07 | 1.17;   Triton 1.47 | 2.23;   FlashInfer 3.43 | 1.77",
    ("8-bit KV vs bf16  (h_q=16, b >= 32, s_k >= 4096)", GREEN, True),
    "FP8 1.55-1.70x,  INT8 1.54-1.60x"], fill="#ffffff", stroke=GREEN, tsize=15.5, size=13.5, align="left", lh=1.5, pad=12)

s.save("fig2_decode_kernel.svg")
print("fig2 ok")
