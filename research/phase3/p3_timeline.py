"""Per-layer kernel timeline from an nsys sqlite of bench_layer.py (fused, graph, skinny GEMM, T <= 32).

A layer starts at the qkv_a skinny GEMM (grid.x = 3648 / 32 = 114) and ends at the o_proj skinny GEMM (grid.x = 2048 / 32 = 64,
grid.z = 1).  Prints per position the median duration, the median idle gap before it and the kernel, then the per-step
bookkeeping kernels (positions / slots / metadata) outside the layers.

usage: python p3_timeline.py <sqlite> [skip_layers]
"""
import re
import sqlite3
import statistics as stt
import sys


def short(name):
    return re.sub(r"<.*", "", name)[-44:]


db = sqlite3.connect(sys.argv[1])
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 16
rows = db.execute(
    "select k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k "
    "join StringIds s on k.shortName = s.id order by k.start").fetchall()


def is_qkv(r):
    return "skinny_gemm" in r[2] and r[3] == 114 and r[5] == 1


def is_oproj(r):
    return "skinny_gemm" in r[2] and r[3] == 64 and r[5] == 1


layers, outside = [], []
cur = None
prev_end = None
for r in rows:
    gap = (r[0] - prev_end) if prev_end is not None else 0
    prev_end = r[1]
    if is_qkv(r):
        cur = []
        layers.append(cur)
    if cur is not None:
        cur.append((r, gap))
        if is_oproj(r):
            cur = None
    else:
        outside.append((r, gap))
layers = [l for l in layers[skip:] if l and is_oproj(l[-1][0])]
n = stt.mode(len(l) for l in layers)
layers = [l for l in layers if len(l) == n]
print(f"{sys.argv[1].split('/')[-1]}: layers used {len(layers)}, kernels per layer {n}")
print(f"{'pos':>3} {'dur_us':>7} {'gap_us':>7} {'grid':>12} {'blk':>4}  kernel")
td = tg = 0.0
for k in range(n):
    d = stt.median((l[k][0][1] - l[k][0][0]) / 1e3 for l in layers)
    g = stt.median(l[k][1] / 1e3 for l in layers)
    r = layers[0][k][0]
    td += d
    tg += g
    print(f"{k:>3} {d:7.2f} {g:7.2f} {r[3]:>5}x{r[4]}x{r[5]:<3} {r[6]:>4}  {short(r[2])}")
print(f"sum {td:7.2f} {tg:7.2f}   busy + gaps = {td + tg:.2f} us per layer")
# per-step bookkeeping (outside the layers), last 40 entries -> one step
names = {}
for r, g in outside[len(outside) // 2:]:
    names.setdefault(short(r[2]), []).append((r[1] - r[0]) / 1e3)
print("outside layers (per occurrence, median us):",
      "; ".join(f"{k} {stt.median(v):.2f}" for k, v in names.items()))
