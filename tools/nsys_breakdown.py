#!/usr/bin/env python
"""Per-kernel time and inter-kernel idle gaps from an nsys sqlite export.

    python tools/nsys_breakdown.py x.sqlite [--nvtx RANGE] [--top 25] [--last N]

* kernels are summed per demangled name (count, total, mean, % of busy time);
* "idle" = GPU time between consecutive kernels on the same stream (launch / dependency bubbles);
* --nvtx RANGE keeps only kernels that start inside an NVTX range with that text (e.g. one decode step);
* --last N keeps only the last N kernels (steady state, skips warm-up).
"""
import argparse
import sqlite3
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--nvtx", default=None)
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--last", type=int, default=0)
a = ap.parse_args()
con = sqlite3.connect(a.db)
cur = con.cursor()
tabs = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
kn = "CUPTI_ACTIVITY_KIND_KERNEL"
if kn not in tabs:
    raise SystemExit("no kernel table in " + a.db)
rows = cur.execute(f"select k.start, k.end, k.streamId, coalesce(s.value, '?') from {kn} k "
                   f"left join StringIds s on s.id = k.demangledName order by k.start").fetchall()
if a.nvtx and "NVTX_EVENTS" in tabs:
    rng = cur.execute("select start, end from NVTX_EVENTS where text = ? and end is not null", (a.nvtx,)).fetchall()
    rows = [r for r in rows if any(s <= r[0] <= e for s, e in rng)]
if a.last:
    rows = rows[-a.last:]
if not rows:
    raise SystemExit("no kernels")
by = defaultdict(lambda: [0, 0])
for s, e, st, n in rows:
    by[n][0] += 1
    by[n][1] += e - s
busy = sum(v[1] for v in by.values())
span = max(r[1] for r in rows) - min(r[0] for r in rows)
gaps, last = 0, {}
for s, e, st, n in rows:
    if st in last and s > last[st]:
        gaps += s - last[st]
    last[st] = max(last.get(st, 0), e)
print(f"kernels: {len(rows)}   busy: {busy/1e3:.1f} us   span: {span/1e3:.1f} us   "
      f"inter-kernel idle (same stream): {gaps/1e3:.1f} us ({100*gaps/max(span,1):.1f}% of span)")
print(f"{'count':>6} {'total_us':>10} {'mean_us':>9} {'%busy':>6}  kernel")
for n, (c, t) in sorted(by.items(), key=lambda kv: -kv[1][1])[: a.top]:
    print(f"{c:6d} {t/1e3:10.1f} {t/1e3/c:9.2f} {100*t/busy:6.1f}  {n[:100]}")
