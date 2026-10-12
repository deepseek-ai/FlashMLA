#!/usr/bin/env python
"""Print the key metrics of an `ncu --page raw --csv` export (one block per profiled kernel launch)."""
import csv
import sys

KEYS = ["gpu__time_duration.sum", "launch__grid_size", "launch__block_size", "launch__registers_per_thread",
        "launch__shared_mem_per_block_dynamic", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes_read.sum", "dram__bytes_write.sum",
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.pct_of_peak_sustained_active", "lts__t_sector_hit_rate.pct", "l1tex__t_sector_hit_rate.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct", "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct", "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct", "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "smsp__warp_issue_stalled_not_selected_per_warp_active.pct", "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "smsp__sass_inst_executed_op_ldsm.sum",
        "smsp__inst_executed_op_local_ld.sum", "smsp__inst_executed_op_local_st.sum"]

rows = list(csv.reader(open(sys.argv[1])))
hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
if hi is None:
    raise SystemExit("no kernel rows in " + sys.argv[1])
hdr = rows[hi]
for r in rows[hi + 1:]:
    if len(r) != len(hdr) or not r[0].strip().isdigit():
        continue  # units row etc.
    d = dict(zip(hdr, r))
    print(f"\n--- {d.get('Kernel Name', '?')[:110]}  (id {d.get('ID')}, block {d.get('Block Size', '')}, grid {d.get('Grid Size', '')})")
    for k in KEYS:
        if k in d and d[k] != "":
            print(f"  {k:78s} {d[k]}")
