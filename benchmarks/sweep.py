#!/usr/bin/env python
"""Template-parameter sweep driver for the sm86 MLA decode kernel (plan section 3, P3).

Builds one extension per parameter combination (different compile-time instance), benchmarks every build with
bench_decode.py (impl "v2") on the same shapes and picks the best configuration per shape.

INTERFACE EXPECTED FROM THE KERNEL (csrc/sm86, owner SK)
--------------------------------------------------------
* Every tunable is a preprocessor macro  FLASH_MLA_SM86_<NAME>  (e.g. FLASH_MLA_SM86_BM, _NT, _STAGES, _WARPS, _COMBINE),
  read in the headers with `#ifndef FLASH_MLA_SM86_<NAME> / #define ... <default>` so that a plain build still works.
* setup.py turns the env var  FLASH_MLA_DEFINES="FLASH_MLA_SM86_BM=16 FLASH_MLA_SM86_NT=32"  into -D flags for nvcc and
  g++ (already implemented, owner SA).  The sweep sets it per configuration; nothing else needs to change in setup.py.
* Optional: the module may expose  flash_mla_cuda.sm86_config() -> str  (compiled-in configuration); bench_decode.py records
  it in the CSV `extra` column so a result can always be traced back to its instance.
* The default dispatch inside impl="v2" must honour the compiled-in values (a sweep build has ONE instance), i.e. the
  macros select the instance that impl="v2" launches for every M.  If the kernel dispatches over several instances at run
  time, expose them as macros that restrict the dispatch table instead.

SPEC FILE (json)
----------------
    {"grid": {"BM": [16, 32], "NT": [16, 32], "STAGES": [2, 3, 4]},       # cross product -> FLASH_MLA_SM86_<K>=<v>
     "constraint": "BM * NT <= 1024",                                      # optional python expression over the params
     "extra_defines": "FLASH_MLA_SM86_PIPELINE=1",                         # optional, added to every config
     "configs": {"hand_picked": "FLASH_MLA_SM86_BM=16 FLASH_MLA_SM86_NT=64"}}   # optional explicit configs (name -> defines)

USAGE
-----
    python benchmarks/sweep.py --spec benchmarks/sweep_spec.json --worktree /data2/LiuJun/infra/flashmla-wt/kernel \\
           --gpu 1 --preset main --jobs 12 [--tag first] [--check] [--no-build] [--rebuild]

Per configuration the build lands in /data2/LiuJun/tmp/flashmla/sweep/<tag>/<cfg>/ (never in the repo; ptxas registers /
spills are parsed from the build log); results: <tag>/results.csv (all rows), <tag>/summary.md (best per shape + table),
<tag>/best.json (shape -> config).
"""
import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_ROOT = "/data2/LiuJun/tmp/flashmla/sweep"
ENV_ACT = "source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data2/LiuJun/conda/envs/flashmla"


def expand(spec):
    cfgs = {}
    grid = spec.get("grid") or {}
    names = list(grid)
    for vals in itertools.product(*[grid[n] for n in names]) if names else []:
        p = dict(zip(names, vals))
        if spec.get("constraint") and not eval(spec["constraint"], {}, dict(p)):
            continue
        name = "_".join(f"{k}{v}" for k, v in p.items())
        cfgs[name] = " ".join(f"FLASH_MLA_SM86_{k}={v}" for k, v in p.items())
    cfgs.update(spec.get("configs") or {})
    extra = spec.get("extra_defines", "")
    return {n: (d + " " + extra).strip() for n, d in cfgs.items()}


def build(cfg, defines, wt, out_dir, jobs, log, rebuild):
    os.makedirs(out_dir, exist_ok=True)
    have = any(f.startswith("flash_mla_cuda") and f.endswith(".so") for f in os.listdir(out_dir))
    if have and not rebuild:
        return True
    cmd = (f"{ENV_ACT} && cd {wt} && export FLASH_MLA_ARCH=86 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS={jobs} NVCC_THREADS=2 "
           f"FLASH_MLA_LINEINFO=1 FLASH_MLA_DEFINES='{defines}' && "
           f"python setup.py build_ext --build-lib {out_dir} --build-temp {out_dir}/tmp")
    with open(log, "w") as f:
        r = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def ptxas_summary(log):
    """(max registers, total spill bytes) over the sm86 kernels in a build log."""
    regs, spill, cur = 0, 0, ""
    try:
        for line in open(log, errors="replace"):
            m = re.search(r"Compiling entry function '([^']+)'", line)
            if m:
                cur = m.group(1)
            m = re.search(r"Used (\d+) registers", line)
            if m and ("sm86" in cur or "mla" in cur):
                regs = max(regs, int(m.group(1)))
            m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if m and ("sm86" in cur or "mla" in cur):
                spill += int(m.group(1)) + int(m.group(2))
    except OSError:
        pass
    return regs, spill


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--worktree", required=True, help="worktree whose csrc/sm86 and flash_mla/ are swept")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--jobs", type=int, default=12, help="MAX_JOBS for each build (keep <= nproc/2)")
    ap.add_argument("--preset", default="main")
    ap.add_argument("--hq", default=None); ap.add_argument("--sq", default=None); ap.add_argument("--b", default=None)
    ap.add_argument("--sk", default=None); ap.add_argument("--page", default=None)
    ap.add_argument("--impl", default="v2")
    ap.add_argument("--modes", default="eager")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-build", action="store_true", help="reuse existing builds, skip missing ones")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the configurations and exit")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    cfgs = expand(spec)
    print(f"[sweep] {len(cfgs)} configurations", flush=True)
    if a.dry_run:
        for n, d in cfgs.items():
            print(f"  {n}: {d}")
        return 0
    root = os.path.join(SWEEP_ROOT, a.tag)
    os.makedirs(root, exist_ok=True)
    built = {}
    for n, d in cfgs.items():
        out_dir = os.path.join(root, n)
        log = os.path.join(root, f"{n}.build.log")
        if a.no_build:
            ok = os.path.isdir(out_dir)
        else:
            t0 = time.time()
            ok = build(n, d, a.worktree, out_dir, a.jobs, log, a.rebuild)
            print(f"[sweep] build {n}: {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s) -> {log}", flush=True)
        regs, spill = ptxas_summary(log)
        built[n] = dict(ok=ok, dir=out_dir, regs=regs, spill=spill, defines=d)

    rows = []
    shape_args = []
    for flag, val in (("--hq", a.hq), ("--sq", a.sq), ("--b", a.b), ("--sk", a.sk), ("--page", a.page)):
        if val is not None:
            shape_args += [flag, val]
    if not shape_args:
        shape_args = ["--preset", a.preset]
    for n, info in built.items():
        if not info["ok"]:
            continue
        csv_path = os.path.join(root, f"{n}.csv")
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=a.gpu,
                   PYTHONPATH=f"{info['dir']}:{a.worktree}")
        cmd = ["python", os.path.join(HERE, "bench_decode.py"),
               "--impls", a.impl, "--modes", a.modes, "--out", csv_path, "--tag", f"sweep_{n}"] + shape_args \
              + (["--check"] if a.check else [])
        print(f"[sweep] bench {n}", flush=True)
        log = os.path.join(root, f"{n}.bench.log")
        with open(log, "w") as f:
            subprocess.run(["bash", "-c", f"{ENV_ACT} && " + " ".join(map(str, cmd))], env=env, stdout=f, stderr=subprocess.STDOUT)
        if os.path.exists(csv_path):
            for r in csv.DictReader(open(csv_path)):
                r["config"], r["regs"], r["spill_bytes"] = n, info["regs"], info["spill"]
                rows.append(r)

    if not rows:
        print("[sweep] no results")
        return 1
    fields = list(rows[0].keys())
    with open(os.path.join(root, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # best config per shape (by median time, status ok, optionally accuracy-gated by --check cos_out < 8e-5)
    shape_key = lambda r: (r["dtype"], r["kv"], r["h_q"], r["s_q"], r["b"], r["s_k"], r["varlen"], r["page"], r["mode"])
    by_shape = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        if r.get("cos_out") and float(r["cos_out"]) > 8e-5:
            continue
        by_shape.setdefault(shape_key(r), []).append(r)
    best = {}
    lines = ["| dtype | kv | h_q | s_q | b | s_k | page | mode | best config | time_us | GB/s | %BW | TFLOPS | runner-up | its time_us |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, rs in by_shape.items():
        rs.sort(key=lambda r: float(r["time_us"]))
        b0 = rs[0]
        ru = rs[1] if len(rs) > 1 else None
        best["|".join(k)] = dict(config=b0["config"], time_us=float(b0["time_us"]), defines=built[b0["config"]]["defines"])
        lines.append("| " + " | ".join([k[0], k[1], k[2], k[3], k[4], k[5], k[7], k[8], b0["config"], b0["time_us"], b0["gbps"],
                                        b0["pct_bw"], b0["tflops"], ru["config"] if ru else "", ru["time_us"] if ru else ""]) + " |")
    lines += ["", "| config | defines | max regs | spill bytes | built |", "|---|---|---|---|---|"]
    for n, i in built.items():
        lines.append(f"| {n} | `{i['defines']}` | {i['regs']} | {i['spill']} | {i['ok']} |")
    open(os.path.join(root, "summary.md"), "w").write("\n".join(lines) + "\n")
    json.dump(best, open(os.path.join(root, "best.json"), "w"), indent=1)
    print("\n".join(lines))
    print(f"[sweep] done -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
