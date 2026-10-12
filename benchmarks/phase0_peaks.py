"""Measure achievable DRAM bandwidth and tensor-core matmul throughput on the selected GPU.

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python benchmarks/phase0_peaks.py [--smi-index 1]
Prints a JSON summary on the last line. nvidia-smi clocks/power are sampled during the run.
"""
import argparse, json, statistics, subprocess, threading, time

import torch


def timed(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1e-3)
    return ts


class SmiSampler:
    Q = "clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"

    def __init__(self, idx):
        self.idx, self.rows, self._stop = idx, [], False

    def _run(self):
        while not self._stop:
            out = subprocess.run(["nvidia-smi", "-i", str(self.idx), f"--query-gpu={self.Q}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
            try:
                v = [x.strip() for x in out.split(",")]
                self.rows.append((float(v[0]), float(v[1]), float(v[2]), float(v[3]), v[4]))
            except Exception:
                pass
            time.sleep(0.3)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self._stop = True; self.t.join()

    def summary(self):
        if not self.rows:
            return {}
        sm = [r[0] for r in self.rows]; pw = [r[2] for r in self.rows]; tp = [r[3] for r in self.rows]
        return dict(n=len(self.rows), sm_clock_mhz_median=statistics.median(sm), sm_clock_mhz_min=min(sm), sm_clock_mhz_max=max(sm),
                    mem_clock_mhz=self.rows[-1][1], power_w_median=statistics.median(pw), power_w_max=max(pw),
                    temp_c_max=max(tp), throttle_reasons_last=self.rows[-1][4])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smi-index", type=int, default=1); ap.add_argument("--gib", type=float, default=1.0)
    a = ap.parse_args()
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    res = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda)
    with SmiSampler(a.smi_index) as smi:
        # --- DRAM: device-to-device copy (reads N bytes, writes N bytes) and a read-only reduction
        n = int(a.gib * (1 << 30))
        x = torch.empty(n, dtype=torch.uint8, device=dev); y = torch.empty_like(x)
        x.random_(0, 255)
        ts = timed(lambda: y.copy_(x), reps=200, warmup=20)
        t = statistics.median(ts)
        res["copy_1GiB"] = dict(median_ms=t * 1e3, GBps_rw=2 * n / t / 1e9, GBps_rw_best=2 * n / min(ts) / 1e9)
        xb = x.view(torch.bfloat16)
        ts = timed(lambda: xb.sum(dtype=torch.float32), reps=200, warmup=20)
        t = statistics.median(ts)
        res["sum_read_1GiB"] = dict(median_ms=t * 1e3, GBps_read=n / t / 1e9, GBps_read_best=n / min(ts) / 1e9)
        del x, y, xb; torch.cuda.empty_cache()
        # --- tensor core: 8192^3 matmul
        for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            A = torch.randn(8192, 8192, device=dev, dtype=dt); B = torch.randn(8192, 8192, device=dev, dtype=dt)
            ts = timed(lambda: torch.matmul(A, B), reps=60, warmup=15)
            t = statistics.median(ts); fl = 2 * 8192 ** 3
            res[f"matmul_{name}_8192"] = dict(median_ms=t * 1e3, TFLOPS=fl / t / 1e12, TFLOPS_best=fl / min(ts) / 1e12)
            del A, B
    res["smi"] = smi.summary()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
