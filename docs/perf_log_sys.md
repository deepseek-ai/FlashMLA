# Performance log, system side (RTX 3090 / sm_86, FlashMLA Ampere optimization)

Kernel-side iteration notes live in `docs/perf_log_kernel.md` (owned by the kernel work); this file holds the environment,
measured peaks, baselines, benchmark-harness results, fusion / CUDA Graph and quantization results.

Convention: one section per topic; each row = shape -> time / GB/s / TFLOPS / key ncu metrics.
Raw logs: `/data2/LiuJun/tmp/flashmla/{logs,ncu,nsys}`.

## 0. Environment (Phase 0)

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (sm_86, 82 SM, 24 GB), `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>` (SA: GPU1 bench / GPU2 fusion, SK: GPU0) |
| driver | 570.195.03 |
| conda env | `/data2/LiuJun/conda/envs/flashmla` (python 3.11.17). Activation: `source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data2/LiuJun/conda/envs/flashmla` |
| torch | **2.7.1+cu126** (`torch.version.cuda` = 12.6), triton 3.3.1, numpy 2.4.6, pandas 3.0.6, ninja 1.13.2, pytest 9.1.1, nvtx 0.2.16; `pip check` clean |
| CUDA toolkit for nvcc | `CUDA_HOME=/data2/LiuJun/cuda/cuda-12.4` (nvcc 12.4.131), set by the env's `activate.d/cuda124.sh` (PATH + LD_LIBRARY_PATH; undone by `deactivate.d/cuda124.sh`). `/usr/local/cuda-11.8` has no nvcc and `/usr/local/cuda` is a dangling link, so the plan's nvcc 11.8 was replaced by 12.4 (no new toolkit installed) |
| host compiler | gcc/g++ 9.4.0 |
| CUTLASS | submodule `csrc/cutlass` @ `afa1772203677c5118fcd82537a9c8fefbcc7008` (3.8.0 + a few commits). **Compiles cleanly with nvcc 12.4 for sm_86** (all legacy sm80 kernels, 0 spills) |
| build | `tools/build.sh [dir]` = `FLASH_MLA_ARCH=86 python setup.py build_ext --inplace` (MAX_JOBS=12, NVCC_THREADS=2); use with `PYTHONPATH=<dir>`. `setup.py` globs `csrc/sm86/*.{cu,cpp}` into `flash_mla_cuda` and `csrc/fused/*.{cu,cpp}` into a separate extension `flash_mla_fused_cuda` |
| profilers | ncu `/opt/nvidia/nsight-compute/2024.1.1/ncu`, nsys `/usr/local/bin/nsys` (clocks cannot be locked, no sudo) |
| network | non-interactive ssh does not load the proxy settings; use `bash -ic` (or `source ~/.bashrc` is a no-op because of the interactive guard) for anything that needs the network |

## 1. Measured peaks on GPU 1 (`benchmarks/phase0_peaks.py`, median of many reps)

Measured with torch 2.7.1+cu118 during the first Phase 0 attempt; hardware-bound, so valid for the cu126 stack.

| metric | median | best | notes |
|---|---|---|---|
| DRAM D2D copy 1 GiB (R+W bytes) | **841.6 GB/s** | 842.9 | 90% of 936 GB/s spec |
| DRAM read-only (bf16 sum of 1 GiB) | **873.8 GB/s** | 875.3 | 93% of spec; the relevant ceiling for decode (KV is read-only) |
| bf16 matmul 8192^3 (fp32 accumulate) | **70.0 TFLOPS** | 70.7 | ~ spec 71 TFLOPS |
| fp16 matmul 8192^3 | **66.4 TFLOPS** | 67.6 | |

nvidia-smi during the run: SM clock median 1680 MHz (min 1515, max 1905), memory 9501 MHz, power median 319 W (max 325 W of
350 W limit), temp <= 55 C, throttle reason 0x4 (SW power cap) at the end of the run. Clocks cannot be locked.
Raw output: `/data2/LiuJun/tmp/flashmla/logs/peaks_run1.log`.

Targets derived from these peaks: memory-bound h_q=16 -> >= 80% of 874 GB/s = ~700 GB/s (KV read bytes);
compute-bound h_q=128 -> >= 60% of 70 TFLOPS = ~42 TFLOPS.

## 2. Legacy kernel baseline (fork's sm80 kernels on the 3090)

### 2.1 The unmodified kernel does not launch

`CUDA_VISIBLE_DEVICES=1 python benchmarks/legacy_baseline.py` with the unmodified kernels:

```
CUDA error (.../csrc/flash_fwd_mla_kernel_sm80.h:689): invalid argument        (cooperative kernel)
CUDA error (.../csrc/flash_fwd_mla_bf16_ws_sm80.cu:14): invalid argument       (warp-specialized kernel)
```

Both come from `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage))` followed by
`exit(1)`. `SharedStorage` = Q 32x576 bf16 (36 864 B) + double-buffered K 2x32x576 bf16 (73 728 B) + P (2 048 B) + reduce buffer
= **113 280 B** (ws kernel: 113 664 B), above the sm_86 opt-in limit of **101 376 B** (99 KB).

### 2.2 Minimal fix (commit on `opt/sm86`)

* `csrc/flash_fwd_mla_kernel_sm80.h`: `kNWarpsM = (kBlockM >= 32) ? 2 : 1`; `mha_fwd_splitkv_mla<T,576,false>::run` queries
  `cudaDevAttrMaxSharedMemoryPerBlockOptin` and uses the original `<576,32,32,8>` tile when it fits (A100 behaviour unchanged),
  otherwise `<576,16,32,4>` (BlockM=16, 4 warps, **93 504 B** smem, 0 spills). Throws `std::runtime_error` if even that does not fit.
* `csrc/flash_fwd_mla_bf16_ws_sm80.cu`: the warp-specialized kernel hard-wires an 8-warp 2Mx4N PV tiling to BlockM=32, so there is no
  minimal fix; it now throws a clear `RuntimeError` ("needs 113664 B of shared memory, device allows 101376 B") instead of `exit(1)`.
* `flash_mla/flash_mla_interface.py`: `impl="legacy"` (default for now; `"legacy_ws"` = warp_spec) and `warp_spec` default False.
  `impl="v2"` raises NotImplementedError until the new kernel lands.
* Not changed: `get_mla_metadata` still uses block_size_m=32 (num_sm_parts = 82 / ceil(M/32)); with BlockM=16 this gives
  twice as many m-blocks per part, same work per SM (verified by the table below).

Correctness: the fork's `tests/test_flash_mla_sm80.py` (b=128, s_k 4096/8192, h_q 16/32/64/128, s_q 1/2, causal, varlen, NaN-padded KV,
cos_diff < 8e-5 for out and lse) passes all 32 cases (`logs/legacy_correctness.log`, exit 0). cos_diff in the table below is ~2e-6.

### 2.3 Legacy-coop (BlockM=16) timings, GPU 1, s_q=1, page 32, bf16 (`benchmarks/legacy_baseline.py`)

GB/s counts KV + Q + O bytes, TFLOPS = 2*s_q*h_q*sum(s_k)*(576+512); `triton.testing.do_bench` (L2 flushed), 25 ms warmup / 100 ms rep.
GPU at 1905 MHz / 227 W / 67 C when the run started.

| h_q | b | s_k | time (us) | GB/s | % of 874 GB/s | TFLOPS | % of 70 TFLOPS | cos_diff |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 1024 | 16.4 | 74 | 8% | 2.18 | 3% | 1.8e-06 |
| 16 | 1 | 4096 | 25.8 | 184 | 21% | 5.52 | 8% | 1.6e-06 |
| 16 | 1 | 8192 | 29.8 | 318 | 36% | 9.57 | 14% | 1.5e-06 |
| 16 | 64 | 1024 | 113.3 | 686 | 78% | 20.14 | 29% | 2.1e-06 |
| 16 | 64 | 4096 | 377.0 | 807 | 92% | 24.21 | 35% | |
| 16 | 64 | 8192 | 724.3 | 837 | 96% | 25.20 | 36% | |
| 16 | 128 | 1024 | 209.6 | 742 | 85% | 21.78 | 31% | |
| 16 | 128 | 4096 | 737.9 | 825 | 94% | 24.74 | 35% | |
| 16 | 128 | 8192 | 1422.8 | 852 | 97% | 25.66 | 37% | |
| 128 | 1 | 1024 | 26.4 | 55 | 6% | 10.82 | 15% | 1.8e-06 |
| 128 | 1 | 4096 | 46.5 | 107 | 12% | 24.54 | 35% | 1.6e-06 |
| 128 | 1 | 8192 | 72.2 | 135 | 15% | 31.60 | 45% | 1.5e-06 |
| 128 | 64 | 1024 | 453.2 | 206 | 24% | 40.28 | 58% | 2.1e-06 |
| 128 | 64 | 4096 | 1697.1 | 188 | 22% | 43.02 | 61% | |
| 128 | 64 | 8192 | 3327.1 | 187 | 21% | 43.89 | 63% | |
| 128 | 128 | 1024 | 892.3 | 209 | 24% | 40.91 | 58% | |
| 128 | 128 | 4096 | 3499.6 | 183 | 21% | 41.73 | 60% | |
| 128 | 128 | 8192 | 6987.6 | 178 | 20% | 41.80 | 60% | |

**Finding:** with the BlockM=16 fix the legacy kernel is *already* near the DRAM roofline for the large-batch memory-bound case
(h_q=16, b>=64, s_k>=4096: 807-852 GB/s = 92-97% of the measured read-only peak) and above the plan's compute target for h_q=128
(41-44 TFLOPS = 60-63% of the measured bf16 peak). The headroom for the new kernel is therefore: small batch / latency
(b=1: 318 GB/s at s_k=8192, 29.8 us vs the ~19 us target of 1.3*bytes/BW+5 us), the b=64, s_k=1024 corner (78%), higher h_q
efficiency, the split-KV combine launch, FP16, 64-token pages, and bandwidth-bound wins from the quantized KV cache.

## 3. Benchmark harness and baselines (GPU 1, torch 2.7.1+cu126; FlashInfer in the vllm env)

Tools (all under `benchmarks/`, CSVs in `/data2/LiuJun/tmp/flashmla/logs`, schema in `common.FIELDS`):

* `bench_decode.py` benchmarks whichever `flash_mla` is on `PYTHONPATH` (`pkg` = package default, `legacy`, `legacy_ws`, `v2`) plus the vendored
  vLLM/SGLang Triton kernel (`triton`, best of num_kv_splits 2..128 per shape). `--modes eager,graph`; eager = CUDA-event time with a 256 MB
  L2-flush before every rep (median of 40-500 reps, includes the Python launch path), graph = N=27 layers (as many as fit in 12 GB) captured
  in one CUDA graph, **each layer with its own KV cache**, reported per layer. `--check` compares out and lse with an fp32 reference on NaN-padded
  caches (cos_diff < 8e-5 is the gate; every row below is ~2e-6). Block tables are a random permutation of the pages.
* `bench_flashinfer.py` (same CSV, vllm env, `fa2` MLA wrapper, `tmp/flashmla/run_flashinfer.sh`): JIT built fine on sm_86 and is correct
  (cos_diff 3e-6). `bench_one.py` is the ncu/nsys target. `sweep.py` + `sweep_spec_example.json` is the template-parameter sweep driver
  (interface for SK: macros `FLASH_MLA_SM86_<NAME>`, `FLASH_MLA_DEFINES` env var, see `docs/interface_requests.md`); validated with a 2-config
  dummy sweep (builds into `tmp/flashmla/sweep/<tag>/<cfg>/`, 2 min per build with MAX_JOBS=8).
* `tools/prof_ncu.sh`, `tools/prof_nsys.sh` (+ `ncu_summary.py`, `nsys_breakdown.py`, `ncu_key_metrics.txt`): plan section 4 commands; the metric names were
  validated with `ncu --query-metrics --chip ga102`.
* Commands used: `python benchmarks/bench_decode.py --preset main --page 32 --impls legacy,triton --modes eager,graph --check`,
  `run_flashinfer.sh --preset main --page 32 --modes eager,graph --check`; further presets `varlen`, `longk`, `mtp`, `matrix` (page 16/32/64 x
  h_q 8..128 x s_q 1..2) are in `logs/base_*.csv` and `logs/fi_*.csv` (legacy only runs page 32, so it is absent from the page 16/64 rows).

### 3.1 Main shapes, s_q=1, page 32, bf16 (legacy = BlockM=16 fix; triton = vLLM 0.18.1 kernel; flashinfer 0.6.6 fa2)

The legacy/triton/flashinfer ratio columns are time(other)/time(legacy). Legacy GB/s and TFLOPS use the conventions of section 2.3.

**eager** (graph = per layer of a 27-layer graph, each layer with its own KV cache)

| h_q | b | s_k | legacy us | triton us | flashinfer us | legacy GB/s | legacy TFLOPS | triton/legacy | flashinfer/legacy |
|---|---|---|---|---|---|---|---|---|---|
| 16.0 | 1.0 | 1024.0 | 16.4 | 18.4 | 32.8 | 74.0 | 2.2 | 1.13 | 2.0 |
| 16.0 | 1.0 | 4096.0 | 23.6 | 30.7 | 66.6 | 202.0 | 6.0 | 1.3 | 2.83 |
| 16.0 | 1.0 | 8192.0 | 29.7 | 41.0 | 88.1 | 319.0 | 9.6 | 1.38 | 2.96 |
| 16.0 | 64.0 | 1024.0 | 112.6 | 143.4 | 305.2 | 690.0 | 20.3 | 1.27 | 2.71 |
| 16.0 | 64.0 | 4096.0 | 374.8 | 510.0 | 1409.0 | 812.0 | 24.4 | 1.36 | 3.76 |
| 16.0 | 64.0 | 8192.0 | 717.8 | 968.7 | 2965.5 | 845.0 | 25.4 | 1.35 | 4.13 |
| 16.0 | 128.0 | 1024.0 | 205.8 | 277.5 | 630.8 | 755.0 | 22.2 | 1.35 | 3.06 |
| 16.0 | 128.0 | 4096.0 | 729.1 | 963.6 | 2438.1 | 835.0 | 25.0 | 1.32 | 3.34 |
| 16.0 | 128.0 | 8192.0 | 1415.2 | 1860.6 | 5280.8 | 857.0 | 25.8 | 1.31 | 3.73 |
| 128.0 | 1.0 | 1024.0 | 25.6 | 27.6 | 56.3 | 57.0 | 11.1 | 1.08 | 2.2 |
| 128.0 | 1.0 | 4096.0 | 46.1 | 72.7 | 95.2 | 108.0 | 24.8 | 1.58 | 2.07 |
| 128.0 | 1.0 | 8192.0 | 70.7 | 130.0 | 103.4 | 138.0 | 32.3 | 1.84 | 1.46 |
| 128.0 | 64.0 | 1024.0 | 444.4 | 960.5 | 696.3 | 210.0 | 41.1 | 2.16 | 1.57 |
| 128.0 | 64.0 | 4096.0 | 1681.4 | 3613.7 | 2718.7 | 190.0 | 43.4 | 2.15 | 1.62 |
| 128.0 | 64.0 | 8192.0 | 3403.8 | 7274.5 | 5610.5 | 183.0 | 42.9 | 2.14 | 1.65 |
| 128.0 | 128.0 | 1024.0 | 909.3 | 2064.3 | 1260.5 | 205.0 | 40.2 | 2.27 | 1.39 |
| 128.0 | 128.0 | 4096.0 | 3613.7 | 8125.4 | 5309.4 | 177.0 | 40.4 | 2.25 | 1.47 |
| 128.0 | 128.0 | 8192.0 | 8279.0 | 16579.6 | 11529.7 | 150.0 | 35.3 | 2.0 | 1.39 |

**graph** (graph = per layer of a 27-layer graph, each layer with its own KV cache)

| h_q | b | s_k | legacy us | triton us | flashinfer us | legacy GB/s | legacy TFLOPS | triton/legacy | flashinfer/legacy |
|---|---|---|---|---|---|---|---|---|---|
| 16.0 | 1.0 | 1024.0 | 10.2 | 15.1 | 25.6 | 119.0 | 3.5 | 1.48 | 2.51 |
| 16.0 | 1.0 | 4096.0 | 18.8 | 26.4 | 63.5 | 253.0 | 7.6 | 1.41 | 3.38 |
| 16.0 | 1.0 | 8192.0 | 24.8 | 36.3 | 84.7 | 381.0 | 11.5 | 1.46 | 3.41 |
| 16.0 | 64.0 | 1024.0 | 109.4 | 141.7 | 308.5 | 710.0 | 20.8 | 1.3 | 2.82 |
| 16.0 | 64.0 | 4096.0 | 372.2 | 514.3 | 1509.4 | 817.0 | 24.5 | 1.38 | 4.06 |
| 16.0 | 64.0 | 8192.0 | 715.0 | 978.1 | 3274.4 | 848.0 | 25.5 | 1.37 | 4.58 |
| 16.0 | 128.0 | 1024.0 | 203.0 | 278.3 | 675.4 | 766.0 | 22.5 | 1.37 | 3.33 |
| 16.0 | 128.0 | 4096.0 | 723.3 | 977.2 | 2835.8 | 841.0 | 25.2 | 1.35 | 3.92 |
| 16.0 | 128.0 | 8192.0 | 1407.9 | 1872.1 | 5578.9 | 861.0 | 25.9 | 1.33 | 3.96 |
| 128.0 | 1.0 | 1024.0 | 20.3 | 23.9 | 50.6 | 72.0 | 14.1 | 1.18 | 2.49 |
| 128.0 | 1.0 | 4096.0 | 43.2 | 68.2 | 93.5 | 116.0 | 26.4 | 1.58 | 2.16 |
| 128.0 | 1.0 | 8192.0 | 71.1 | 125.8 | 93.8 | 137.0 | 32.1 | 1.77 | 1.32 |
| 128.0 | 64.0 | 1024.0 | 460.0 | 972.7 | 718.5 | 203.0 | 39.7 | 2.11 | 1.56 |
| 128.0 | 64.0 | 4096.0 | 1739.4 | 3662.6 | 2951.5 | 184.0 | 42.0 | 2.11 | 1.7 |
| 128.0 | 64.0 | 8192.0 | 3448.6 | 7378.5 | 5887.0 | 180.0 | 42.3 | 2.14 | 1.71 |
| 128.0 | 128.0 | 1024.0 | 947.5 | 2117.4 | 1491.9 | 197.0 | 38.5 | 2.23 | 1.57 |
| 128.0 | 128.0 | 4096.0 | 4038.2 | 8333.9 | 5901.8 | 158.0 | 36.2 | 2.06 | 1.46 |
| 128.0 | 128.0 | 8192.0 | 8657.5 | 17039.0 | 11758.4 | 144.0 | 33.7 | 1.97 | 1.36 |

Observations:
* The BlockM=16 legacy kernel is the best of the three baselines on every main shape. Triton is 1.1-1.4x slower at h_q=16 (it peaks at 650 GB/s = 75% of the
  read-only peak) and 1.1-2.3x slower at h_q=128 (2.0-2.3x for b >= 64, where it reaches only 20 TFLOPS = 29%). FlashInfer fa2 is 2.0-4.6x slower at h_q=16 and 1.3-2.5x at h_q=128, as predicted from
  its 100 KB-smem fallback (one stage, 16-token tile).
* CUDA graph helps only latency-bound shapes: b=1, s_k=1024 goes 16.4 -> 10.2 us (legacy), 18.4 -> 15.1 (triton); at b>=64 eager == graph within 1-2%.
  Small-batch legacy is far from the roofline (b=1, s_k=8192: 29.7 us, 319 GB/s = 36% of peak; target for the new kernel: <= 1.3 * 10.6 us + 5 us = 19 us).
* Long context, b=1 (page 64, `logs/base_longk.csv`): triton 57 / 86 / 146 us at s_k = 16k / 32k / 64k for h_q=16 and 243 / 473 / 890 us for h_q=128;
  FlashInfer 104 / 166 / 308 us (h_q=16) and 162 / 308 / 537 us (h_q=128), i.e. FlashInfer wins h_q=128 long-context b=1 against Triton.

### 3.2 SK's v2a kernel through the same harness (probe, not a result of this branch)

`PYTHONPATH=wt/kernel` at its 20:49 build, page 64, eager: h_q=16 b=4 s_k=4096 41.98 us, b=64 378.9 us (803 GB/s, 92%); h_q=128 b=4 103.4 us
(44 TFLOPS = 63%), b=64 1550 us (47 TFLOPS = 67%); cos_diff 1.7e-6. `pkg` (default impl) == `v2`. Raw: `logs/kernel_probe.csv`. This confirms that the harness
drives the workspace / preallocated out / lse path of the new interface.

## 4. Profiling method check (legacy kernel, h_q=16, b=64, s_k=4096, page 32; `tools/prof_ncu.sh`, `NCU_SET=key`, base clocks)

| metric | main kernel (`flash_fwd_splitkv_mla_kernel<576,16,32,4>`) | combine kernel |
|---|---|---|
| duration (base clock) | 373.7 us | 11.5 us |
| grid x block | 82 x 128 threads (1 CTA/SM, 4 warps) | 1024 x 128 |
| registers / dynamic smem | 168 / 93.5 KB | 40 / 0 |
| DRAM throughput (% of peak) | **90.8%** | 53.7% |
| tensor pipe active | 22.2% | 0 |
| L2 / L1 sector hit rate | 1.7% / 0.4% | 20.8% / 4.7% |
| stall long scoreboard / barrier / short scoreboard / mio throttle | 23.4% / 11.1% / 2.4% / 1.2% | 67.3% / 9.9% / 2.2% / 0.3% |
| shared-store bank conflicts | 34 663 (ld: 0) | 0 |
| local ld/st (spills) | 0 / 0 | 0 / 0 |

nsys (`tools/prof_nsys.sh`, 4-layer graph, b=64): per layer 366 us main + 7.1 us combine kernel; the combine launch is ~2% of a layer at b=64, but at b=1
(10 us per call) it is a visible fraction, which is what the fused combine of the new kernel removes.

## 5. Decode prologue fusion and whole-layer CUDA Graph (GPU 2 for development, GPU 1 for the timed runs)

### 5.1 Fused prologue (`csrc/fused/mla_decode_prologue.cu`, extension `flash_mla_fused_cuda`, front end `flash_mla/mla_prologue.py`)

One launch (grid (H+1, T), 96 threads) does RoPE(q_pe) + q = [q_nope_abs | q_pe_rot] (q_nope read through its bmm-output strides) and, in the extra block per token,
RMSNorm(kv_c) (optional, HF semantics: cast to dtype before the weight product) + RoPE(k_pe) + the paged-cache write at `slot = block_table[b, pos/page]*page + pos%page`,
`pos = cache_seqlens[b] - s_q + j`, either bf16/fp16 or FP8/INT8 + per-token scale. In place, no allocation, no host sync (CUDA-graph safe). Tests: `tests/test_fused_prologue.py`
(page 16/32/64, h 8/16/128, s_q 1/2, neox and interleaved rope, bf16/fp16, strided inputs, untouched-rows check, graph replay with changing seqlens): all pass; rope outputs
within 1 ulp of the fp32-then-round reference, un-normalized latents bit-exact.

Micro-benchmark (`benchmarks/bench_prologue.py`, H=16, ctx 4096, per call; the unfused "torch ref" is norm + 2 ropes + cat + index-put = ~20 launches):

| T (tokens) | torch ref eager | torch ref in graph | fused bf16 eager | fused bf16 in graph | fused fp8 / int8 in graph |
|---|---|---|---|---|---|
| 1 | 552 us | 58.6 us | 5.1 us | 1.8 us | 3.0 / 2.3 us |
| 8 | 553 us | 65.8 us | 5.1 us | 2.0 us | 3.2 / 2.5 us |
| 32 | 557 us | 67.1 us | 6.1 us | 2.4 us | 3.5 / 2.7 us |
| 64 | 562 us | 69.0 us | 8.2 us | 2.8 us | 4.0 / 3.2 us |
| 256 | 561 us | 89.8 us | 16.4 us | 13.9 us | 14.7 / 13.9 us |

### 5.2 DeepSeek-V2-Lite decode layer (`flash_mla/mla_decode_layer.py`, `benchmarks/bench_layer.py`; attn = legacy, page 32, s_k = 4096, bf16)

Layer = q_proj+kv_a_proj (one GEMM when fused) -> W_UK absorption bmm -> prologue -> attention -> W_UV bmm -> o_proj. Per-layer time; graph = 27 layers chained in one
graph incl. the device-side metadata / slot computation, each layer with its own weights and KV cache. Fused and unfused outputs agree (cos_diff <= 2.6e-6).

| b | unfused eager | fused eager | unfused graph | fused graph | kernels/layer unfused -> fused | fused vs unfused (eager / graph) | fused graph vs unfused eager |
|---|---|---|---|---|---|---|---|
| 1 | 804.6 us | 76.5 us | 116.3 us | 68.5 us | 40 -> 7 | 10.5x / 1.70x | 11.7x |
| 8 | 876.7 us | 125.2 us | 179.3 us | 113.5 us | 42 -> 8 | 7.0x / 1.58x | 7.7x |
| 32 | 882.8 us | 265.9 us | 324.6 us | 256.4 us | 42 -> 8 | 3.3x / 1.27x | 3.4x |
| 64 | 894.1 us | 453.5 us | 511.9 us | 441.3 us | 43 -> 9 | 2.0x / 1.16x | 2.0x |

nsys per-op breakdown, b=8, 8-layer graph (`logs/nsys_breakdown_layer_{fused,unfused}_b8.txt`, kernels inside the `decode_step` NVTX range, per layer = total/64):
unfused 43.8 kernels and 186 us busy per layer (attention 50.4 us, 5 GEMM/bmm kernels 63 us, ~37 elementwise/cat/index kernels 73 us); fused 9.8 kernels and 118 us busy
(attention 50.3 us = 43%, 4 GEMM/bmm kernels 56 us, prologue 2.3 us, combine 3.5 us, residual elementwise 2 us, metadata 10.8 us once per step). The fused layer leaves
the attention kernel and the GEMMs as the only significant cost; at b>=32 the graph speed-up shrinks because the attention kernel dominates.

## 6. Quantized KV cache: write path and end-to-end accuracy

Format: `docs/kv_quant_format.md` (unchanged except one clarification: both divisions are IEEE `__fdiv_rn`). 644 B/token vs 1152 B (0.559x).
Reference: `flash_mla/kv_quant.py` (`quantize_kv`, `dequantize_kv`); CUDA write path inside the fused prologue. **Bit-exact** against the reference on 20 000 random heavy-tailed rows
and on a table of every representable code, every rounding tie and +-1 bf16 ulp around ties (fp8 and int8) (`tests/test_kv_quant.py`). Element-wise error (4096 rows x 512 latent dims, gauss / log-normal channel
scales / laplace): fp8 relative RMS < 4% (half an e4m3 step), max |err|/amax <= 2^-4; int8 max |err|/amax <= 0.5/127; rope dims exact.

End-to-end single layer (`benchmarks/eval_kv_quant.py`): one decode token through the full layer (new token quantized by the CUDA write path) with fp32 attention over the dequantized
cache vs the bf16 cache; contexts are built from synthetic hidden states through the layer's own kv_a_proj/RMSNorm/RoPE; regimes `gauss`, `outlier` (4 massive-activation channels x30, token-dependent
magnitude), `heavy` (log-normal per-channel latent scales); q sharpness x1 / x4; 3 seeds; ctx 1k / 4k / 16k (36 rows per kind, `logs/eval_kv_quant.md`).
cos_diff = 1 - 2<x,y>/(|x|^2+|y|^2). No real DeepSeek weights exist on the server, so the statistics are synthetic.

| kind | cos_diff(o) range over all rows | worst row | rel L2(o) range | cos_diff(layer output) worst |
|---|---|---|---|---|
| fp8 e4m3 | 5.2e-05 ... 1.7e-03 (mean over seeds), max 2.4e-03 | gauss, ctx 16k, q x4 | 1.0% ... 5.8% | 2.4e-03 |
| int8 | 6.5e-06 ... 2.2e-04 (mean over seeds), max 2.5e-04 | heavy, ctx 1k, q x4 | 0.4% ... 2.1% | 2.6e-04 |

int8 is 2.4-13x more accurate than e4m3 with a per-token scale (7 effective bits on the bulk of the distribution vs 3 mantissa bits) and costs the same bytes; fp8 degrades with sharper attention
(fewer tokens average out the rounding noise). Accuracy gates used by `tests/test_kv_quant.py::test_end_to_end_accuracy_gate` (about 2x the worst measured): **cos_diff < 5e-3 for fp8, < 5e-4 for int8**,
for both the latent attention output and the layer output; the in-kernel dequant has to meet them against the bf16 reference. These are far above the 8e-5 gate of the lossless path, so the
quantized cache stays an opt-in switch.
