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
