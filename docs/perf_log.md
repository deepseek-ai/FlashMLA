# Performance log (RTX 3090 / sm_86, FlashMLA Ampere optimization)

Convention: one section per kernel version; each row = shape -> time / GB/s / TFLOPS / key ncu metrics.
Raw logs: `/data2/LiuJun/tmp/flashmla/{logs,ncu,nsys}`.

## 0. Environment (Phase 0)

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (sm_86, 82 SM, 24 GB), physical index 1 used (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`) |
| driver | 570.195.03 |
| conda env | `/data2/LiuJun/conda/envs/flashmla` (python 3.11.17) |
| torch | 2.7.1+cu118 (`torch.version.cuda` = 11.8), triton 3.3.1, numpy 2.4.6, pandas 3.0.6, ninja 1.13.2, pytest 9.1.1, nvtx 0.2.16 |
| host compiler | gcc/g++ 9.4.0 |
| CUDA toolkit for nvcc | **BLOCKED**: `/usr/local/cuda-11.8` contains only cuBLAS/cuDNN libs and headers (no `bin/nvcc`, no `cuda_runtime.h`). Existing complete toolkits: `/usr/local/cuda-12.1` (nvcc 12.1.105), `/data2/LiuJun/cuda/cuda-12.4` (nvcc 12.4.131). See the Phase 0 report. |
| CUTLASS | submodule `csrc/cutlass` @ `afa1772203677c5118fcd82537a9c8fefbcc7008` (3.8.0 + a few commits, 2025-02-20); not yet compiled with any nvcc |
| profilers | ncu `/opt/nvidia/nsight-compute/2024.1.1/ncu`, nsys `/usr/local/bin/nsys` (not clock-lockable, no sudo) |

## 1. Measured peaks on GPU 1 (`benchmarks/phase0_peaks.py`, torch 2.7.1+cu118, median of many reps)

| metric | median | best | notes |
|---|---|---|---|
| DRAM D2D copy 1 GiB (R+W bytes) | **841.6 GB/s** | 842.9 | 90% of 936 GB/s spec |
| DRAM read-only (bf16 sum of 1 GiB) | **873.8 GB/s** | 875.3 | 93% of spec; the relevant ceiling for decode (KV is read-only) |
| bf16 matmul 8192^3 (fp32 accumulate) | **70.0 TFLOPS** | 70.7 | ~ spec 71 TFLOPS |
| fp16 matmul 8192^3 | **66.4 TFLOPS** | 67.6 | |

nvidia-smi during the run: SM clock median 1680 MHz (min 1515, max 1905), memory 9501 MHz, power median 319 W (max 325 W of
350 W limit), temp <= 55 C, throttle reason 0x4 (SW power cap) at the end of the run. Clocks cannot be locked (no sudo).
Raw output: `/data2/LiuJun/tmp/flashmla/logs/peaks_run1.log`.

Targets derived from these peaks: memory-bound h_q=16 -> >= 80% of 874 GB/s = ~700 GB/s (KV read bytes);
compute-bound h_q=128 -> >= 60% of 70 TFLOPS = ~42 TFLOPS.

## 2. Legacy kernel baseline (fork's sm80 kernels, unmodified except where noted)

NOT YET MEASURED: needs an nvcc to build the extension (see environment table). The expected failure of the unmodified
kernel on the 3090 (shared memory 111 KB > 99 KB, `cudaFuncSetAttribute` failure) is therefore also unconfirmed.
Scripts ready: `benchmarks/legacy_baseline.py` (shapes h_q in {16,128}, s_q=1, b in {1,64,128}, s_k in {1024,4096,8192}).
