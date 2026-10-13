# FlashMLA on Ampere (RTX 3090, sm_86): optimization report

Status: **work in progress**. System / fusion / quantization sections (SA) are filled; the kernel sections (new sm86 kernel, fused combine,
FP8/INT8 dequant in the kernel, ncu evidence for it) and the final before/after matrix are filled at integration. Every number below is
backed by a CSV / log under `/data2/LiuJun/tmp/flashmla/{logs,ncu,nsys}` and by the tables in `docs/perf_log_sys.md` (system side) and
`docs/perf_log_kernel.md` (kernel side).

## 1. Setup and measured peaks

* Toolchain: nvcc 12.4.131 (`/data2/LiuJun/cuda/cuda-12.4`, no new toolkit), torch 2.7.1+cu126, triton 3.3.1, python 3.11, CUTLASS pinned at `afa1772`
  (compiles cleanly for sm_86). `/usr/local/cuda-11.8` has no nvcc, so the plan's nvcc 11.8 was replaced.
* Measured peaks on one RTX 3090 (GPU1): D2D copy 841.6 GB/s, read-only 873.8 GB/s (the decode ceiling), bf16 matmul 70.0 TFLOPS, fp16 66.4 TFLOPS.
* Targets from these peaks: memory-bound h_q=16 >= 700 GB/s (80%); compute-bound h_q=128 >= 42 TFLOPS (60%).

## 2. Optimization list

| # | change | where | status |
|---|---|---|---|
| 1 | Legacy kernel made runnable on sm_86: the original needs 113 280 B smem (limit 101 376 B, `cudaFuncSetAttribute` -> `invalid argument` -> `exit(1)`); BlockM=16 / 4 warps = 93.5 KB, 0 spills | `csrc/flash_fwd_mla_kernel_sm80.h` | done (commit `edf148f`) |
| 2 | New sm86 kernel (Q in registers, QK split-K inside the CTA, multi-stage cp.async ring), parallel metadata, graph-safe API with workspace | `csrc/sm86/`, `flash_mla/flash_mla_interface.py` | SK |
| 3 | Fused split-KV combine, FP16, template sweep | `csrc/sm86/` | SK |
| 4 | Fused decode prologue (RoPE + q concat + RMSNorm + paged KV write, optional FP8/INT8 quantize-on-write) as one launch | `csrc/fused/` (extension `flash_mla_fused_cuda`) | done |
| 5 | Whole-layer CUDA Graph, DeepSeek-V2-Lite decode layer | `flash_mla/mla_decode_layer.py` | done (legacy attention; re-run with v2 at integration) |
| 6 | Quantized KV cache format, write path, accuracy gates | `docs/kv_quant_format.md`, `flash_mla/kv_quant.py`, `csrc/fused/` | write side done; in-kernel dequant SK |
| 7 | Benchmark harness, baselines, profiling scripts, sweep driver | `benchmarks/`, `tools/` | done |

## 3. Baselines before the new kernel (GPU1, s_q=1, page 32, bf16, eager, L2 flushed; full tables incl. CUDA-graph mode in `docs/perf_log_sys.md` section 3)

| h_q | b | s_k | legacy (BlockM=16) | vLLM/SGLang Triton | FlashInfer 0.6.6 fa2 |
|---|---|---|---|---|---|
| 16 | 1 | 8192 | 29.7 us, 319 GB/s (36%) | 41.0 us | 88.1 us |
| 16 | 64 | 4096 | 374.8 us, 812 GB/s (93%) | 510.0 us (597 GB/s) | 1409 us |
| 16 | 128 | 8192 | 1415 us, 857 GB/s (98%) | 1861 us (652 GB/s) | 5281 us |
| 128 | 1 | 8192 | 70.7 us, 32 TFLOPS | 130.0 us | 103.4 us |
| 128 | 64 | 4096 | 1681 us, 43.4 TFLOPS (62%) | 3614 us (20 TFLOPS) | 2719 us |
| 128 | 128 | 8192 | 8279 us, 35.3 TFLOPS (50%) | 16580 us | 11530 us |

The fixed legacy kernel is already the best baseline everywhere and sits at 92-98% of the read-only roofline for h_q=16, b >= 64, s_k >= 4096. The remaining
headroom for the new kernel is small-batch latency (b=1: 36% of peak), the b=64 / s_k=1024 corner, h_q=128 efficiency above 62%, the combine launch, FP16, page size 16/64,
and the bandwidth reduction of the quantized cache. First probe of SK's v2a kernel (page 64): h_q=128, b=64, s_k=4096: 1550 us / 47 TFLOPS (67%).

## 4. Profiling evidence

ncu (legacy main kernel, h_q=16, b=64, s_k=4096): DRAM throughput 90.8% of peak, tensor pipe 22%, 168 regs, 93.5 KB smem, 0 spills, long-scoreboard stall 23%,
barrier stall 11%, 34.7 k shared-store bank conflicts (`docs/perf_log_sys.md` section 4). nsys per-op breakdowns of the decode layer in section 5 below.
Kernel-side ncu evidence for the new kernel: `docs/perf_log_kernel.md` (to be merged).

## 5. Decode-layer fusion and CUDA Graph (DeepSeek-V2-Lite shapes, legacy attention, s_k=4096, per layer)

| b | unfused eager | fused eager | unfused graph | fused graph | kernels/layer | fused graph vs unfused eager |
|---|---|---|---|---|---|---|
| 1 | 804.6 us | 76.5 us | 116.3 us | 68.5 us | 40 -> 7 | 11.7x |
| 8 | 876.7 us | 125.2 us | 179.3 us | 113.5 us | 42 -> 8 | 7.7x |
| 32 | 882.8 us | 265.9 us | 324.6 us | 256.4 us | 42 -> 8 | 3.4x |
| 64 | 894.1 us | 453.5 us | 511.9 us | 441.3 us | 43 -> 9 | 2.0x |

* The prologue alone: 552-562 us (20 PyTorch ops, eager) or 59-90 us (same ops inside a graph) -> 5.1-16 us eager / 1.8-13.9 us in a graph as one kernel.
* Eager is launch-bound (about 800 us per layer is ~40 launches); the graph removes the launch cost (unfused 805 -> 116 us at b=1) and fusion removes the remaining elementwise kernels and
  gaps (116 -> 68.5 us at b=1, 1.7x). With larger batches the attention kernel dominates the layer (43% of busy time at b=8) and the gain shrinks to 1.16x at b=64.
* nsys (b=8): unfused 43.8 kernels / 186 us busy per layer, fused 9.8 kernels / 118 us busy per layer.

## 6. Quantized KV accuracy (single layer, dequant-reference attention vs bf16 cache; synthetic MLA-shaped caches, 3 regimes x ctx 1k/4k/16k x q sharpness x1/x4 x 3 seeds)

| format | bytes/token | write path vs torch reference | cos_diff(attention output) | cos_diff(layer output) worst | gate |
|---|---|---|---|---|---|
| bf16 | 1152 | - | 0 | 0 | 8e-5 (lossless gate) |
| fp8 e4m3 + fp32 per-token scale, rope bf16 | 644 (0.559x) | bit-exact | 5e-5 ... 2.4e-3 | 2.4e-3 | < 5e-3 |
| int8 + fp32 per-token scale, rope bf16 | 644 (0.559x) | bit-exact | 6e-6 ... 2.5e-4 | 2.6e-4 | < 5e-4 |

int8 is 2.4-13x more accurate than e4m3 at the same size; both are opt-in. Speed side (target >= 1.6x for memory-bound h_q=16): needs the in-kernel dequant, SK.

## 7. What did not pan out / notes

* The warp-specialized legacy kernel cannot be fixed minimally (the 8-warp 2Mx4N PV tiling is hard-wired to BlockM=32); it now raises a clear RuntimeError on sm_86.
* A dummy sweep (`benchmarks/sweep.py`) takes ~2 min per configuration build with MAX_JOBS=8; sweeps should be bounded to <= 20 instances.
* No real DeepSeek weights are available on the server; accuracy numbers use synthetic statistics, chosen to be adversarial for per-token scaling.
