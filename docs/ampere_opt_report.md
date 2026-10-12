# FlashMLA on Ampere (RTX 3090, sm_86): optimization report

Status: **skeleton**. Filled in as the work lands. Owner: SA (system + fusion). Detailed per-version numbers live in
`docs/perf_log_sys.md` (system side) and `docs/perf_log_kernel.md` (kernel side); raw logs in `/data2/LiuJun/tmp/flashmla/{logs,ncu,nsys}`.

## 1. Setup and measured peaks

- Toolchain / environment: see `docs/perf_log_sys.md` section 0.
- Measured peaks on one RTX 3090 (GPU1): D2D copy 841.6 GB/s, read-only 873.8 GB/s, bf16 matmul 70.0 TFLOPS, fp16 66.4 TFLOPS.

## 2. Optimization list

| # | change | where | status |
|---|---|---|---|
| 1 | Legacy kernel made runnable on sm_86 (BlockM=16, 4 warps) | `csrc/flash_fwd_mla_kernel_sm80.h` | done (Phase 0) |
| 2 | New sm86 kernel (Q in registers, QK split-K inside the CTA, multi-stage cp.async KV ring) | `csrc/sm86/` | SK |
| 3 | Fused split-KV combine, persistent workspace, CUDA-Graph-safe API | `csrc/sm86/`, `flash_mla/flash_mla_interface.py` | SK |
| 4 | Fused decode prologue (RoPE + q concat + paged KV write, optional FP8/INT8 quantize-on-write) | `csrc/fused/` | SA |
| 5 | Whole-layer CUDA Graph | `flash_mla/mla_decode_layer.py` | SA |
| 6 | Quantized KV cache (FP8 e4m3 / INT8, per-token scale) | `docs/kv_quant_format.md`, `csrc/fused/`, `csrc/sm86/` | SA (write) + SK (read) |

## 3. Before / after per shape

TBD (filled from `benchmarks/bench_decode.py` CSVs).

## 4. Profiling evidence (ncu / nsys)

TBD.

## 5. Decode-layer fusion and CUDA Graph

TBD.

## 6. Quantized KV accuracy

TBD.

## 7. What did not pan out

TBD.
