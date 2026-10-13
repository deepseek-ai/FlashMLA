# Results: baseline to optimized

Everything on this page was measured on one otherwise idle RTX 3090. Notation and method are in [README.md](README.md).
Times are microseconds; "per layer" means a CUDA graph of 27 layers divided by 27.

![Figure 4](figures/fig4_results.png)

## 1. Hardware ceilings

| Quantity | Value | How it was obtained |
|---|---|---|
| GPU | RTX 3090, 82 SMs, sm_86, 101,376 B of shared memory per block, 6 MB of L2 | |
| Read-only bandwidth (the ceiling for decode) | **873.8 GB/s** | `benchmarks/phase0_peaks.py`; device-to-device copy: 841.6 GB/s |
| bf16 tensor throughput, FP32 accumulation | **70.0 TFLOPS** | 8192³ matmul; fp16: 66.4 TFLOPS |
| Ridge point | about 80 FLOP/B, M ≈ 42 | 70.0e12 / 873.8e9 |
| Cost of a dependent empty kernel inside a CUDA graph | 0.67 µs | measured |

## 2. Scorecard against the targets set at the start

| Target | Measured | Verdict |
|---|---|---|
| h_q=16, s_q=1, b=64–128, s_k=4k–8k: at least 80% of the measured bandwidth | graph 821–864 GB/s (median 847), 94–99%; eager 93–99% | met |
| h_q=128: at least 60% of the bf16 peak | eager 50.8–52.6 TFLOPS (73–75%); graph 46.5–50.6 (66–72%); fp16 77% of its peak | met |
| b≤4: latency at most 1.3 × (KV bytes / bandwidth) + 5 µs | graph: met from 37.7 MB of KV upward (b=1 s_k=32768, b=4 s_k=8192); 2.8–4.0 µs over with 18.9 MB or less; eager misses everywhere (2.3–8.6 µs) | partly missed |
| FP8 KV in the memory-bound regime: at least 1.6× | graph, b≥32, s_k≥4096: 1.55–1.70× (only b=32, s_k=4096 is below 1.6×); eager ≥ 1.60× | one cell short |
| Faster than Triton, legacy and FlashInfer over the whole matrix | fastest or tied everywhere; one exception: eager h_q=16, b=1, s_k=1024 is 0.93× of legacy | met |
| Correctness | 760 tests pass; sanitizer clean; graph replay with changing lengths correct | met |

## 3. The decode kernel alone

### 3.1 Main shapes against three other implementations (CUDA graph, page size 32, bf16, µs per layer)

| Shape | v2 | legacy | Triton | FlashInfer 0.6.6 | legacy / v2 | Triton / v2 | FlashInfer / v2 |
|---|---|---|---|---|---|---|---|
| h16 b1 s_k1024 | 9.4 | 10.3 | 15.4 | 25.5 | 1.09 | 1.63 | 2.70 |
| h16 b1 s_k4096 | 15.9 | 21.0 | 27.0 | 62.8 | 1.32 | 1.70 | 3.95 |
| h16 b1 s_k8192 | 22.9 | 27.9 | 37.0 | 83.8 | 1.22 | 1.62 | 3.66 |
| h16 b64 s_k4096 | 370.9 | 371.3 | 529.6 | 1405.2 | 1.00 | 1.43 | 3.79 |
| h16 b128 s_k8192 | 1404.4 | 1410.8 | 1928.5 | 5144.6 | 1.00 | 1.37 | 3.66 |
| h128 b1 s_k8192 | 58.8 | 69.5 | 128.8 | 90.2 | 1.18 | 2.19 | 1.54 |
| h128 b64 s_k4096 | 1499.5 | 1779.9 | 3742.6 | 2631.0 | 1.19 | 2.50 | 1.75 |
| h128 b64 s_k8192 | 2885.2 | 3509.2 | 7733.4 | 5134.6 | 1.22 | 2.68 | 1.78 |
| h128 b128 s_k8192 | 5820.5 | 6991.5 | 15777.2 | 10254.9 | 1.20 | 2.71 | 1.76 |

- "legacy" is the fork's kernel with 16 query rows per CTA so that it fits into shared memory. The unmodified fork does
  not launch.
- Triton is the SGLang / vLLM grouped decode kernel. FlashInfer falls into its small-shared-memory configuration on this
  card (64-row Q tile in shared memory, 16-token KV tile, one stage) and computes QK redundantly in both warp groups.

### 3.2 Geometric mean of `implementation / v2` over the matrix (minimum–maximum)

| Mode | h_q | legacy | Triton | FlashInfer |
|---|---|---|---|---|
| graph, page 32 | 16 | 1.07 (1.00–1.32) | 1.47 (1.34–1.70) | 3.43 (2.70–4.18) |
| graph, page 32 | 128 | 1.17 (1.02–1.23) | 2.23 (1.17–2.71) | 1.77 (1.54–2.18) |
| eager, page 32 | 16 | 1.03 (0.93–1.25) | 1.38 (1.20–1.52) | 3.11 (2.07–3.98) |
| eager, page 32 | 128 | 1.16 (1.01–1.24) | 2.22 (1.13–2.64) | 1.77 (1.39–2.13) |
| graph, page 64, full matrix | 8 / 16 / 32 / 64 / 128 | not applicable (legacy needs page 32) | 1.41 / 1.63 / 2.07 / 2.29 / 2.46 | 3.17 / 2.87 / 2.20 / 1.80 / 1.73 |

- The one point where v2 is slower than legacy is eager h_q=16, b=1, s_k=1024 (15.4 vs 14.3 µs): 1.2 MB of KV, almost all
  of the time is launch and fixed cost. Under a graph v2 is ahead (9.4 vs 10.3 µs).
- A tie at h_q=16 and b≥64 is expected: the repaired legacy kernel already reads at 92–98% of the DRAM peak. The profiler
  shows both kernels reading the same 304 MB in times 0.2% apart.
- FP16 behaves like bf16 (h_q=128, b=64, s_k=4096, eager: 1435 vs 1421 µs).

### 3.3 Distance to the ceilings (v2, bf16, page 64)

| Regime | Measured | Share of the measured peak |
|---|---|---|
| h_q=16, b≥64, s_k≥4096, eager | 814–862 GB/s (median 843) | 93–99% |
| h_q=16, b≥64, s_k≥4096, graph | 821–864 GB/s (median 847) | 94–99% |
| h_q=128, b≥16, s_k≥4096, eager | 50.8–52.6 TFLOPS (median 51.9) | 73–75% |
| h_q=128, b≥16, s_k≥4096, graph | 46.5–50.6 TFLOPS (median 49.9) | 66–72% |
| For reference: the original kernel on an A100 (its author's numbers) | 464 GB/s, 59 TFLOPS | 23%, 19% of the A100's peaks |

The h_q=128 kernel is about 4% slower under a graph than eager. Sampled SM clocks are similar; power is 318 W against
269 W. The likely cause is clock and power behaviour of a compute-bound kernel under sustained load; this was not pinned
down.

### 3.4 Profiler evidence (Nsight Compute, full metric set, page 64, bf16 unless noted)

| Shape and kernel | Time | Key metrics | Reading |
|---|---|---|---|
| h16 b64 s_k4096, v2 (16 rows, 16-token tiles, 3 stages; 82 CTAs × 128 threads) | 365.2 µs | reads 304.4 MB = 833 GB/s (95%); tensor pipe 16%; no shared-memory bank conflicts on loads | bandwidth-bound, at the peak |
| same shape, legacy | 365.8 µs | reads 304.4 MB; 93.5 KB of shared memory; 49% more shared-memory load traffic than v2 | tied; the extra traffic hides behind the bandwidth limit |
| h16 b64 s_k4096, v2 FP8 (phase 1) | 222.3 µs (1.64×) | reads 171.2 MB; DRAM 86.6%; stalls: math-pipe throttle 32.1%, wait 13.5%, long scoreboard 5.0% | not bandwidth-bound; see phase 2 |
| h128 b64 s_k4096, v2 (32 rows, 32-token tiles, 2 stages; 4 × 20 CTAs × 256 threads) | 1.389 ms | tensor pipe 38.2% (100% would be the FP16-accumulate rate; FP32 accumulation caps at 50%, so this is 76% of what is reachable); L2 hit rate 74.7% | compute-bound, close to the reachable ceiling |
| same shape, legacy | 1.523 ms | tensor pipe 31.6%; L2 hit rate 86.5% | v2 is 1.21× faster in cycles (2.35 M vs 2.85 M) |
| h16 b1 s_k8192, v2 kernel + combine | 21.9 + 7.6 µs | the kernel reads 9.5 MB | the combine is a quarter of the time: fixed cost dominates small batches |

## 4. The 8-bit KV cache

Format: 512 latent values as FP8 e4m3 or INT8 with one fp32 scale per token, the 64 RoPE values kept in bf16.
644 bytes per token instead of 1152 (0.559×), so 1.79× would be the ideal speed-up of a purely bandwidth-bound kernel.

### 4.1 Phase 1 default (one group of 4 warps per CTA): h_q=16, page 64, 8-layer graph, same harness and GPU

| b | s_k | bf16 eager / graph µs | FP8 eager / graph µs (speed-up) | INT8 eager / graph µs (speed-up) |
|---|---|---|---|---|
| 32 | 4096 | 196.6 / 190.9 | 126.0 / 121.6 (1.56× / 1.57×) | 121.8 / 125.1 (1.61× / 1.53×) |
| 32 | 8192 | 368.6 / 364.4 | 216.1 / 230.8 (1.71× / 1.58×) | 223.2 / 237.7 (1.65× / 1.53×) |
| 64 | 4096 | 374.8 / 370.4 | 222.2 / 234.9 (1.69× / 1.58×) | 225.3 / 242.4 (1.66× / 1.53×) |
| 64 | 8192 | 714.8 / 712.2 | 431.1 / 458.9 (1.66× / 1.55×) | 447.5 / 475.6 (1.60× / 1.50×) |
| 128 | 4096 | 722.9 / 719.8 | 432.9 / 462.3 (1.67× / 1.56×) | 450.6 / 479.5 (1.60× / 1.50×) |
| 128 | 8192 | 1403.9 / 1402.4 | 865.3 / 911.5 (1.62× / 1.54×) | 897.0 / 943.7 (1.57× / 1.49×) |

Small batches gain less because fixed cost dominates: geometric mean 1.24× (1.08–1.39) for b≤4 and 1.45× at b=16 under a
graph. At h_q≥64 the 8-bit cache is slower than bf16 (0.93–0.97×).

### 4.2 Phase 2 default (two KV groups per CTA when the load per part is at least 64 units)

Time of the phase-1 default divided by time of the phase-2 default, FP8, graph, h_q=16, page 64 (above 1 = faster now):

| b (rows) × s_k (columns) | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 1.01 | 1.02 | 1.02 | 1.00 | 1.05 |
| 32 | 1.00 | 1.01 | 0.99 | 1.04 | 1.08 |
| 48 | 1.03 | 1.01 | 1.03 | 1.08 | 1.06 |
| 64 | 1.02 | 1.01 | 1.06 | 1.10 | 1.07 |
| 96 | 1.00 | 1.04 | 1.08 | 1.12 | not run |
| 128 | 1.00 | 1.05 | 1.11 | 1.07 | not run |

- FP8 over the whole grid: 0.993–1.115×; nothing below 0.999 at s_k≤2048.
- Against bf16, graph, b≥32, s_k≥4096: **FP8 1.55–1.70×**, **INT8 1.54–1.60×**. The only FP8 cell below 1.6× is b=32,
  s_k=4096: its load is 50 units per part, so it runs one group.
- INT8 against its phase-1 default: 0.992–1.072×.
- Eager has cells at 0.97–0.99 where the load per part is 75–103 units (the same cells gain 3–5% under a graph); at
  s_k≥8192 and b≥64 eager gains 3–8% too.

### 4.3 The opt-in "two CTAs per SM" plan (`FLASH_MLA_SM86_QCFG=16,2,2,32`), phase 1

Speed-up over bf16, graph, h_q=16:

| b | s_k | FP8 default | FP8, 2 CTAs per SM | INT8 default | INT8, 2 CTAs per SM |
|---|---|---|---|---|---|
| 4 | 4096 | 1.38× | 1.20× | 1.36× | 1.20× |
| 16 | 4096 | 1.53× | 1.44× | 1.50× | 1.41× |
| 32 | 8192 | 1.58× | 1.65× | 1.53× | 1.61× |
| 64 | 4096 | 1.58× | 1.61× | 1.53× | 1.60× |
| 64 | 8192 | 1.55× | 1.68× | 1.50× | 1.62× |
| 128 | 4096 | 1.56× | 1.58× | 1.51× | 1.57× |
| 128 | 8192 | 1.54× | 1.61× | not run | 1.58× |

It doubles the number of parts, which helps long contexts and hurts short ones (0.87–0.94× at s_k=1024). The phase-2
default gets the long-context gain without that loss, so this plan is kept only as an option.

### 4.4 Accuracy

Single layer, dequantized cache against the bf16 cache; synthetic MLA-shaped caches (3 distributions × context 1k / 4k /
16k × query sharpness ×1 / ×4 × 3 seeds). No real model weights were available on the test machine.

| Format | Bytes per token | Write path against the PyTorch reference | Cosine distance, attention output | Worst cosine distance, layer output | Gate |
|---|---|---|---|---|---|
| bf16 | 1152 | | 0 | 0 | 8e-5 (lossless gate) |
| FP8 e4m3 + fp32 scale, RoPE bf16 | 644 | bit-identical | 5e-5 to 2.4e-3 | 2.4e-3 | below 5e-3 |
| INT8 + fp32 scale, RoPE bf16 | 644 | bit-identical | 6e-6 to 2.5e-4 | 2.6e-4 | below 5e-4 |

INT8 is 2.4–13× more accurate than FP8 at the same size and nearly the same speed. Prefer INT8.

## 5. The whole decode layer

Structure (DeepSeek-V2-Lite, h_q=16): q_proj / kv_a_proj → W_UK absorption → prologue → attention → W_UV → o_proj.

### 5.1 Phase 1: fusion and the whole-layer graph (µs per layer, s_k=4096)

| b | Unfused, eager | Unfused, graph | Fused graph, legacy kernel (page 32) | Fused graph, v2 bf16 | v2 FP8 | v2 INT8 | Fused graph, Triton (page 64) | Fused v2 over unfused eager |
|---|---|---|---|---|---|---|---|---|
| 1 | 820.1 | 114.7 | 71.1 | **65.8** | 66.1 | 65.4 | 76.4 | 12.5× |
| 8 | 895.1 | 184.9 | 115.7 | 117.9 | 102.3 | **101.5** | 133.8 | 7.6× |
| 32 | 903.1 | 330.0 | 258.2 | 259.3 | **194.3** | 194.8 | 355.7 | 3.5× |
| 64 | 899.2 | 515.0 | 440.9 | 440.0 | **310.5** | 322.6 | 599.6 | 2.0× |

At s_k=1024: b=1 825.8 → 58.2 µs (**14.2×**); b=64 890.0 → 177.4 µs (5.0×), 142.7 µs with FP8.

- The eager layer is launch-bound: about 40 launches cost 820–900 µs whatever the batch size. The graph removes the
  launch cost (b=1: 820 → 115 µs); fusion removes the small elementwise kernels and the gaps between them
  (115 → 66 µs).
- With larger batches attention dominates and fusion matters less (1.17× at b=64). There the 8-bit cache is the lever:
  b=64, s_k=4096 gains another 1.42× (440 → 310.5 µs).

### 5.2 Phase 2: skinny GEMM, same-build A/B (27-layer graph, page 64, µs per layer)

| Variant | b1 s1k | b1 s4k | b8 s1k | b8 s4k | b32 s1k | b32 s4k | b64 s1k | b64 s4k |
|---|---|---|---|---|---|---|---|---|
| bf16 KV, cuBLAS projections | 57.7 | 65.2 | 81.5 | 115.4 | 122.2 | 255.9 | 175.3 | 438.2 |
| bf16 KV, skinny GEMM | 54.1 | 61.3 | **68.4** | **102.0** | 106.8 | 239.3 | 168.4 | 430.7 |
| FP8 KV, cuBLAS projections | 58.5 | 65.2 | 80.7 | 100.4 | 106.4 | 188.4 | 140.2 | 303.2 |
| FP8 KV, skinny GEMM | 54.9 | 61.5 | **66.8** | **86.1** | 90.5 | 171.9 | 134.1 | 300.8 |

The four GEMMs at b=8 in isolation: 61.7 → 41.5 µs; inside the layer (Nsight Systems): 62.4 → 43.4 µs.

### 5.3 The layer after phase 2 (27-layer graph, µs per layer, s_k=4096)

| b | bf16 KV | FP8 KV | Phase 1 bf16 / FP8 |
|---|---|---|---|
| 1 | 61.5 | 61.6 | 65.8 / 66.1 |
| 8 | **102.3** | **86.3** | 117.9 / 102.3 |
| 32 | 239.5 | 172.0 | 259.3 / 194.3 |
| 64 | 431.4 | 291.2 | 440.0 / 310.5 |

With FP8 KV and long contexts, the two KV groups add: b=64, s_k=8192 523.3 → 487.3 µs (1.07×); b=64, s_k=4096
300.7 → 291.2 µs; b=32, s_k=8192 284.7 → 275.3 µs.

### 5.4 Cumulative: one layer from the baseline to the current build (µs per layer, s_k=4096)

| Stage | b=1 | b=8 | b=32 | b=64 |
|---|---|---|---|---|
| Unfused PyTorch, eager | 820.1 | 895.1 | 903.1 | 899.2 |
| Unfused PyTorch, whole-layer CUDA graph | 114.7 | 184.9 | 330.0 | 515.0 |
| Fused layer + Triton attention | 76.4 | 133.8 | 355.7 | 599.6 |
| Fused layer + legacy kernel | 71.1 | 115.7 | 258.2 | 440.9 |
| Phase 1: fused layer + v2 kernel | 65.8 | 117.9 | 259.3 | 440.0 |
| Phase 2: + skinny GEMM (bf16 KV) | 61.5 | 102.3 | 239.5 | 431.4 |
| Phase 2: + 8-bit KV cache (FP8) | 61.6 | 86.3 | 172.0 | 291.2 |
| Speed-up of phase 2 (bf16) over unfused eager | 13.3× | 8.7× | 3.8× | 2.1× |
| Speed-up of phase 2 (bf16) over unfused graph | 1.87× | 1.81× | 1.38× | 1.19× |
| Speed-up of phase 2 (FP8) over unfused graph | 1.86× | 2.14× | 1.92× | 1.77× |

### 5.5 What a layer consists of (Nsight Systems, b=8, s_k=4096, µs per layer)

| Version | Kernels per layer | Attention | Combine | Prologue | GEMMs | Other | Busy time |
|---|---|---|---|---|---|---|---|
| Unfused (legacy attention, page 32) | about 44 | 51.5 | 4.2 | spread over small PyTorch kernels, counted under "other" | 67.8 (5 GEMMs) | about 80 (about 36 elementwise / copy / index kernels) | about 203 |
| Fused, v2, bf16 KV (phase 1) | 8 | 51.6 | 6.7 | 2.5 | 60.4 | 2.3 | about 123.5 |
| Fused, v2, FP8 KV (phase 1) | 8 | 33.8 | 6.4 | 4.0 (includes quantize-on-write) | 58.7 | 2.3 | about 105 |

One metadata kernel (4.2 µs) runs once per decode step and is shared by all layers. Phase 2 takes the four GEMMs plus
the reshape copy (62.4 µs in this trace) down to 43.4 µs and leaves 7 kernels per layer.

### 5.6 One layer at batch 1 on the time axis (phase-2 build, s_k=4096, bf16)

Per-kernel times from an Nsight Systems trace of the 27-layer graph; each includes about 0.4 µs of tracing overhead.

| Kernel | Grid | Time µs | Byte floor µs | Above the floor |
|---|---|---|---|---|
| qkv_a (skinny GEMM, 14.9 MB of weights) | 114 × 1 × 1 | 20.80 | 17.1 | +3.7 |
| W_UK (skinny GEMM over 16 heads, 2.1 MB) | 32 × 1 × 16 | 4.77 | 2.4 | +2.4 |
| prologue | (h_q + 1) × T | 2.40 (FP8: 3.84–4.06) | 0 | +2.4 |
| decode v2 (82 parts, 4.7 MB of KV) | 82 | 12.64 | 5.4 | +7.2 |
| combine (reads 2.6 MB of partials from L2) | by row count | 4.29 (b=8: 6.46) | 0 | +4.3 |
| W_UV (skinny GEMM over 16 heads, 2.1 MB) | 8 × 1 × 16 | 4.87 | 2.4 | +2.5 |
| o_proj (skinny GEMM, 8.4 MB) | 64 × 1 × 1 | 14.46 | 9.6 | +4.9 |
| Total | | 64.2 | 37.0 | +27 |

This table is the starting point of phase 3.

## 6. Phase 3 prototypes (research code, not in the library)

Measured on the real 27-layer graph, s_k=4096 unless noted, µs per layer. "Stack" is P1 (as an external kernel after the
combine) + P2 (emulated through the minimum-units knob) + P3 (a patched copy of the GEMM kernel, 2 MB budget).

| Shape | Phase-2 build | P1 + P2 + P3 stack | Change |
|---|---|---|---|
| b=1, bf16 | 61.12 | 56.18 | −4.9 |
| b=2, bf16 | 67.45 | 61.35 | −6.1 |
| b=4, bf16 | 78.4 | 73.65 | −4.8 |
| b=8, bf16 | 101.7 | 97.4 | −4.3 |
| b=1, FP8 | 61.6 | 57.37 | −4.2 |
| b=8, FP8 | 86.9 | 82.45 | −4.5 |

The external discard kernel costs one extra launch (0.7–0.9 µs) that disappears once the discard lives inside the
combine. The harnesses are in [research/phase3/](../../research/README.md). Per-item numbers and the projection for all
six items are in [04-phase3-plan.md](04-phase3-plan.md).

## 7. Correctness

- 760 tests: v2 kernels 140, 8-bit kernels 226, fused prologue 24, fused layer 8, KV quantization 18, end-to-end layer
  152, skinny GEMM 128, the fork's own test with `impl="legacy"` and `impl="v2"` 64.
- Coverage: output and LSE against an fp32 reference (cosine distance below 8e-5); KV padded with NaN beyond the
  sequence length; page sizes 16 / 32 / 64; h_q 8 / 16 / 32 / 64 / 128; s_q 1 and 2 with the causal tail; variable
  lengths; b=1 with 32k tokens (many splits); lengths below one tile and of exactly one token; bf16 and fp16; CUDA-graph
  replay with changed lengths; two runs bit-identical; workspace reuse across shapes; the 8-bit accuracy gates.
- `compute-sanitizer` memcheck, racecheck, synccheck and initcheck report no errors.
- The skinny GEMM is checked against fp32 `F.linear` / `bmm` and for run-to-run bit equality; quantize-on-write in the
  fused prologue is bit-identical to the PyTorch reference.
