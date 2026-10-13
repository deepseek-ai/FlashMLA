# Phase 1: the decode kernel, the graph-safe API, the 8-bit KV cache and the fused layer

Phase 1 answers one question: how fast can MLA decode run on a card with 99 KB of shared memory per block, half-rate
FP32-accumulate tensor cores and no Hopper features? The answer is a new kernel (`impl="v2"`) and the pieces around it.
Results are in [01-results.md](01-results.md).

![Figure 2](figures/fig2_decode_kernel.png)

## 1. Where the original port stands on this card

The Ampere port this branch starts from (`csrc/flash_fwd_mla_kernel_sm80.h`) has four problems on an RTX 3090.

1. **It does not launch.** Q (32 rows × 576), double-buffered K, P and the reduction buffer need 113,280 B of shared
   memory; sm_86 allows 101,376 B. `cudaFuncSetAttribute` fails and the process exits.
2. **Pages must be 32 tokens.** vLLM uses 64, so the kernel cannot be dropped into it.
3. **The inner loop is slow.** Warps are laid out 2 × 4 over M × N, so each warp computes a 16 × 8 slice of S and reloads
   its Q fragments from shared memory for every MMA; P goes through shared memory; there are about four CTA-wide
   barriers per iteration and only two pipeline stages. Its author measured 464 GB/s (23% of the peak) and 59 TFLOPS
   (19%) on an A100.
4. **Fixed cost.** The combine is a separate kernel for every shape, the scheduler metadata is computed by a single
   thread, and only bf16 is supported.

The first step was the smallest change that lets this kernel run: 16 query rows per CTA instead of 32 (4 warps, 93.5 KB,
no register spills). That is `impl="legacy"`, the baseline. It already reads at 92–98% of the DRAM peak for h_q=16 and
large batches and reaches 41–44 TFLOPS (60–63%) at h_q=128.

## 2. The v2 kernel

Source: `csrc/sm86/mla_decode_sm86.cuh` (bf16 / fp16), `csrc/sm86/mla_sm86.h` (tile plans), `csrc/sm86/sm86_ptx.cuh`
(inline PTX).

### 2.1 Layout: Q in registers, QK split along K, PV split by columns

One CTA owns BM query rows (heads × s_q) and walks a contiguous range of cached tokens.

- **BM = 16**: 4 warps. **BM = 32**: 8 warps, organized as 2 groups of 4 that each own 16 rows.
- Warp `j` of a group owns dimensions `[144j, 144j + 144)` of the 576-dim QK product (9 `k16` steps). Its slice of Q is
  loaded once and stays in registers as MMA A-fragments (36 registers). Q never enters shared memory, and that is what
  frees the shared memory for a deep KV pipeline.
- For PV, warp `j` owns value columns `[128j, 128j + 128)`. Its output accumulator is 16 × 128 fp32 values in 64
  registers.

### 2.2 One tile, step by step

1. **QK.** Each warp multiplies its 144 dimensions against the tile: a partial S of 16 rows × Nt tokens.
2. **Exchange.** Every warp writes its partial S into a double-buffered shared-memory slot. After a named barrier
   (`bar.sync id, 128`, group-local) every warp reads the four partials and adds them in the same fixed order, so all four
   warps hold a bit-identical S.
3. **Softmax** is computed redundantly in every warp (row maximum, running sum, rescale of the accumulator when the
   maximum grows). Because S is identical everywhere, no second exchange is needed.
4. **P stays in registers.** The fp32 probabilities are converted in place to bf16 A-fragments; there is no trip through
   shared memory.
5. **PV.** The same KV tile is read again, transposed (`ldmatrix.trans`), and each warp accumulates its 128 columns.

### 2.3 Pipeline

- All warps issue `cp.async` copies into a multi-stage ring; `cp.async.wait_group` plus one CTA barrier per stage
  hand tiles over.
- Shared-memory rows are XOR-swizzled so that both `ldmatrix` and `ldmatrix.trans` are free of bank conflicts.
- The page id of the next tile is looked up one tile ahead (b=128, s_k=4096: 832 → 734 µs).
- A tile never crosses a page: Nt divides the page size. Tails are zero-filled with `cp.async` of source size 0, so
  padding (even NaN) cannot leak into the result.
- Masking applies only to the last tile of a request and to the causal tail when s_q = 2.

### 2.4 Plans and resources

| Instance | Registers per thread | Shared memory | Tile and stages | Used for |
|---|---|---|---|---|
| original fork | | 113,280 B: does not launch | 32 tokens, 2 stages | |
| v2, BM16 | 187 | 59.4 KB | 16 tokens, 3 stages | M ≤ 16 |
| v2, BM32 | 212 | 90.0 KB | 32 tokens, 2 stages | M > 16 |
| v2, 8-bit KV | 218 (two KV groups: 254–255) | 49.4 KB with one group | 32 tokens, 2 stages | M ≤ 16 with `kv_dtype` |

Every instance compiles without register spills and runs one CTA per SM. With 32 rows, 32-token tiles are 12–13% faster
than 16-token tiles (h_q=128, b=64, s_k=4096: 1602 → 1394 µs). For h_q=128 the 128 rows are cut into 4 m-blocks of 32,
each with 20 parts; the four m-blocks read the same KV, which they share through L2 (74.7% hit rate).

With a page size of 16 the BM32 tile is limited to 16 tokens, and h_q=128 is 16–18% slower than with page 64. Page 64
and page 32 perform the same.

## 3. Split-KV scheduling

Source: `csrc/flash_fwd_mla_metadata.cu`.

All requests are laid on one line of 32-token units; every request costs `max(1, ceil(seqlen / 32)) + 2` units (the
2 units model its fixed overhead). The line is cut into `Pe = min(P, total)` equal parts, P = 82 for BM16, and part `p`
owns units `[floor(p · total / Pe), floor((p + 1) · total / Pe))`. Each part becomes one persistent CTA.

- **Parallel and on the device.** Every part finds its begin and end with a binary search over the prefix sums, in one
  small kernel. No value is read back by the host, which keeps the whole decode capturable in a CUDA graph.
- **Balanced cut.** An earlier cut that gave every part `ceil(total / P)` units left SMs idle (b=1, s_k=8192: only about
  65 of 82 parts had work). The balanced cut gives every SM a part. Phase 3 revisits the part count from the point of
  view of the whole layer (item P2 in [04-phase3-plan.md](04-phase3-plan.md)).
- **Metadata row** (8 ints per part): begin index, begin offset, end index, end offset, first split index, the sequence
  length of the first request (saves the CTA one dependent load at start-up), the load per part in units (used by the
  8-bit kernel, see phase 2), and a spare.
- The granule is fixed at 32 so that the legacy kernel can consume the same metadata.

A request that spans several parts is split. Each of its parts writes a partial output (fp32) and a log-sum-exp into a
preallocated workspace, and a combine kernel merges them: `O = Σ_s exp(lse_s − lse) · O_s`.

## 4. Combine variants

For few output rows the combine is a "wide" kernel: one CTA per (row, quarter of the 512 columns), with all loads of the
partials issued before the LSE weights are known. For many rows it is one warp per row. The variant is picked by row
count: up to 16 rows wide16, up to 32 wide8, up to 96 wide4, otherwise warp-per-row.

Effect: b=1 9.2 → 5.5 µs; b=32, s_k=1024 11.7 → 5.2 µs; h_q=16, b=64, s_k=4096 387 → 374 µs for the whole decode.

## 5. A graph-safe API

`flash_mla_with_kvcache(..., out=, lse=, workspace=)` accepts preallocated outputs and a persistent workspace (partial
outputs, partial LSE, counters), sized for the upper bound `b + parts`. With these three arguments the call allocates
nothing and never synchronizes with the host. The grid depends only on the part count, which is static, so the launch can
be recorded in a CUDA graph and replayed while sequence lengths change. `alloc_mla_workspace` creates the workspace once;
all layers and steps reuse it.

## 6. The 8-bit KV cache

Source: `csrc/sm86/mla_decode_sm86_q.cuh`, `flash_mla/kv_quant.py`; format in
[../kv_quant_format.md](../kv_quant_format.md).

- **Format.** The 512 latent values of a token are stored as FP8 e4m3 or INT8 with one fp32 scale per token; the 64 RoPE
  values stay in bf16. A row is 640 bytes plus 4 bytes of scale: 644 instead of 1152 bytes.
- **Dequantization in the kernel.** sm_86 has no FP8 conversion instruction and no FP8 tensor cores. After `cp.async`
  lands the bytes in shared memory, the kernel converts them to bf16 fragments with LOP3 / PRMT bit operations (FP8) or
  through an fp32 magic-number addition (INT8). The MMA is still bf16 with FP32 accumulation. The per-token scale is applied to S
  and P outside the MMA.
- **Quantize on write.** The fused prologue (section 7) quantizes each new token as it writes it; the result is
  bit-identical to the PyTorch reference in `kv_quant.py`.
- **Where it helps.** Only where bandwidth is the limit. At h_q=16, b≥32: eager 1.56–1.71×, graph 1.47–1.58× in phase 1.
  At h_q≥64 it is slower than bf16 (0.93–0.97×).
- **Why not 1.79×.** The profiler shows DRAM at only 86.6% and math-pipe throttle as the largest stall: the kernel has
  one warp per scheduler and nothing hides the latency of the tensor instruction. Phase 2 fixes this with a second group
  of warps.
- **Accuracy.** INT8 is 2.4–13× more accurate than FP8 at the same size (table in the results document).

An opt-in plan runs two CTAs per SM (`FLASH_MLA_SM86_QCFG=16,2,2,32`). It doubles the part count: 3–11% faster for long
contexts, 0–13% slower at s_k ≤ 2048. The phase-2 default supersedes it.

## 7. The fused prologue and the decode layer

Source: `csrc/fused/mla_decode_prologue.cu`, `flash_mla/mla_prologue.py`, `flash_mla/mla_decode_layer.py`.

A DeepSeek-V2-Lite decode layer is `q_proj / kv_a_proj → W_UK absorption → [RoPE(q_pe), RoPE(k_pe), RMSNorm(kv_c),
concatenate q, write the new KV row into the paged cache] → attention → W_UV → o_proj`. In PyTorch the bracketed part is
about 30 small kernels (index, gather, elementwise, copy).

- **One kernel for the bracket.** The fused prologue does RoPE on q_pe and k_pe, RMSNorm on kv_c, builds
  `q = [q_nope_absorbed | q_pe]` and writes `[kv_c | k_pe]` into the right page slot, optionally quantizing it. It
  computes positions and page slots itself from the sequence lengths and the block table. Inside a graph it takes
  1.8–14 µs; the same operations in PyTorch take 59–90 µs inside a graph and 552–562 µs eager.
- **q_proj and kv_a_proj are one GEMM** over a concatenated weight (3648 × 2048).
- **The whole step is one CUDA graph** (`capture_layer_graph`): metadata kernel once per step, then for every layer the
  kernels of that layer. Per layer: 40–43 kernels unfused, 8 fused (7 after phase 2).

The eager layer is launch-bound: about 40 launches cost 820–900 µs regardless of the batch size. The graph removes the
launch cost (b=1: 820 → 115 µs) and fusion removes the small kernels and the gaps between them (115 → 66 µs).

## 8. What the profiler says about what is left

- **h_q=16, large batch**: bandwidth-bound at 95% of the DRAM peak; tensor pipe 16%. Nothing left in the kernel.
- **h_q=128**: tensor pipe at 38.2%, which is 76% of what FP32 accumulation allows on this card. With L2-resident data
  the BM16 kernel does 36 TFLOPS-equivalent; its cycles per instruction (4.4) are 1.0 issued, 1.48 math-pipe throttle,
  0.95 fixed-latency wait, 0.34 short scoreboard, 0.20 long scoreboard, 0.16 barrier. What remains is dependency latency
  that only more warps could hide.
- **Small batch.** A trace of the 82 CTAs at b=1, s_k=8192 shows the median CTA getting its first tile only after
  **4.1 µs** (metadata row → block table → `cp.async` → DRAM: three serial memory round trips, plus the launch). It then
  streams about 100 tokens at the DRAM peak and finishes at 16–18 µs; the tail (last tile, write of the partial output,
  imbalance) is about 3 µs; the combine adds 5.3–5.9 µs. The kernel is at the DRAM limit as soon as it runs; the gap at
  b≤4 is fixed cost. Ampere has neither programmatic dependent launch nor clusters to overlap it.

That last finding, repeated at the level of the whole layer, is what phases 2 and 3 are about.

## 9. Tools built along the way

| Tool | Purpose |
|---|---|
| `benchmarks/phase0_peaks.py` | bandwidth and tensor peaks of the card |
| `benchmarks/bench_decode.py` | eager and graph timing of the decode kernel for v2, legacy, Triton; GB/s, TFLOPS, share of the peak; thermal control |
| `benchmarks/bench_flashinfer.py` | the same shapes on FlashInfer, in its own environment |
| `benchmarks/bench_layer.py`, `bench_prologue.py` | whole layer and prologue, fused and unfused, eager and graph |
| `benchmarks/eval_kv_quant.py` | accuracy of the 8-bit formats |
| `benchmarks/sweep.py` | builds and times kernel instances over (rows, tile, stages, CTAs per SM) |
| `benchmarks/summarize_matrix.py` | turns the CSV logs into `docs/bench_matrix.md` |
| `tools/prof_ncu.sh`, `tools/ncu_summary.py` | Nsight Compute runs and a key-metric summary |
| `tools/prof_nsys.sh`, `tools/nsys_breakdown.py` | Nsight Systems runs and a per-layer kernel breakdown |
