# What did not work

Every entry here was built or prototyped and measured. They are kept so that nobody spends the time again, and because
several of them changed how the later work was done. "Retry when" names the condition under which the idea deserves
another look. Code for the phase-2 and phase-3 entries is in [research/](../../research/README.md).

## 1. Rules learned

1. **At the ceiling, stop.** For h_q=16 and large batches the repaired legacy kernel already reads at 92–98% of the DRAM
   peak. No kernel design can beat it there by more than noise; v2 ties it. The same holds for tile-plan sweeps once a
   kernel is bandwidth-bound (differences below 1.5%).
2. **Anything that changes the bytes in flight per CTA must be uniform across the launch.** DRAM bandwidth is shared in
   proportion to what each CTA has in flight. Mixing CTAs with one and two tiles in flight made the launch slower than
   either uniform choice, because the starved CTAs decide the tail.
3. **Synthetic microbenchmarks do not decide questions about L2 or DRAM state.** The KV prefetch gained up to 1.9 µs in a
   synthetic chain and lost up to 1.7 µs in the real layer; the best prefetch budget was 1 MB in the chain and 2 MB in
   the layer. Decisions are made by A/B on the real 27-layer graph.
4. **A prefetch is not free and not asynchronous in the way one hopes.** On this GPU `prefetch.global.L2` blocks the
   issuing thread, and the issuing kernel does not retire until the prefetches have landed. It pays only inside a
   latency-bound or DRAM-idle window, issued by all threads.
5. **More warps hide latency; fewer instructions do not.** The 8-bit kernel and the h_q=128 kernel both stall on the
   tensor instruction, not on arithmetic capacity. Removing instructions did nothing; a second warp per scheduler did.
6. **A branch can cost more than the work it skips.** Skipping an identity rescale moved 64 multiplies into their own
   basic block, where the compiler could no longer interleave them with tensor instructions: 3–5% slower.
7. **Fixed cost, not throughput, is what small batches pay for.** At b≤4 the kernel streams at the DRAM peak as soon as
   it runs; the gap is launches, serial start-up loads, tails and the combine.
8. **Compare profiler reports in cycles.** Clock control does not pin the SM clock on this card.

## 2. Phase 1

| Approach | Result | Reason | Retry when |
|---|---|---|---|
| Run the original fork unmodified | fails to launch | 113,280 B of shared memory against a limit of 101,376 B | never |
| Minimal fix of the fork's warp-specialized kernel (`legacy_ws`) | abandoned; raises an error on sm_86 | its 8-warp PV tiling hard-codes 32 rows per CTA | never |
| **v2b**: a producer warp with mbarriers (warp specialization) | correct, not faster | the bandwidth-bound case is already at the peak; the compute-bound case waits on dependencies, not on barriers; with 32 rows the extra warp pushes registers into spills | hardware with TMA / wgmma (sm_90) |
| **v2c**: software-pipelined tile loop | 163 → 157 µs with L2-resident data, nothing on real shapes; 15% slower with 32 rows | the assembler already interleaves tensor, exponential and load instructions inside the loop; register pressure with 32 rows | never |
| **Fused last-arriver combine** (the last CTA of a request merges the partials; counters reset themselves; deterministic) | b=128: 750.6 vs 743.4 µs (noise); b=32, s_k=1024: 73.7 → **81.9 µs** | fence + atomic + a serial merge of 96–640 KB of partials on one SM sit on every split's critical path; rows above the split limit still need the combine launch, so no launch is saved | kept as `FLASH_MLA_SM86_FUSE`, off by default |
| Cooperative in-kernel combine (grid barrier, distributed merge) | estimated only | at most about 3 µs at b≤2, needs a cooperative launch, scales badly with rows | sm_90 clusters / distributed shared memory |
| Two CTAs per SM for the bf16 kernel | 133 vs 163 µs with L2-resident data, not worth it on real shapes | the part count doubles: more partial traffic and a bigger combine | never |
| Scheduler sweeps (overhead units, minimum units per part) | below 2% | the fixed cost is not in the cut granularity | see P2 in the phase-3 plan for a load-dependent rule |
| Tile-plan sweep for h_q=16 (16/3, 16/4, 2 CTAs × 16/2, 32/2, …) | below 1.5% | tile plans do not matter once the kernel is DRAM-bound | never |
| 8-bit KV at h_q≥64 | 0.93–0.97× | compute-bound; dequantization adds work | hardware with FP8 tensor cores |
| CUDA graph for the decode kernel alone at b≥64 | eager ≈ graph | launch cost vanishes in a kernel of hundreds of µs; the graph matters for small batches and for the whole layer | never |
| Beating legacy at h_q=16, large batch, bf16 | no headroom | see rule 1 | never |
| 8-bit kernel with two CTAs per SM as the default | long contexts +3–11%, s_k≤2048 0–13% slower | the part count doubles; short requests are cut into more pieces | kept as an option; superseded by the two KV groups |

## 3. Phase 2

| Approach | Result | Reason | Retry when |
|---|---|---|---|
| Two KV groups **chosen per part** | mixed launches much slower: b=32, s_k=4096 151 vs 122 µs | rule 2 | never; the choice is made per launch |
| Two KV groups always on | 0.84–1.10× of the old default | each part pays for a second Q load and the merge; short contexts lose | never; gated by the load per part |
| Combine reads all splits in one round | 4.28 → 4.26 µs | the rounds already overlapped; the chain from LSE to the weights is the bottleneck | never |
| Softmax skips the rescale when the row maximum is unchanged | decode 3–5% slower (h_q=128: 54.1 → 52.7 TFLOPS) | rule 6 | never |
| h_q=128: M groups half a tile out of phase (ping-pong) | 0.96–0.98× | barrier stall falls (1.30 → 1.08) but instructions rise 5% and math-pipe throttle is unchanged: the FP32-accumulate tensor pipe is the limit (76% of the reachable rate) | FP16 accumulation becomes acceptable, or fewer tensor instructions per tile |
| Fewer dequantization instructions in the 8-bit kernel (convert once per tile into bf16 shared memory) | not built | the profiler put 96% of the math-pipe throttle on the tensor instruction and the ALU pipe at 27–34%: instructions were not the limit | the ALU pipe becomes the limit |
| Weight absorption for the projections (W_q·W_UK or W_UV·W_o merged offline) | not built | about 3× the weight bytes; a pure loss when the projections are bandwidth-bound | never for decode |
| Split-K across CTAs in the skinny GEMM | never faster in the sweep | enough CTAs already; the reduction adds a serial tail | never |
| Skinny GEMM for qkv_a / o_proj above 32 tokens | cuBLAS is as fast or faster | more tokens per weight byte: the regular kernels become efficient | never |
| Moving the scheduler metadata into the decode kernel | not built | the chain depth is the same (the lengths still have to be read), and the metadata is computed once per step for all 27 layers: about 0.15 µs per layer | never |

## 4. Phase 3 research

| Approach | Result | Reason | Retry when |
|---|---|---|---|
| Prefetch the first KV tile before the decode kernel (from the prologue or from a parallel branch) | synthetic: −0.9 to −1.9 µs; in the layer: **+0.1 to +1.7 µs** for all three placements at b = 1, 4, 8 | rule 3: the layer has no purely idle window there; the metadata → block table chain and DRAM competition with W_UK cancel the gain | the decode start-up chain is restructured, or on sm_90+ |
| Physical page numbers in the metadata row | not built: warming only the metadata and the block table gains nothing (42.03 vs 41.96 µs) | that hop is not on the critical path; it would also tie the metadata to the block table | never |
| A separate prefetch kernel between two kernels | 39.33 → 40.03 µs | rule 4, plus one more launch | never |
| Prefetch issued after the GEMM epilogue | 1.1–1.5 µs per layer worse than with the last weight load | too late: the next kernel is about to start | never |
| Prefetch issued by 9 of 128 threads | 300–400 GB/s; first recorded as "prefetch does not help" | rule 4: one prefetch in flight per thread | never; spread over all threads |
| GEMM prefetch without discarding the partials | no gain at b=4 and b=8 | the write-back of the dirty partials competes with the prefetch for DRAM | never; P1 comes first |
| Partial outputs stored as bf16 | not built | the discard removes the write-back; the rest is traffic inside L2; it would change the numerics | DRAM writes of the decode kernel are still visible after P1 |
| Combine and W_UV in one kernel | not built | one CTA per output row against a stream of weight tiles: the structures do not match | P6 fails |
| Spin-waiting to imitate programmatic dependent launch | not built | burns SMs, can deadlock | never |
| A fixed minimum of units per part | b=1: 4 → +0.25, 8 → +7.5, 16 → +23 µs per layer | too few parts lengthen the critical path | never; P2 follows the load |
| Nsight Compute kernel replay to measure L2 state across kernels | implausible numbers | replay does not reproduce the cache state the previous kernel left | never; use graph-timing A/Bs |

## 5. Open, not rejected

- **Small-batch latency of the stand-alone decode kernel.** With less than about 35 MB of KV it is 2.8–4.0 µs over the
  target under a graph. The start-up chain (about 4 µs), the tail (about 3 µs) and the combine (about 5 µs) are fixed
  costs that Ampere offers no feature to overlap; the estimate of what is still removable is 1–3 µs.
- **h_q=8.** Half of every m16 tile is wasted; swapping the MMA operands as in the skinny GEMM should help.
- **Accuracy with real weights.** The 8-bit numbers come from synthetic caches that are deliberately unfriendly to a
  per-token scale.
- **End to end in a serving stack.** v2 supports 64-token pages and a graph-safe workspace, so it can serve as the MLA
  decode back end of vLLM on a 3090; that has not been measured.
