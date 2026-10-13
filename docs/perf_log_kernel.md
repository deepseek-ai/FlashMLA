# Kernel performance log (SK): sm_86 MLA decode kernels

Everything here was measured on GPU 0 of the server (RTX 3090, sm_86, 82 SMs, clocks unlocked: ~1.7-1.9 GHz under load),
nvcc 12.4 + torch 2.7.1+cu126, CUDA C++ / inline PTX only. Peaks measured by SA (`docs/perf_log_sys.md`): DRAM read-only
**873.8 GB/s**, bf16 MMA (fp32 accumulate) **70.0 TFLOPS**. Raw logs: `/data2/LiuJun/tmp/flashmla/logs/sk_*`, profiles in
`/data2/LiuJun/tmp/flashmla/{ncu,nsys}/sk_*`. Timing harness: SA's `benchmarks/bench_decode.py` (graph mode = an 8-layer CUDA
graph, every layer with its own KV cache, time per layer; eager = L2-flushed single call) unless noted.

## 1. What was built

| variant | what | status |
|---|---|---|
| **v2a** (default) | Q in registers, QK split-K over 4 warps (144-dim slices), PV split in 128-column slices, partial-S exchange through smem + named barrier, P stays in registers (A fragments), multistage `cp.async` ring with XOR-swizzled rows, page 16/32/64, zero-filled tails | kept, default |
| v2b | v2a data path with one producer warp (`cp.async.mbarrier.arrive.noinc`) and full/empty mbarriers | correct, **no gain** (tie when DRAM bound, 8% slower when L2 resident): kept as an option (`FLASH_MLA_SM86_SPEC=1`) |
| v2c | v2a with QK(n+1) issued before softmax(n)/PV(n) and a split-phase mbarrier S exchange | correct, **no gain** (-3% L2-resident at BM16, +15% worse at BM32): kept as an option (`_PIPE=1`) |
| fp8 e4m3 / int8 | 8-bit KV tile stays 8-bit in smem; in-register conversion (`prmt`/`lop3`/`fma.bf16x2`, no `cvt.e4m3` on sm_86); per-token scale applied in fp32 | kept, bf16 and fp16 |
| combine | separate kernels: column-split "wide" (16/8/4 split-groups) for few rows, warp-per-row for many rows; optional fused last-arriver combine in the main kernel | separate kernels default; fused is off by default (no gain) |
| metadata | single-block device kernel, balanced cuts `floor(p*T/P)`, graph safe; also stores `seqlen_k` of each part's first segment | kept |

Default plan (`mla_sm86.h`): rows `M = s_q*h_q/h_k <= 16` -> BM=16 (4 warps), Nt=16, 3 stages; `M > 16` -> BM=32 (8 warps),
Nt=32, 2 stages (Nt=16/3 stages when the page has 16 tokens). 0 register spills in every instantiated variant
(`ptxas -v`: 185-224 regs/thread; the v2b BM32 variants spill because the producer warp caps the register budget at 168 and are
not instantiated). Quantized plan: Nt=32, 2 stages, BM = 16 or 32.

## 2. Findings, in order

1. **v2a first light** (b=128, s_k=4096, h_q=16, page 64): 832 us -> 734 us after the page-id look-ahead (KV 823 GB/s, 94%);
   cos_diff(out) ~ 1.8e-6, cos_diff(lse) ~ 1e-15 over the whole plan section 5 matrix.
2. **Config sweeps are flat at h_q=16.** Graph-mode per-layer us for BM16 plans {16/3, 16/4 pipe, 2 CTAs/SM x 16/2, 32/2, 16/4}
   differ by < 1.5% everywhere (b=1..128, s_k=1k..8k): once the stream is DRAM bound the tile configuration does not matter. Largest
   spread: b=1, s_k=1024: 10.5 (2 CTAs/SM) .. 11.3 us.
3. **Where the time goes at small batch** (globaltimer trace, `-DMLA_SM86_TRACE`; b=1, s_k=8192, 82 CTAs): median CTA reaches its first
   tile at **4.1 us** (metadata -> block table -> `cp.async` -> DRAM = three dependent round trips + launch), streams its ~100 tokens, and
   ends at 16-18 us; b=4: first tile 4.1 us, loop done at ~49 us (45 us at ~838 GB/s = near peak), kernel end 50-53 us.
   So once running the kernel is at the DRAM limit; the b<=4 gap to the ideal bytes/BW time is fixed overhead: ~4 us start-up, ~3 us tail
   (last tile + partial-O write + imbalance), and the combine kernel (1.5 us minimum for a trivial kernel, 5.3-5.9 us with
   the 20-82-way merge) plus launch gaps. nsys, b=1: decode 20.8 us + combine 5.7 us; fixed cost of a 1-tile decode kernel: 5.8 us.
4. **Scheduler**: `ceil(total/P)` payloads left the last parts empty (b=1, s_k=8192: 258 units over 82 parts -> payload 4 -> ~65 parts
   busy; one SM was active for 1,480 cycles of 36,950). Balanced cuts `floor(p*T/P)` fixed the distribution. Stored `seqlen_k` saves one
   dependent load at start-up. Overhead (`OV`) / minimum-blocks sweeps moved results by < 2%; defaults OV=2, no minimum.
   (The granule stays 32 tokens so `impl="legacy"` (BlockN=32) can consume the same metadata.)
5. **Combine kernels** (nsys, us): old 128-thread/row kernel 9.2 (b=1), 5.6 (b=4), 11.7 (b=32, 512 rows). New wide16 5.5 (b=1), wide8
   5.25 (b=2), wide4 5.9 (b=4; legacy 5.5), warp-per-row 7.0 (b=8), 5.4 (b=16), 5.2 (b=32), 6.5 (b=64). Policy by row count in
   `launch_combine`. Effect: h_q=16, b=64, s_k=4096 387 -> 374 us; b=32/s_k=1024 combine 11.7 -> 5.2 us. Remaining combine floor ~5 us
   = launch + `num_splits` load + LSE -> weights dependency chain.
6. **Fused last-arriver combine A/B** (counters self-reset, order-independent => deterministic; tests incl. graph replay): no measurable
   gain (b=128: 750.6 vs 743.4 us, noise) and a loss when few CTAs finish together (b=32, s_k=1024: 73.7 -> 81.9 us): the fence + atomic +
   serial merge of 96-640 KB of partials by one SM sits on the critical path of every split segment, while the unfused combine runs off
   it. The combine launch is still needed for rows with more splits than the fuse limit, so no launch is saved. Off by default
   (`FLASH_MLA_SM86_FUSE`=0), kept for experiments and tested.
7. **Why h_q=16 bf16 is at parity with legacy**: both kernels reach 91-99% of the DRAM peak for b >= 16 (SA's BlockM=16 legacy fix); the
   headroom was only at small batch, where fixed costs dominate (finding 3). A cooperative in-kernel combine (global barrier + distributed
   merge) was considered and rejected: ~3 us gain at best for b <= 2, needs a cooperative launch and scales badly with the number of rows.
8. **Compute side (ncu, L2-resident data, BM16, one CTA/SM)**: 36 TFLOPS-equivalent (1.28 us per 16-token tile per SM); tensor pipe active
   26% (= ~52% of the fp32-accumulate ceiling); issue slots 23%; CPI 4.4 = selected 1.0 + math-pipe throttle 1.48 + fixed-latency wait 0.95 +
   short scoreboard 0.34 + long scoreboard 0.20 + barrier 0.16. Two CTAs per SM (two warps per scheduler): 133 vs 163 us for the same
   work (44 TFLOPS-eq., tensor pipe 35% active), but it doubles the number of splits and does not pay off on real DRAM-bound shapes.
   v2c (software pipelining) moves 163 -> 157 us only: the SASS shows ptxas already interleaves HMMA/MUFU/LDSM inside the loop, the
   remaining stalls are dependency waits that only more warps can hide.
9. **BM32 with 32-token tiles and 2 stages** (h_q >= 32, `M > 16`): graph mode, s_k=4096: h_q=32 b=64 438 -> 386 us, h_q=64 b=64 816 -> 716
   us, h_q=128 b=64 1602 -> 1394 us (**-13%**; 44 -> 52 TFLOPS). Fewer barriers/exchanges per token and four independent QK
   accumulators instead of two. 3 stages of 32 tokens would need 110 KB (> 99 KB), v2c needs >= 3 stages, so this combination is the
   maximum that fits.
10. **fp8 / int8** (h_q=16, graph of 8 layers x own cache, ~3 GB of bf16 caches): see table below. ncu b=128, s_k=4096 (before the Nt=32
    plan): fp8 DRAM 82% of spec peak vs 94% for bf16; no memory stalls (long scoreboard 0.12), ALU pipe 28%, tensor 26%, 717 instr/tile vs
    475 for bf16: the 8-bit path is limited by per-tile latency/issue, not by bytes; Nt=32/2 stages (fewer syncs) gave +3-4%. At
    h_q >= 64 (compute bound) 8-bit KV only ties bf16 (0.95-1.07x), as expected (the saving is bytes, not FLOPs).
11. **Sanitizers** (`compute-sanitizer`, small shapes, all variants and both quantized kernels): memcheck, synccheck, initcheck: 0 errors;
    racecheck: 0 hazards for v2a/v2c/fused-combine/fp8/int8. Found and fixed: lane-divergent `mbarrier.init` in v2c (synccheck "missing
    init" at BM32; one initialising thread now). v2b: synccheck reports a benign "missing wait" on the `empty` barriers (last arrivals
    before kernel exit are never waited for) and racecheck, which does not model mbarrier, flags the ring; v2b is not the default.

## 3. Results with the final defaults

Targets (plan section 5): h_q=16, s_q=1, b=64-128, s_k=4096-8192 >= 80% of measured DRAM bandwidth -> **93-99%**. h_q=128 >= 60% of
bf16 peak -> **71-76% (49.8-53.3 TFLOPS)** for b >= 4. Latency b=1..4 <= 1.3 x (KV bytes / BW) + 5 us: met when the KV is >= ~35 MB
(b=4 s_k=8192: 57.5 vs 61.1 us; b=1 s_k=32768: 57.6 vs 61.1; s_k=65536: 100.9 vs 117), **missed for smaller KV** (b=1: 9.9 us vs 6.8 at
s_k=1024; 16.3 vs 12.0 at 4096; 22.8 vs 19.0 at 8192; b=4 s_k=4096: 36.0 vs 33.1): the fixed ~10-12 us of start-up + tail + combine +
launch is above the 5 us budget (finding 3).
fp8/int8 >= 1.6x over bf16 for the memory-bound case -> **1.73-1.75x** (b >= 32), 1.64x (b=16).

### v2 vs legacy (page 32, bf16, s_q=1; CUDA-graph = per-layer time of an 8-layer graph, each layer with its own KV cache)

| case (h_q, b, s_k) | legacy graph us | v2 graph us | speedup | v2 eager us (legacy) | v2 GB/s (% of 874) | v2 TFLOPS (% of 70) |
|---|---|---|---|---|---|---|
| h16 b1 sk1024 | 10.8 | 9.9 | 1.09x | 15.4 (15.2) | 123 (14%) | 3.6 (5%) |
| h16 b1 sk4096 | 21.2 | 16.3 | 1.30x | 20.5 (25.6) | 292 (33%) | 8.8 (13%) |
| h16 b1 sk8192 | 28.0 | 22.8 | 1.23x | 27.6 (31.7) | 415 (47%) | 12.5 (18%) |
| h16 b4 sk1024 | 16.0 | 15.9 | 1.01x | 20.5 (20.5) | 306 (35%) | 9.0 (13%) |
| h16 b4 sk4096 | 35.8 | 36.0 | 0.99x | 39.9 (39.9) | 529 (60%) | 15.9 (23%) |
| h16 b4 sk8192 | 57.7 | 57.5 | 1.00x | 61.4 (62.5) | 658 (75%) | 19.8 (28%) |
| h16 b16 sk1024 | 35.8 | 36.0 | 0.99x | 39.9 (39.9) | 540 (62%) | 15.9 (23%) |
| h16 b16 sk4096 | 102.1 | 103.3 | 0.99x | 107.5 (105.5) | 736 (84%) | 22.1 (32%) |
| h16 b16 sk8192 | 189.6 | 189.9 | 1.00x | 193.5 (193.5) | 798 (91%) | 24.0 (34%) |
| h16 b64 sk1024 | 109.3 | 108.4 | 1.01x | 112.6 (112.5) | 717 (82%) | 21.1 (30%) |
| h16 b64 sk4096 | 369.9 | 370.9 | 1.00x | 373.8 (375.7) | 820 (94%) | 24.6 (35%) |
| h16 b64 sk8192 | 713.0 | 711.6 | 1.00x | 716.8 (715.8) | 852 (97%) | 25.6 (37%) |
| h16 b128 sk1024 | 203.6 | 203.4 | 1.00x | 204.8 (205.8) | 764 (87%) | 22.4 (32%) |
| h16 b128 sk4096 | 726.0 | 719.7 | 1.01x | 722.9 (729.1) | 845 (97%) | 25.4 (36%) |
| h16 b128 sk8192 | 1405.6 | 1403.3 | 1.00x | 1407.0 (1409.0) | 864 (99%) | 26.0 (37%) |
| h128 b1 sk1024 | 21.0 | 20.6 | 1.02x | 24.6 (24.6) | 71 (8%) | 13.8 (20%) |
| h128 b1 sk4096 | 42.5 | 36.7 | 1.16x | 39.9 (45.1) | 136 (16%) | 31.1 (44%) |
| h128 b1 sk8192 | 67.3 | 56.7 | 1.19x | 58.4 (67.6) | 171 (20%) | 40.2 (57%) |
| h128 b4 sk1024 | 42.2 | 33.8 | 1.25x | 36.9 (45.1) | 173 (20%) | 33.8 (48%) |
| h128 b4 sk4096 | 118.5 | 96.4 | 1.23x | 94.2 (114.7) | 207 (24%) | 47.3 (68%) |
| h128 b4 sk8192 | 223.0 | 181.9 | 1.23x | 175.1 (214.0) | 214 (24%) | 50.2 (72%) |
| h128 b16 sk1024 | 134.9 | 110.0 | 1.23x | 108.5 (132.1) | 212 (24%) | 41.5 (59%) |
| h128 b16 sk4096 | 443.0 | 366.1 | 1.21x | 349.2 (428.0) | 218 (25%) | 49.9 (71%) |
| h128 b16 sk8192 | 860.5 | 706.6 | 1.22x | 686.1 (840.7) | 220 (25%) | 51.7 (74%) |
| h128 b64 sk1024 | 459.9 | 382.7 | 1.20x | 367.6 (445.4) | 244 (28%) | 47.7 (68%) |
| h128 b64 sk4096 | 1716.7 | 1410.8 | 1.22x | 1382.4 (1679.4) | 227 (26%) | 51.8 (74%) |
| h128 b64 sk8192 | 3385.1 | 2776.8 | 1.22x | 2769.9 (3366.9) | 224 (26%) | 52.6 (75%) |
| h128 b128 sk1024 | 901.4 | 746.0 | 1.21x | 718.8 (879.6) | 250 (29%) | 48.9 (70%) |
| h128 b128 sk4096 | 3404.2 | 2787.4 | 1.22x | 2789.4 (3380.2) | 230 (26%) | 52.4 (75%) |
| h128 b128 sk8192 | 6854.9 | 5620.7 | 1.22x | 5561.3 (6731.8) | 221 (25%) | 52.0 (74%) |

### v2, page 64 (vLLM page size; legacy cannot run it)

| case | graph us | eager us | GB/s (%) | TFLOPS (%) |
|---|---|---|---|---|
| h16 b1 sk1024 | 9.9 | 15.4 | 123 (14%) | 3.6 (5%) |
| h16 b1 sk4096 | 16.3 | 20.5 | 292 (33%) | 8.8 (13%) |
| h16 b1 sk8192 | 23.2 | 27.6 | 409 (47%) | 12.3 (18%) |
| h16 b4 sk1024 | 16.0 | 20.5 | 304 (35%) | 8.9 (13%) |
| h16 b4 sk4096 | 36.4 | 41.0 | 523 (60%) | 15.7 (22%) |
| h16 b4 sk8192 | 57.7 | 62.5 | 656 (75%) | 19.8 (28%) |
| h16 b16 sk1024 | 36.4 | 41.0 | 534 (61%) | 15.7 (22%) |
| h16 b16 sk4096 | 104.1 | 107.5 | 731 (84%) | 21.9 (31%) |
| h16 b16 sk8192 | 190.2 | 194.6 | 797 (91%) | 24.0 (34%) |
| h16 b64 sk1024 | 108.8 | 113.6 | 714 (82%) | 21.0 (30%) |
| h16 b64 sk4096 | 370.7 | 373.8 | 821 (94%) | 24.6 (35%) |
| h16 b64 sk8192 | 712.2 | 717.8 | 851 (97%) | 25.6 (37%) |
| h16 b128 sk1024 | 201.6 | 204.8 | 771 (88%) | 22.6 (32%) |
| h16 b128 sk4096 | 720.1 | 724.0 | 845 (97%) | 25.4 (36%) |
| h16 b128 sk8192 | 1401.3 | 1404.9 | 865 (99%) | 26.1 (37%) |
| h128 b1 sk1024 | 20.9 | 24.6 | 70 (8%) | 13.7 (20%) |
| h128 b1 sk4096 | 37.5 | 41.0 | 133 (15%) | 30.4 (43%) |
| h128 b1 sk8192 | 57.6 | 59.4 | 169 (19%) | 39.6 (57%) |
| h128 b4 sk1024 | 34.3 | 36.9 | 170 (19%) | 33.3 (48%) |
| h128 b4 sk4096 | 97.0 | 95.2 | 206 (24%) | 47.0 (67%) |
| h128 b4 sk8192 | 183.2 | 176.1 | 212 (24%) | 49.8 (71%) |
| h128 b16 sk1024 | 110.1 | 108.5 | 212 (24%) | 41.5 (59%) |
| h128 b16 sk4096 | 365.2 | 352.3 | 219 (25%) | 50.0 (71%) |
| h128 b16 sk8192 | 715.9 | 688.1 | 217 (25%) | 51.0 (73%) |
| h128 b64 sk1024 | 384.6 | 369.7 | 243 (28%) | 47.5 (68%) |
| h128 b64 sk4096 | 1418.5 | 1393.7 | 226 (26%) | 51.5 (74%) |
| h128 b64 sk8192 | 2798.3 | 2774.0 | 222 (25%) | 52.2 (75%) |
| h128 b128 sk1024 | 751.7 | 730.1 | 248 (28%) | 48.6 (69%) |
| h128 b128 sk4096 | 2826.4 | 2806.8 | 226 (26%) | 51.7 (74%) |
| h128 b128 sk8192 | 5656.8 | 5560.3 | 220 (25%) | 51.6 (74%) |

### v2: long context, b=1 (many splits)

| case | mode | us | GB/s (%) | TFLOPS (%) | cos_diff out/lse |
|---|---|---|---|---|---|
| h16_sq1_b1_sk16384_p64 | eager | 43.0 | 440 (50%) | 13.3 (19%) | 1.4e-06/2.4e-15 |
| h16_sq1_b1_sk16384_p64 | graph | 36.4 | 520 (60%) | 15.7 (22%) | 1.4e-06/2.4e-15 |
| h16_sq1_b1_sk32768_p64 | eager | 61.4 | 615 (70%) | 18.6 (27%) | 1.4e-06/1.0e-15 |
| h16_sq1_b1_sk32768_p64 | graph | 57.6 | 656 (75%) | 19.8 (28%) | 1.4e-06/1.0e-15 |
| h16_sq1_b1_sk65536_p64 | eager | 104.4 | 723 (83%) | 21.9 (31%) | 1.5e-06/4.4e-16 |
| h16_sq1_b1_sk65536_p64 | graph | 100.9 | 749 (86%) | 22.6 (32%) | 1.5e-06/4.4e-16 |
| h128_sq1_b1_sk16384_p64 | eager | 99.3 | 193 (22%) | 45.9 (66%) | 1.5e-06/2.0e-15 |
| h128_sq1_b1_sk16384_p64 | graph | 98.6 | 194 (22%) | 46.3 (66%) | 1.5e-06/2.0e-15 |
| h128_sq1_b1_sk32768_p64 | eager | 179.2 | 212 (24%) | 50.9 (73%) | 1.4e-06/1.3e-15 |
| h128_sq1_b1_sk32768_p64 | graph | 182.8 | 208 (24%) | 49.9 (71%) | 1.4e-06/1.3e-15 |
| h128_sq1_b1_sk65536_p64 | eager | 343.0 | 221 (25%) | 53.2 (76%) | 1.4e-06/1.9e-15 |
| h128_sq1_b1_sk65536_p64 | graph | 355.1 | 213 (24%) | 51.4 (73%) | 1.4e-06/1.9e-15 |

### v2: MTP s_q=2 causal

| case | mode | us | GB/s (%) | TFLOPS (%) | cos_diff out/lse |
|---|---|---|---|---|---|
| h16_sq2_b1_sk4096_p64 | eager | 30.7 | 156 (18%) | 9.3 (13%) | 1.5e-06/2.6e-15 |
| h16_sq2_b1_sk4096_p64 | graph | 22.8 | 210 (24%) | 12.5 (18%) | 1.5e-06/2.6e-15 |
| h16_sq2_b16_sk4096_p64 | eager | 120.8 | 634 (73%) | 37.8 (54%) | 1.7e-06/2.3e-15 |
| h16_sq2_b16_sk4096_p64 | graph | 116.2 | 659 (75%) | 39.3 (56%) | 1.7e-06/2.3e-15 |
| h16_sq2_b64_sk4096_p64 | eager | 396.3 | 773 (88%) | 46.1 (66%) | 1.7e-06/1.8e-15 |
| h16_sq2_b64_sk4096_p64 | graph | 389.9 | 786 (90%) | 46.8 (67%) | 1.7e-06/1.8e-15 |
| h128_sq2_b1_sk4096_p64 | eager | 58.4 | 90 (10%) | 39.1 (56%) | 1.6e-06/1.8e-15 |
| h128_sq2_b1_sk4096_p64 | graph | 55.3 | 95 (11%) | 41.3 (59%) | 1.6e-06/1.8e-15 |
| h128_sq2_b16_sk4096_p64 | eager | 689.2 | 122 (14%) | 53.0 (76%) | 1.7e-06/1.8e-15 |
| h128_sq2_b16_sk4096_p64 | graph | 704.8 | 120 (14%) | 51.8 (74%) | 1.7e-06/1.8e-15 |
| h128_sq2_b64_sk4096_p64 | eager | 2741.2 | 123 (14%) | 53.3 (76%) | 1.7e-06/2.6e-15 |
| h128_sq2_b64_sk4096_p64 | graph | 2764.2 | 122 (14%) | 52.8 (75%) | 1.7e-06/2.6e-15 |

### v2: h_q = 32 / 64 (eager)

| case | mode | us | GB/s (%) | TFLOPS (%) | cos_diff out/lse |
|---|---|---|---|---|---|
| h32_sq1_b1_sk1024_p64 | eager | 18.4 | 68 (8%) | 3.9 (6%) | 1.8e-06/1.7e-15 |
| h32_sq1_b1_sk4096_p64 | eager | 25.6 | 187 (21%) | 11.1 (16%) | 1.5e-06/2.0e-15 |
| h32_sq1_b16_sk1024_p64 | eager | 46.1 | 434 (50%) | 24.8 (35%) | 1.9e-06/1.3e-15 |
| h32_sq1_b16_sk4096_p64 | eager | 117.8 | 651 (74%) | 38.8 (55%) | 1.7e-06/2.0e-15 |
| h32_sq1_b64_sk1024_p64 | eager | 130.0 | 615 (70%) | 35.1 (50%) | 2.0e-06/1.7e-15 |
| h32_sq1_b64_sk4096_p64 | eager | 395.3 | 775 (89%) | 46.2 (66%) | 1.7e-06/2.3e-15 |
| h64_sq1_b1_sk1024_p64 | eager | 20.5 | 64 (7%) | 7.0 (10%) | 1.8e-06/1.3e-15 |
| h64_sq1_b1_sk4096_p64 | eager | 29.7 | 164 (19%) | 19.2 (27%) | 1.6e-06/2.3e-15 |
| h64_sq1_b16_sk1024_p64 | eager | 61.4 | 344 (39%) | 37.1 (53%) | 2.0e-06/1.7e-15 |
| h64_sq1_b16_sk4096_p64 | eager | 179.2 | 434 (50%) | 50.9 (73%) | 1.7e-06/2.0e-15 |
| h64_sq1_b64_sk1024_p64 | eager | 194.6 | 434 (50%) | 46.9 (67%) | 2.1e-06/1.3e-15 |
| h64_sq1_b64_sk4096_p64 | eager | 703.5 | 442 (51%) | 51.9 (74%) | 1.7e-06/2.3e-15 |

### v2: fp16

| case | mode | us | GB/s (%) | TFLOPS (%) | cos_diff out/lse |
|---|---|---|---|---|---|
| h16_sq1_b1_sk4096_p64 | eager | 22.5 | 211 (24%) | 6.3 (10%) | 2.4e-08/2.9e-15 |
| h16_sq1_b1_sk4096_p64 | graph | 16.6 | 286 (33%) | 8.6 (13%) | 2.4e-08/2.9e-15 |
| h16_sq1_b64_sk4096_p64 | eager | 373.8 | 814 (93%) | 24.4 (37%) | 2.6e-08/3.3e-15 |
| h16_sq1_b64_sk4096_p64 | graph | 370.2 | 822 (94%) | 24.7 (37%) | 2.6e-08/3.3e-15 |
| h128_sq1_b1_sk4096_p64 | eager | 39.9 | 125 (14%) | 28.6 (43%) | 2.5e-08/3.3e-15 |
| h128_sq1_b1_sk4096_p64 | graph | 37.0 | 135 (15%) | 30.8 (46%) | 2.5e-08/3.3e-15 |
| h128_sq1_b64_sk4096_p64 | eager | 1412.1 | 226 (26%) | 51.7 (78%) | 2.7e-08/3.1e-15 |
| h128_sq1_b64_sk4096_p64 | graph | 1424.1 | 225 (26%) | 51.3 (77%) | 2.7e-08/3.1e-15 |

### v2: varlen (normal(mean, mean/2) lengths, eager)

| case | mode | us | GB/s (%) | TFLOPS (%) | cos_diff out/lse |
|---|---|---|---|---|---|
| h16_sq1_b64_sk4096_p64_var | eager | 366.6 | 815 (93%) | 24.4 (35%) | 2.0e-06/1.7e-15 |
| h16_sq2_b64_sk4096_p64_var | eager | 388.1 | 775 (89%) | 46.2 (66%) | 2.1e-06/1.8e-15 |
| h64_sq1_b64_sk4096_p64_var | eager | 686.1 | 445 (51%) | 52.2 (75%) | 2.0e-06/2.1e-15 |
| h64_sq2_b64_sk4096_p64_var | eager | 1367.0 | 230 (26%) | 52.4 (75%) | 2.0e-06/1.8e-15 |
| h128_sq1_b64_sk4096_p64_var | eager | 1364.0 | 230 (26%) | 52.5 (75%) | 2.0e-06/2.3e-15 |
| h128_sq2_b64_sk4096_p64_var | eager | 2698.2 | 123 (14%) | 53.1 (76%) | 2.0e-06/2.1e-15 |


### fp8 / int8 vs bf16 (h_q=16, s_q=1, page 64, graph of N layers with own caches; `python sk_bench_quant.py`)

```
shape                  kind           us      GB/s    TFLOPS   speedup
h16 b128 sk4096        bf16        772.6     781.7      23.6     1.00x
h16 b128 sk4096        fp8         447.2     755.0      40.8     1.73x
h16 b128 sk4096        int8        447.2     755.0      40.8     1.73x
h16 b64 sk8192         bf16        767.5     787.0      23.8     1.00x
h16 b64 sk8192         fp8         442.6     762.8      41.2     1.73x
h16 b64 sk8192         int8        442.6     762.8      41.2     1.73x
h16 b32 sk4096         bf16        203.3     742.9      22.5     1.00x
h16 b32 sk4096         fp8         116.3     725.9      39.2     1.75x
h16 b32 sk4096         int8        116.5     724.8      39.2     1.75x
h16 b16 sk4096         bf16        109.6     689.0      20.8     1.00x
h16 b16 sk4096         fp8          66.9     630.5      34.1     1.64x
h16 b16 sk4096         int8         66.9     630.5      34.1     1.64x
h16 b8 sk4096          bf16         59.3     636.3      19.2     1.00x
h16 b8 sk4096          fp8          42.0     501.9      27.1     1.41x
h16 b8 sk4096          int8         42.4     498.1      26.9     1.40x
h16 b4 sk4096          bf16         35.6     529.5      16.0     1.00x
h16 b4 sk4096          fp8          25.8     409.1      22.1     1.38x
h16 b4 sk4096          int8         25.9     407.1      22.0     1.38x
h16 b2 sk8192          bf16         35.4     532.6      16.1     1.00x
h16 b2 sk8192          fp8          25.6     412.3      22.3     1.38x
h16 b2 sk8192          int8         25.3     417.4      22.6     1.40x
h16 b1 sk8192          bf16         23.0     409.6      12.4     1.00x
h16 b1 sk8192          fp8          18.1     291.3      15.7     1.27x
h16 b1 sk8192          int8         18.1     291.3      15.7     1.27x
```

### Config sweep (BM16, graph mode us per layer, h_q=16, page 64)

```
shape                         16/3 (default)   16/4 pipe     2 CTAs/SM x 16/2   16/32 x2        16/4
h16_sq1_b1_sk1024_p64               11.0           11.3           10.5             10.9           11.1
h16_sq1_b1_sk8192_p64               23.2           24.1           24.8             23.2           24.3
h16_sq1_b4_sk4096_p64               37.1           38.0           38.9             36.7           38.0
h16_sq1_b16_sk4096_p64             103.9          105.9          104.4            103.6          107.3
h16_sq1_b64_sk4096_p64             370.0          370.3          371.3            369.0          371.3
h16_sq1_b128_sk4096_p64            720.3          720.4          727.3            719.0          721.3
h16_sq1_b128_sk8192_p64           1400.5         1400.8         1408.7           1399.0         1401.2
```

## 4. Not done / negative results

* **Swap-AB for h_q=8**: not implemented; h_q=8 (M=8) is deeply memory bound (half-empty m16 tiles are irrelevant there).
* **Fused combine**: implemented and tested (deterministic, counters self-reset, graph replay) but disabled: no gain (finding 6).
* **v2b / v2c**: implemented, tested, sanitizer-checked where applicable, not faster than v2a (findings 8, 9).
* **b <= 4 latency target** missed for KV < ~35 MB (fixed ~10-12 us overhead, finding 3). Not attempted: cooperative/in-kernel
  combine (needs cooperative launch; ~3 us at b <= 2) and PDL-style overlap of the kernel start-up (sm_90 only).
* 8-bit KV gives no speed-up for compute-bound shapes (h_q >= 64); use bf16 there.
* The quantized kernels support page 16/32/64, Nt=16/32, BM16/32, bf16/fp16 queries; `impl="legacy"` has no 8-bit path.
