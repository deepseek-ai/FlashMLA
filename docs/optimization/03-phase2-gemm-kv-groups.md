# Phase 2: the layer projections and the 8-bit kernel

After phase 1 the attention kernel is at the hardware ceiling of its regime, and the cost moves elsewhere. Phase 2
started with three profiler questions and turned two of the answers into code. Results are in
[01-results.md](01-results.md); panel A of Figure 3 shows the GEMM.

![Figure 3](figures/fig3_gemm_and_l2_plan.png)

## 1. Measure first

Nsight Compute, full metric set, a build with line information so that stalls can be attributed to source lines.

| Question | Finding | Consequence |
|---|---|---|
| Attention is only 42% of the fused layer at b=8. What are the four GEMMs doing? | At 1–64 tokens cuBLAS opens 32–64 CTAs for 82 SMs. DRAM throughput: qkv_a 61–63%, W_UK 19–39%, W_UV 22–38%, o_proj 51–63% of the peak. The four GEMMs plus a reshape copy are 50% of the layer's busy time. | Write a GEMM for this shape (section 2) |
| The 8-bit kernel stalls on math-pipe throttle (32%) at only 86.6% DRAM. Is it short of ALU capacity? | No. 96% of the math-pipe throttle sits on the tensor instruction; the ALU pipe is at 27–34%. There is one warp per scheduler, so nothing hides the latency of the tensor instruction. | Add warps, do not remove dequantization instructions (section 3) |
| Where does the h_q=128 kernel (BM32) stall? | Cycles per instruction 9.68: math-pipe throttle 4.71, barrier 1.30 (the per-tile CTA barrier is 8.5% of all samples, the S exchange inside a group 4.9%), wait 1.07. The tensor pipe is at 38.2%, which is 76% of what FP32 accumulation allows. | Try to recover the barrier time (section 4); it did not pay |

One correction came out of this: `--clock-control base` does not pin the SM clock on this card (1.67–1.96 GHz depending
on the kernel). Comparisons across profiler reports are therefore made in cycles. In cycles the v2 kernel at h_q=128 is
1.21× faster than legacy (2.35 M vs 2.85 M), not the 1.10× that the wall times of the two reports suggest.

## 2. The weight-streaming skinny GEMM

Source: `csrc/fused/skinny_gemm.cu`, `csrc/fused/skinny_gemm.h`, `flash_mla/skinny_gemm.py`; used by
`flash_mla/mla_decode_layer.py` with `gemm="auto"` (the default).

### 2.1 Why cuBLAS is the wrong tool here

A decode layer multiplies T = 1…64 tokens by 2–15 MB of weights:

| Projection | Weight | Bytes |
|---|---|---|
| qkv_a (q_proj and kv_a_proj as one GEMM) | 3648 × 2048 | 14.9 MB |
| W_UK (absorption, 16 heads) | 16 × [512 × 128] | 2.1 MB |
| W_UV (16 heads) | 16 × [128 × 512] | 2.1 MB |
| o_proj | 2048 × 2048 | 8.4 MB |

Every weight byte is used T times, far below the 80 FLOP/B ridge of the card. These are weight-streaming problems: the
only thing that matters is how many weight bytes are in flight on every SM. cuBLAS picks kernels tuned for large M and
runs too few CTAs, each with little in flight.

### 2.2 Design

`y[t, g, n] = Σ_k x[t, g, k] · w[g, n, k]`, bf16 or fp16 in, fp32 accumulation, one rounding on output.

- **Grid** `(N / n_tile, splits, G)`: one CTA owns `n_tile` weight rows of one head, all T tokens and a contiguous range
  of K. 128 threads.
- **Stream.** K is consumed in chunks of 64 elements through a `cp.async` ring of 3–8 stages, sized to keep about 48 KB
  in flight per CTA. Weight rows and token rows are both 128-byte shared-memory rows, XOR-swizzled by row.
- **Swapped operands.** The weight tile is the A operand of `mma.sync m16n8k16` (16 weight rows per m-tile), the tokens
  are the B operand (8 per n-tile). One to eight tokens waste nothing beyond the n8 granule.
- **Split-K inside the CTA.** Warp `w` takes `k16` step `w` of every chunk, so no fragment is loaded twice and no warp
  waits for another inside a chunk.
- **Deterministic epilogue.** The four warp partials are summed in warp order through shared memory and rounded once.
  An optional split-K across CTAs stores fp32 partials and lets the last-arriving CTA of a tile add them in split
  order; its counters return to zero, so graph replay is safe.
- **Strided views.** x and y may be strided: W_UK reads q_nope straight out of the qkv_a output, and W_UV writes the
  `[T, 16 × 128]` input of o_proj in place. A 2.3 µs reshape copy disappears, and with it one kernel per layer (8 → 7).

### 2.3 Rules found by sweeping

| Rule | Reason |
|---|---|
| `splits = 1` | split-K across CTAs never paid at these sizes |
| `n_tile = 32`; 16 for the per-head matrices at T ≤ 16 | keeps 64–512 CTAs streaming on 82 SMs |
| ring of 32, 48 or 64 KB | within 2% of each other; 48 KB chosen |
| single-matrix projections (qkv_a, o_proj) go back to cuBLAS above 32 tokens | cuBLAS catches up as T grows; the per-head matrices stay on the skinny kernel up to 64 tokens |

CTA counts at 8 tokens: qkv_a 57 → 114, W_UK 256 small → 512, W_UV 32 → 128, o_proj 64 → 64 with a deeper ring.

### 2.4 Result

- The streaming phase runs at about 870 GB/s. What remains is roughly 2.6 µs of fixed cost per launch.
- The four GEMMs at b=8: 61.7 → 41.5 µs (the byte floor is 31.5 µs).
- The layer at b=8, s_k=4096: 115.4 → 102.0 µs with bf16 KV (1.13×), 100.4 → 86.1 µs with FP8 KV (1.17×); at s_k=1024
  1.19× and 1.21×. At b=64 the gain is 1.01–1.05×, because attention dominates there.
- Correctness: against fp32 `F.linear` / `bmm`, run-to-run bit equality, and the layer tests with both GEMM back ends.

## 3. Two KV groups per CTA for the 8-bit kernel

Source: `csrc/sm86/mla_decode_sm86_q.cuh`, `csrc/flash_fwd_mla_metadata.cu`.

### 3.1 Design

The 8-bit kernel for M ≤ 16 gets a second group of 4 warps. Group `g` processes tiles `g, g + 2, g + 4, …` of the part.
Each group has its own 2-stage ring, its own S exchange buffer and its own named barrier; there is no per-tile barrier
across the two groups. At the end of the part the two groups' `(m, l, O)` are merged through shared memory in a fixed
order (`O = O₀ · e^(m₀ − m) + O₁ · e^(m₁ − m)`) and group 0 writes the result.

The part count stays at 82, so the partial and combine traffic does not change. Every scheduler now has two warps: while
one is in a tensor instruction the other can dequantize. 254–255 registers per thread, no spills, one CTA per SM.

### 3.2 It only pays on long parts

Each part pays a fixed cost for loading Q twice and for the merge. With two groups always on, the grid is 0.84–1.10× of
the old default: short contexts get slower.

### 3.3 Choosing per part is wrong

The first attempt chose one or two groups per part from the part's own tile count. A launch that mixed both kinds was
much slower than either uniform choice: b=32, s_k=4096 took 151 µs against 122 µs for the old default. The profiler
showed the slowest SM at 248 k cycles against 199 k.

The reason: a one-group CTA has one tile in flight, a two-group CTA has two, and DRAM bandwidth is shared in proportion
to the bytes in flight. The one-group CTAs starve, and they decide when the kernel ends.

**Rule: a choice that changes the bytes in flight per CTA has to be the same for the whole launch.**

### 3.4 One choice per launch, without a host sync

The metadata kernel already computes the total load and the part count. It now writes the load per part
(`total / parts`, in 32-token units, including the fixed overhead per request, identical for all parts) into the sixth
int of every metadata row. Each CTA compares it with a threshold (`FLASH_MLA_SM86_KGMIN`, default 64) and all CTAs reach
the same decision. Nothing is read back by the host, so the launch stays graph-safe.

### 3.5 Result

- Against the phase-1 default over the grid b = 16…128 × s_k = 1024…16384: FP8 0.993–1.115×, INT8 0.992–1.072×. Long
  contexts gain 4–11%; nothing at s_k ≤ 2048 is slower than 0.999×.
- Against bf16 under a graph at b≥32, s_k≥4096: FP8 1.55–1.70×, INT8 1.54–1.60×.
- In the layer with FP8 KV: b=64, s_k=8192 523.3 → 487.3 µs (1.07×).
- `FLASH_MLA_SM86_QCFG=16,2,1,32` restores the single-group plan; `FLASH_MLA_SM86_KGMIN=0` forces two groups.

## 4. Tried in phase 2 and not kept

The three patches are in [research/phase2_rejected/](../../research/README.md) and apply to this branch.

| Idea | Result | Why |
|---|---|---|
| Combine reads all splits in one round (96 per round instead of 64, so the 82 splits of b=1 need one round) | 4.28 → 4.26 µs, bit-identical | the second round already overlapped the first; the chain from LSE to the weights is the bottleneck |
| Softmax skips the rescale of O when the row maximum did not change | decode 3–5% slower (h_q=128: 54.1 → 52.7 TFLOPS), bit-identical | the branch moves 64 multiplies into their own basic block, where they can no longer be interleaved with tensor instructions |
| h_q=128: the two M groups run half a tile out of phase (ping-pong), no per-tile CTA barrier | 0.96–0.98×, bit-identical | barrier stall 1.30 → 1.08, but 5% more instructions and the same math-pipe throttle; the kernel is bound by the FP32-accumulate tensor pipe, and shifting phases adds no throughput |

The target of 55 TFLOPS at h_q=128 was not reached; the default stays at 53–54 TFLOPS (eager, b≥64, s_k≥4096).

## 5. The layer after phase 2

27-layer graph, µs per layer, s_k=4096:

| b | bf16 KV | FP8 KV | Phase 1 bf16 / FP8 |
|---|---|---|---|
| 1 | 61.5 | 61.6 | 65.8 / 66.1 |
| 8 | 102.3 | 86.3 | 117.9 / 102.3 |
| 32 | 239.5 | 172.0 | 259.3 / 194.3 |
| 64 | 431.4 | 291.2 | 440.0 / 310.5 |

At b=1 the layer now moves 32 MB in 61.5 µs. At peak bandwidth those bytes take 37 µs. The remaining 27 µs are not inside
any single kernel's inner loop; they are between the kernels. That is phase 3.
