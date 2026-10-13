# FlashMLA for Ampere (RTX 3090, sm_86)

MLA decode for DeepSeek-style models on GeForce Ampere GPUs: a native sm_86 attention kernel, a fused decode layer that
runs as one CUDA graph, weight-streaming GEMMs for the layer projections and an optional 8-bit KV cache.

This branch builds on the Ampere port [pzhao-eng/FlashMLA](https://github.com/pzhao-eng/FlashMLA) (branch `a100_replace`),
which is itself adapted from [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) (Hopper). The original Ampere
port cannot launch on a GeForce Ampere card: its tile needs 113,280 B of shared memory and sm_86 allows 101,376 B per
block. Everything here was developed and measured on RTX 3090 cards with nvcc 12.4 and PyTorch 2.7.1 (cu126), in CUDA C++
only.

| | |
|---|---|
| Status | Phases 1 and 2 are implemented, tested (760 tests) and measured. Phase 3 is a plan backed by prototype measurements: the prototype harnesses are in [research/phase3/](research/README.md), the implementation in the library is not written yet. |
| Full write-up | [docs/optimization/](docs/optimization/README.md): results, design of every optimization, the phase-3 plan, and what did not work |
| Original working report | [docs/ampere_opt_report.md](docs/ampere_opt_report.md) (Chinese), with the raw logs in `docs/perf_log_*.md` and `docs/bench_matrix.md` |

![One decode layer: kernel pipeline and optimization map](docs/optimization/figures/fig1_layer_pipeline.png)

## Contents

- [Results](#results)
- [What was optimized](#what-was-optimized)
  - [Phase 1: decode kernel, graph-safe API, fused layer, 8-bit KV](#phase-1-decode-kernel-graph-safe-api-fused-layer-8-bit-kv)
  - [Phase 2: skinny GEMM and two KV groups](#phase-2-skinny-gemm-and-two-kv-groups)
  - [Phase 3 (planned): what happens between the kernels](#phase-3-planned-what-happens-between-the-kernels)
  - [What did not work](#what-did-not-work)
- [Build and test](#build-and-test)
- [Usage](#usage)
- [Tunables](#tunables)
- [Benchmarks and profiling](#benchmarks-and-profiling)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [Acknowledgement and citation](#acknowledgement-and-citation)

## Results

All numbers are from one RTX 3090 that was otherwise idle (82 SMs, read-only bandwidth peak 873.8 GB/s, bf16 tensor peak
70.0 TFLOPS with FP32 accumulation). "Graph" means a CUDA graph of 27 layers in which every layer has its own KV cache and
its own weights, reported per layer. `b` is the batch size, `s_k` the number of cached tokens per request, `h_q` the number
of query heads. The complete tables are in [docs/optimization/01-results.md](docs/optimization/01-results.md).

![Results](docs/optimization/figures/fig4_results.png)

### Baseline to optimized, at a glance

| Workload | Baseline | This branch | Gain |
|---|---|---|---|
| Decode kernel, h_q=16, b≥64, s_k≥4096 | original fork: does not launch | 821–864 GB/s, 94–99% of the read peak | runs, at the hardware limit |
| Decode kernel, h_q=128, b≥16, s_k≥4096 | smem-fixed fork ("legacy"): 41–44 TFLOPS | 50.8–52.6 TFLOPS, 73–75% of the bf16 peak | 1.15–1.24× |
| Decode kernel against Triton (SGLang / vLLM) | Triton | 1.47× faster at h_q=16, 2.23× at h_q=128 (geometric mean) | |
| Decode kernel against FlashInfer 0.6.6 | FlashInfer | 3.43× faster at h_q=16, 1.77× at h_q=128 (geometric mean) | |
| 8-bit KV cache, h_q=16, b≥32, s_k≥4096 | bf16 cache | FP8 1.55–1.70×, INT8 1.54–1.60× | 0.559× the bytes |
| Decode layer, b=1, s_k=4096 | 820.1 µs eager, 114.7 µs graph | 61.5 µs | 13.3× / 1.87× |
| Decode layer, b=8, s_k=4096 | 895.1 µs eager, 184.9 µs graph | 102.3 µs (86.3 µs with FP8 KV) | 8.7× / 1.81× (2.14×) |
| Decode layer, b=32, s_k=4096 | 903.1 µs eager, 330.0 µs graph | 239.5 µs (172.0 µs with FP8 KV) | 3.8× / 1.38× (1.92×) |
| Decode layer, b=64, s_k=4096 | 899.2 µs eager, 515.0 µs graph | 431.4 µs (291.2 µs with FP8 KV) | 2.1× / 1.19× (1.77×) |
| Kernels per layer | 40–43 | 7 | |

The decode-layer baseline is an unfused PyTorch implementation of a DeepSeek-V2-Lite attention block that already uses the
v2 attention kernel; "eager" launches each operation separately, "graph" replays the same operations from one CUDA graph.

### One decode layer, step by step (µs per layer, s_k=4096, h_q=16)

| Stage | b=1 | b=8 | b=32 | b=64 |
|---|---|---|---|---|
| Unfused PyTorch, eager | 820.1 | 895.1 | 903.1 | 899.2 |
| Unfused PyTorch, whole-layer CUDA graph | 114.7 | 184.9 | 330.0 | 515.0 |
| Fused layer + Triton attention | 76.4 | 133.8 | 355.7 | 599.6 |
| Fused layer + legacy kernel | 71.1 | 115.7 | 258.2 | 440.9 |
| Phase 1: fused layer + v2 kernel | 65.8 | 117.9 | 259.3 | 440.0 |
| Phase 2: + skinny GEMM (bf16 KV) | **61.5** | **102.3** | **239.5** | **431.4** |
| Phase 2: + 8-bit KV cache (FP8) | 61.6 | **86.3** | **172.0** | **291.2** |
| Phase 3 prototype: P1 + P2 + P3 (bf16), not in the library yet | 56.2 | 97.4 | not run | not run |
| Phase 3 projection: all six items | 51–53.5 | 93–95.5 | unchanged | unchanged |

The phase-1 and phase-2 rows come from separate measurement sessions on the same GPU with the same thermal control;
same-build A/B tables are in the results document.

### The decode kernel against other implementations (CUDA graph, page size 32, bf16, µs per layer)

| Shape | v2 (this branch) | legacy | Triton | FlashInfer | legacy / v2 | Triton / v2 | FlashInfer / v2 |
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

"legacy" is the fork's kernel with the smallest change that lets it launch on sm_86 (16 query rows per CTA instead of 32).
At h_q=16 and large batch both kernels sit at the DRAM peak, so they tie; v2 wins wherever something other than bandwidth
limits the kernel.

## What was optimized

MLA decode has an arithmetic intensity of about 1.89·M FLOP per byte of KV cache (M = h_q × s_q query rows). The RTX 3090
turns from memory-bound to compute-bound at about 80 FLOP/B (M ≈ 42). So h_q=16 is a pure bandwidth problem and h_q≥64 is a
tensor-core problem, and the two need different kernels. A whole decode layer at small batch is a third problem: at b=1 it
streams 32 MB of weights and KV (37 µs at peak bandwidth), and the rest of its time is launches, cold starts and idle DRAM.

### Phase 1: decode kernel, graph-safe API, fused layer, 8-bit KV

![The v2 decode kernel](docs/optimization/figures/fig2_decode_kernel.png)

| # | Optimization | Where | Measured effect |
|---|---|---|---|
| 1 | Legacy fallback: 16 rows per CTA, 93.5 KB of shared memory | `csrc/flash_fwd_mla_kernel_sm80.h` | the fork's kernel launches on sm_86 and becomes the baseline |
| 2 | **v2 kernel**: Q held in registers, QK split-K inside the CTA, PV split by columns, P kept in registers, multi-stage `cp.async` ring | `csrc/sm86/` | h_q=128: 1.15–1.24× over legacy; h_q=16 small batch: 1.09–1.32×; h_q=16 large batch: tie at the DRAM peak |
| 3 | Two default tile plans: M≤16 → 16 rows, 16-token tiles, 3 stages; M>16 → 32 rows, 32-token tiles, 2 stages | `csrc/sm86/mla_sm86.h` | 32-token tiles are 12–13% faster at h_q≥32 |
| 4 | Page sizes 16 / 32 / 64 (64 is what vLLM uses), zero-filled tails | `csrc/sm86/` | page 64 equals page 32 (370.6 vs 370.9 µs) |
| 5 | FP16 instance | `csrc/sm86/mla_decode_sm86_fp16.cu` | 77% of the FP16 peak at h_q=128 |
| 6 | Parallel on-device scheduler metadata and a graph-safe API (`out`, `lse`, `workspace` preallocated, no host sync) | `csrc/flash_fwd_mla_metadata.cu`, `flash_mla/flash_mla_interface.py` | all 82 SMs get work; the decode can be captured in a CUDA graph |
| 7 | Combine kernel chosen by row count (wide16 / wide8 / wide4 / one warp per row) | `csrc/sm86/mla_decode_sm86.cuh` | b=1: 9.2 → 5.5 µs; b=32, s_k=1024: 11.7 → 5.2 µs |
| 8 | **8-bit KV cache**: FP8 e4m3 or INT8 latent with one fp32 scale per token, RoPE dims kept in bf16, dequantized in the kernel with LOP3 / PRMT bit operations | `csrc/sm86/mla_decode_sm86_q.cuh`, [docs/kv_quant_format.md](docs/kv_quant_format.md) | 644 B per token instead of 1152 B; 1.47–1.71× at h_q=16, b≥32 |
| 9 | **Fused decode prologue**: RoPE, RMSNorm, q concatenation and the paged KV write (with optional quantize-on-write) in one kernel | `csrc/fused/mla_decode_prologue.cu` | 1.8–14 µs instead of 59–90 µs of PyTorch operations inside a graph |
| 10 | **Whole-layer CUDA graph** for a DeepSeek-V2-Lite decode layer | `flash_mla/mla_decode_layer.py` | 2.0–14× over the unfused eager layer; 40–43 → 8 kernels per layer |

Design details: [docs/optimization/02-phase1-decode-kernel.md](docs/optimization/02-phase1-decode-kernel.md).

### Phase 2: skinny GEMM and two KV groups

Profiling the fused layer showed that at b=8 the four projections take half of the layer and that cuBLAS streams their
weights at only 19–63% of the DRAM peak, because at 1–64 tokens it opens fewer CTAs than the GPU has SMs.

| # | Optimization | Where | Measured effect |
|---|---|---|---|
| 11 | **Skinny GEMM**: the weight is the A operand of `mma.sync m16n8k16`, tokens are the B operand, a 48 KB `cp.async` ring streams the weight rows | `csrc/fused/skinny_gemm.cu`, `flash_mla/skinny_gemm.py` | the four GEMMs at b=8: 61.7 → 41.5 µs; layer at b=8: 1.13× (bf16), 1.17× (FP8) |
| 12 | **Two KV groups per CTA for the 8-bit kernel**, enabled for the whole launch when the scheduler load per part is at least 64 units | `csrc/sm86/mla_decode_sm86_q.cuh`, `csrc/flash_fwd_mla_metadata.cu` | long contexts +4–11%; FP8 against bf16 now 1.55–1.70× under a CUDA graph |

Design details: [docs/optimization/03-phase2-gemm-kv-groups.md](docs/optimization/03-phase2-gemm-kv-groups.md).

### Phase 3 (planned): what happens between the kernels

![Skinny GEMM and the phase-3 plan](docs/optimization/figures/fig3_gemm_and_l2_plan.png)

At b=1 a layer spends 27 µs above its byte floor. Six items address it. They were measured with prototypes on the real
27-layer graph (external kernels, a patched kernel copy, parallel graph branches). The prototype code is in
[research/phase3/](research/README.md); the items themselves are not implemented in the library yet.

| # | Item | Evidence | b=1, µs per layer | b=8, µs per layer | Output bit-identical |
|---|---|---|---|---|---|
| P1 | `discard.global.L2` the partial outputs after the combine has read them | in-layer A/B | −3.0 | −2.8 | yes |
| P2 | Balanced part count: the fewest parts whose heaviest part is no heavier | in-layer | −1.0 to −1.7 | −1.0 to −1.2 | no (split boundaries move; within test tolerance) |
| P3 | GEMM → GEMM prefetch chain, weights loaded `evict_first` | in-layer, patched kernel copy | −1.7 to −1.9 | −1.4 | yes |
| P4 | W_UK and the prologue in one launch | parallel-branch emulation | −1.5 to −2.4 | −0.8 to −2.3 | yes |
| P5 | No positions / slots bookkeeping on the fused path | in-layer | −0.6 | −0.4 | yes |
| P6 | Prefetch W_UV from inside the combine | synthetic benchmark only | −0.85 to −1.3 | not measured | yes |

P1 + P2 + P3 stacked, measured: b=1 61.1 → 56.2 µs, b=8 101.7 → 97.4 µs. Projection for all six: b=1 51–53.5 µs (−12% to
−16%), b=8 93–95.5 µs (−6% to −9%), b≥32 unchanged. The full plan with code sketches, go / no-go criteria and risks is in
[docs/optimization/04-phase3-plan.md](docs/optimization/04-phase3-plan.md).

### What did not work

Some of the most useful results are negative. The full list with reasons and retry conditions is in
[docs/optimization/05-what-did-not-work.md](docs/optimization/05-what-did-not-work.md); the patches of the rejected phase-2
experiments are in [research/phase2_rejected/](research/README.md). The ones that cost the most time:

- **Warp specialization** (a producer warp with mbarriers) is correct and not faster: the bandwidth case is already at
  the peak and the compute case waits on dependencies, not on barriers.
- **A fused last-arriver combine** puts a serial merge on every split's critical path (b=32, s_k=1024: 73.7 → 81.9 µs).
- **Choosing the number of KV groups per part** makes a launch slower than either uniform choice (151 vs 122 µs): DRAM
  bandwidth is shared in proportion to the bytes each CTA has in flight, so the CTAs with less in flight starve and
  decide the tail. Any choice that changes the bytes in flight per CTA has to be the same for the whole launch.
- **Prefetching the first KV tile before the decode** gains 0.9–1.9 µs in a synthetic chain and loses 0.1–1.7 µs inside the
  real layer. Synthetic microbenchmarks are not accepted as evidence for anything that depends on L2 or DRAM state.
- **8-bit KV at h_q≥64** is slower than bf16 (0.93–0.97×): that regime is compute-bound and dequantization adds work.

## Build and test

Requirements: an Ampere GPU with compute capability 8.6 (developed on RTX 3090), nvcc 12.x (tested with 12.4) and a
PyTorch build for the same CUDA major version (tested with 2.7.1+cu126). CUTLASS is a submodule.

```bash
git clone --recursive -b ampere-3090 https://github.com/KernelFlow-ops/FlashMLA.git
cd FlashMLA
export CUDA_HOME=/path/to/cuda-12.x            # the toolkit that provides nvcc
FLASH_MLA_ARCH=86 python setup.py build_ext --inplace
#   FLASH_MLA_LINEINFO=1                          adds -lineinfo (Nsight Compute source view)
#   FLASH_MLA_DEFINES="FLASH_MLA_SM86_NT_SMALL=32 ..."   sets compile-time kernel tunables

export PYTHONPATH=$PWD
python -m pytest tests/test_flash_mla_sm86.py tests/test_flash_mla_sm86_quant.py tests/test_fused_prologue.py \
    tests/test_fused_layer.py tests/test_kv_quant.py tests/test_decode_layer_e2e.py tests/test_skinny_gemm.py \
    tests/test_flash_mla_sm80.py -q
```

The build produces two extensions: `flash_mla_cuda` (attention kernels, scheduler metadata) and `flash_mla_fused_cuda`
(fused prologue, skinny GEMM). The suite has 760 tests: outputs and LSE against an fp32 reference (cosine distance below
8e-5 for bf16 / fp16), NaN-poisoned padding, page sizes 16 / 32 / 64, h_q 8–128, s_q 1–2, variable lengths, CUDA-graph
replay with changing lengths, run-to-run bit equality, and the 8-bit accuracy gates. The kernels are clean under
`compute-sanitizer` (memcheck, racecheck, synccheck, initcheck).

## Usage

### Attention kernel

```python
import torch, flash_mla

# q [b, s_q, h_q, 576] (512 absorbed latent + 64 RoPE); paged cache [num_blocks, page, 1, 576], page in {16, 32, 64}
md, nsplit = flash_mla.get_mla_metadata(cache_seqlens, s_q * h_q, 1)       # on device, no host sync
ws = flash_mla.alloc_mla_workspace(b, s_q * h_q, 1, md.shape[0])            # once; reused by every layer and step
out = torch.empty(b, s_q, h_q, 512, dtype=q.dtype, device=q.device)
lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=q.device)

o, lse = flash_mla.flash_mla_with_kvcache(
    q, kv_cache, block_table, cache_seqlens, 512, md, nsplit,
    softmax_scale=(128 + 64) ** -0.5,     # DeepSeek: the head dim before absorption, not 576 ** -0.5
    causal=s_q > 1,                       # multi-token prediction (s_q = 2) needs the causal tail
    out=out, lse=lse, workspace=ws,       # with these three given: no allocation, no sync, CUDA-graph capturable
    impl="v2")                            # "v2" (default) | "legacy" (fork kernel, page 32, bf16) | "legacy_ws"
```

`impl="v2"` needs `d=576`, `d_v=512`, a shared K/V cache and 16-byte aligned cache rows.

### 8-bit KV cache

```python
# uint8 [num_blocks, page, 640] + fp32 scale [num_blocks, page]; q stays bf16 / fp16. Format: docs/kv_quant_format.md
kvq, scale = flash_mla.quantize_kv(kv_native.view(-1, page, 576), "int8")   # or "fp8"; INT8 is 2.4-13x more accurate
md, nsplit = flash_mla.get_mla_metadata(cache_seqlens, s_q * h_q, 1, kv_dtype="int8")
o, lse = flash_mla.flash_mla_with_kvcache(q, kvq, block_table, cache_seqlens, 512, md, nsplit,
                                           softmax_scale=(128 + 64) ** -0.5, kv_scale=scale, kv_dtype="int8")
```

Use it for h_q ≤ 32. Accuracy on synthetic MLA-shaped caches (cosine distance of the attention output against the bf16
cache): FP8 5e-5 to 2.4e-3, INT8 6e-6 to 2.5e-4. No real model weights were available on the test machine.

### A whole decode layer

```python
from flash_mla import MLAConfig, MLADecodeLayer, alloc_cache, make_state, capture_layer_graph

# q_proj / kv_a_proj -> W_UK absorption -> fused prologue (RoPE, RMSNorm, concat, paged-cache write, optional
# quantize-on-write) -> v2 attention -> W_UV -> o_proj; the whole step is capturable in one CUDA graph.
# gemm="auto" runs the projections on the skinny GEMM for b * s_q <= 64 (q/kv_a and o_proj go back to cuBLAS above 32
# tokens); gemm="torch" keeps cuBLAS everywhere.
```

`benchmarks/bench_layer.py` is the shortest complete example: it builds 27 layers, captures them with
`capture_layer_graph` and times the replay.

## Tunables

Run-time environment variables (read on every call, meant for sweeps and A/B runs):

| Variable | Default | Meaning |
|---|---|---|
| `FLASH_MLA_SM86_CFG` | by M | `"bm,nt,ns[,minb[,spec[,pipe]]]"`: rows per CTA, tokens per tile, ring stages, CTAs per SM, v2b, v2c |
| `FLASH_MLA_SM86_QCFG` | `16,2,1,32,2` at M≤16 | 8-bit kernels: `"bm,ns[,minb[,nt[,kg]]]"`; `16,2,1,32` is the single-group plan of phase 1, `16,2,2,32` runs 2 CTAs per SM |
| `FLASH_MLA_SM86_KGMIN` | 64 | load per part (32-token units) from which the 8-bit kernel uses two KV groups; 0 = always |
| `FLASH_MLA_SM86_OV` | 2 | scheduler: fixed overhead per request, in 32-token units |
| `FLASH_MLA_SM86_MINBLK` | 0 | scheduler: minimum units per part (0 = off) |
| `FLASH_MLA_SM86_FUSE` | 0 | maximum number of splits merged by the fused last-arriver combine (0 = always use the combine kernel) |
| `FLASH_MLA_SM86_COMBINE` | by row count | forces a combine variant |

The same names (and the per-plan defaults such as `FLASH_MLA_SM86_NT_SMALL`) can be set at compile time through
`FLASH_MLA_DEFINES`; see `csrc/sm86/mla_sm86.h`.

## Benchmarks and profiling

```bash
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle gpu>
python benchmarks/phase0_peaks.py                                  # bandwidth and tensor peaks of the card
python benchmarks/bench_decode.py --preset core --impls v2,legacy,triton --modes eager,graph --cool
python benchmarks/bench_layer.py --b 8 --sk 4096 --page 64 --attn v2 --variants fused --modes graph
python benchmarks/bench_layer_gemms.py                             # the four projections, cuBLAS vs skinny
python benchmarks/summarize_matrix.py                              # regenerates docs/bench_matrix.md from the CSV logs
tools/prof_ncu.sh  <tag> -- python benchmarks/bench_one.py --impl v2 --hq 16 --b 64 --sk 4096 --page 64
tools/prof_nsys.sh <tag> -- python benchmarks/bench_layer.py --b 8 --variants fused --modes graph --layers 8 --nvtx
```

Eager timing flushes L2 with a 256 MB read before every call and takes the median of CUDA-event timings. Graph timing
captures N layers (27 by default) with separate KV caches so that L2 hits do not flatter the result. `--cool` waits for the
GPU to cool down between configurations; SM clock and power are logged with every row. FlashInfer runs in its own
environment through `benchmarks/bench_flashinfer.py` and writes the same CSV format. CSV logs go to `$FLASHMLA_LOG_DIR`.
The scripts under `tools/` and a few defaults in `benchmarks/` contain paths of the machine they were written on (conda
environment, log and profile directories); adjust them before use.

## Repository layout

```
csrc/sm86/                      v2 decode kernels: bf16 / fp16 (mla_decode_sm86.cuh), 8-bit KV (mla_decode_sm86_q.cuh),
                                tile plans and tunables (mla_sm86.h), PTX helpers (sm86_ptx.cuh)
csrc/fused/                     fused decode prologue, skinny GEMM, their Python bindings (fused_api.cpp)
csrc/flash_fwd_mla_metadata.cu  parallel split-KV scheduler
csrc/flash_fwd_mla_kernel_sm80*.h   the fork's kernels ("legacy"), with the sm_86 fallback
flash_mla/                      Python API: attention interface, KV quantization, prologue, skinny GEMM, decode layer
tests/                          760 tests
benchmarks/                     kernel and layer timing, baselines (Triton, FlashInfer, legacy), matrix summarizer
tools/                          build, Nsight Compute and Nsight Systems helpers
research/                       not part of the library: phase-3 prototype harnesses, patches of rejected phase-2 experiments
docs/optimization/              the consolidated write-up: results, phases 1-3, negative results, figures and their sources
docs/                           original working report (Chinese), performance logs, benchmark matrix, KV format
```

## Limitations

- Measured on RTX 3090 only. The kernels target sm_86; other Ampere parts (A100 is sm_80 with a different shared-memory
  limit and a full-rate tensor pipeline) were not tested and would want different tile plans.
- Small batches miss the original latency target: with less than about 35 MB of KV the decode kernel alone is 2.8–4.0 µs
  over `1.3 × bytes / bandwidth + 5 µs` under a CUDA graph. Phase 3 addresses the layer-level part of this.
- `discard.global.L2` and `prefetch.global.L2` (phase 3) are hints whose effect depends on the GPU and driver.
- 8-bit accuracy was evaluated on synthetic data.
- `docs/bench_matrix.md` was generated from the phase-1 build; its 8-bit section predates the two KV groups.

## Acknowledgement and citation

FlashMLA is inspired by [FlashAttention 2 & 3](https://github.com/dao-AILab/flash-attention/) and
[CUTLASS](https://github.com/nvidia/cutlass). The Ampere port this branch starts from is
[pzhao-eng/FlashMLA](https://github.com/pzhao-eng/FlashMLA). The baselines are the Triton decode kernels of
[SGLang](https://github.com/sgl-project/sglang) / [vLLM](https://github.com/vllm-project/vllm) and
[FlashInfer](https://github.com/flashinfer-ai/flashinfer). The FP8 dequantization uses the LOP3 / PRMT bit tricks known
from Marlin. The phase-3 plan builds on NVIDIA's documentation of
[`cuda::discard_memory`](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory/discard_memory.html) and on
Hazy Research's ["Look Ma, No Bubbles!"](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernel},
      author={Jiashi Li},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
