# Research code

Code that produced numbers quoted in [docs/optimization/](../docs/optimization/README.md) but is not part of the shipped
library: the prototypes behind the phase-3 plan, and the patches of phase-2 experiments that were measured and rejected.
Nothing here is imported by `flash_mla`, and nothing here is covered by the test suite.

## `phase3/`: prototypes behind the phase-3 plan

The phase-3 items (P1–P6 in [04-phase3-plan.md](../docs/optimization/04-phase3-plan.md)) are not implemented in the
library. These harnesses measured what each item would be worth, without changing the shipped kernels: they add an extra
kernel to the real 27-layer CUDA graph, swap in a patched copy of a kernel, or run kernels on parallel graph branches.

| File | What it measures | Used for |
|---|---|---|
| `p3_timeline.py` | per-layer kernel timeline (median duration and idle gap per kernel) from an Nsight Systems sqlite of `benchmarks/bench_layer.py` | the "one layer at batch 1" table: 64.2 µs against a byte floor of 37.0 µs |
| `p3_discard_layer.py` | real layer; an extra kernel after the attention that does nothing (`noop`) or issues `discard.global.L2` over the partial outputs (`discard`) | **P1**: discard − noop = −2.6 to −3.3 µs per layer at b = 1, 4, 8 |
| `p3_refresh_emul.py` | real layer; the per-step refresh with and without the positions / slots bookkeeping | **P5**: −0.4 to −0.6 µs per layer |
| `p3_fuse_emul.py` | real layer; W_UK and the prologue serial, on two graph branches, or with the prologue removed (upper bound) | **P4**: −0.8 to −2.4 µs per layer |
| `p3_layer_gemm_pf.py` | real layer; `skinny_bmm` monkeypatched to a kernel copy that can load weights `evict_first` and prefetch the next GEMM's weights, with the discard kernel optionally stacked on top | **P3**, and the P1 + P3 stack |
| `p3_gemm_pf.cuh` | the kernel copy used by `p3_layer_gemm_pf.py` (bf16, no split-K, T ≤ 8) | |
| `p3_pf_chain.cu` | stand-alone program: the four GEMMs of 27 layers in one graph. Mode `pf`: where and how much to prefetch; mode `comb`: prefetch W_UV from a combine-like reader; any other mode: cost of dirty scratch data between the GEMMs and of discarding it | first evidence for P1 and P3, the only evidence for **P6**, and the negative results on prefetch placement |
| `p3_b5_prefetch.py` | synthetic chain: weight-like stream → prefetch kernel → decode; warming the metadata, the block table, the first KV tile or all of the KV | negative results: a separate prefetch kernel, metadata / block-table warming |
| `p3_layer_pf_emul.py` | real layer; the KV first-tile prefetch on a side branch at three different places | negative result: slower in the layer by 0.1–1.7 µs |
| `p3_common.py` | build-directory helper | |

**P2** (balanced part count) has no script. It was emulated with the existing scheduler knob: run `benchmarks/bench_layer.py`
with `FLASH_MLA_SM86_MINBLK = ceil(total / 82)`, where `total = b × (ceil(s_k / 32) + 2)`. For b = 1 and s_k = 4096 that is
`total = 130`, `MINBLK = 2`, 65 parts. The stacked P1 + P2 + P3 measurement is `p3_layer_gemm_pf.py` with the `pf2+discard`
style variants under that environment variable.

### Running

The library must be built and importable (`PYTHONPATH` pointing at the repository root); use an otherwise idle GPU.

```bash
cd research/phase3
export FLASHMLA_P3_BUILD=/some/scratch/dir          # inline extensions are compiled here (default: the system temp dir)
python p3_discard_layer.py --b 1 --sk 4096           # every script: --b, --sk, --kv native|fp8, --layers (default 27)
python p3_layer_gemm_pf.py --b 8 --sk 4096 --seq repo,copy,pf2,pf2+discard
python p3_timeline.py /path/to/trace.sqlite

nvcc -O3 -arch=sm_86 -o p3_pf_chain p3_pf_chain.cu && ./p3_pf_chain 1 pf      # <tokens> <pf | comb | scratch>
```

Each script prints microseconds per layer for every variant, and compares outputs bit for bit where that is meaningful.

### Reading the output

- These are timing harnesses. Variants that run a kernel on a stale input or skip a kernel (`concurrent`, `no_prologue`,
  `F`, …) produce wrong layer outputs on purpose; the docstring of each script says which variants are numerically valid.
- The first repetition of a variant is often 1–3 µs high. Variants are run in an interleaved sequence (`--seq`) so that
  drift shows up as a difference between repetitions of the same variant.
- The extra kernel of `p3_discard_layer.py` costs one launch (0.7–0.9 µs) that a discard inside the combine kernel would
  not pay. That is why the comparison is `discard` against `noop`, not against `orig`.
- Results from the synthetic programs (`p3_pf_chain.cu`, `p3_b5_prefetch.py`) did not always carry over to the real layer;
  the plan relies on the in-layer scripts.

## `phase2_rejected/`: experiments that were measured and not kept

Plain diffs against this branch (`git apply research/phase2_rejected/<file>`). All three leave the output bit-identical.
They are described in [03-phase2-gemm-kv-groups.md](../docs/optimization/03-phase2-gemm-kv-groups.md), section 4.

| Patch | Change | Measured |
|---|---|---|
| `b3_combine_single_round.patch` | the wide combine reads up to 96 splits per round, so the 82 splits of b = 1 need one round | combine 4.28 → 4.26 µs: no gain |
| `b4_rescale_skip.patch` | the softmax skips the rescale of the accumulator when no row maximum of the warp grew | decode 3–5% slower |
| `b4_h128_pingpong.patch` | h_q=128: the second M group runs half a tile behind the first (opt-in: 7th field of `FLASH_MLA_SM86_CFG`, or `-DFLASH_MLA_SM86_PP=1`) | 0.96–0.98× |
