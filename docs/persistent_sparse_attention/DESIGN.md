# Persistent sparse-attention suite design

## Goal

Keep the sparse-attention kernel and its persistent/fused pipeline helpers in
one maintainable FlashMLA component. The implementation uses persistent
scheduling and kernel fusion, but does not require an external runtime package
after the wheel is built.

## Repository layout

There is one native source root and one Python package:

~~~text
csrc/persistent_sparse_attention/
  api.cpp
  pipeline_bindings.cpp
  psa_fwd/
  fwd_for_small_topk/
  deepgemm/                 # AOT launchers and project-owned overlays
  root_fusion/
  faithful_leaves/
  ...                       # remaining pipeline operations

flash_mla/persistent_sparse_attention/
  __init__.py
  ops.py
  aot_adapters.py
  configs/
  kernels/

tests/persistent_sparse_attention/
  reference/               # independent PyTorch oracles
  test_native_gpu.py
  test_faithful_leaves_gpu.py
  test_root_fusion_gpu.py
  test_private_extension.py
  test_packaging_contract.py
~~~

All native entry points are registered in
flash_mla.persistent_sparse_attention_cuda. There is no second csa_cuda
extension and no csrc/csa or flash_mla/csa tree.
The component-specific tests and references likewise live under one dedicated
`tests/persistent_sparse_attention` directory.

## Dependency boundary

CUTLASS and DeepGEMM are build-only git submodules:

| Dependency | Path | Pinned revision |
| --- | --- | --- |
| CUTLASS | csrc/cutlass | 147f5673d0c1c3dcf66f78d677fd647e4a020219 |
| DeepGEMM | csrc/deep_gemm | 559d79fb6994a58b8a15b4b93bf13ccc16edf247 |

Unmodified headers are included directly from the two submodules. The submodule
working trees stay clean. The small include_overlays directory contains only
the kernel adaptations owned by this change and is searched before the
DeepGEMM include directory.

The frontend, logits, and WqB-ready variants are fixed AOT instantiations. Host
code creates CUDA tensor maps directly and launches the compiled kernels. The
installed wheel therefore does not need DeepGEMM, CUTLASS, CuTe, NVCC, NVRTC,
or a packaged header snapshot.

## Build modes

The normal FlashMLA build keeps the existing persistent sparse-attention
prefill/decode surface. The complete fused pipeline suite is opt-in:

~~~bash
git submodule update --init csrc/cutlass csrc/deep_gemm
env FLASH_MLA_BUILD_PERSISTENT_SPARSE_ATTENTION_SUITE=1 python setup.py bdist_wheel
~~~

FLASH_MLA_BUILD_CSA=1 remains a compatibility alias for development branches.
Both modes build the same persistent_sparse_attention_cuda extension; the flag
only controls whether the additional pipeline sources are included.

CUDA 13 builds use the CUDA 13 CCCL include layout and explicitly generate
sm_100a code. The accepted fused sources do not use --use_fast_math so their
arithmetic matches the qualified source implementations.

## API and support boundary

The Python surface is flash_mla.persistent_sparse_attention. It contains the
sparse prefill/decode APIs plus the persistent pipeline operations, fused
frontend, logits, and WqB-ready adapters.

The AOT adapters intentionally accept the qualified B200/GB200 configurations
rather than compiling arbitrary shapes at runtime. Unsupported shapes or SM
counts fail with an explicit argument error.

## Routing and compatibility

The public `flash_mla_sparse_fwd` positional API remains unchanged. Calls
without advanced options continue through the stock FlashMLA extension.
Caller-owned output, persistent topology, ready signaling, fused preprocessing
or epilogue inputs, and the additional KV pool select the unified persistent
sparse-attention extension lazily. Existing `indexer_topk` calls remain on the
stock route.

The component also carries the row-ready decode specialization. It is compiled
and regression-tested even when a current caller only needs prefill, so future
pipeline integrations do not require another dependency split.

The specialized SM100 paths validate their supported head dimensions, grid
sizes, and option combinations explicitly. Unsupported configurations fail at
the API boundary instead of silently selecting a numerically different kernel.

## Maintenance rules

1. Update submodule pins explicitly and validate both revisions at build time.
2. Never edit csrc/deep_gemm or csrc/cutlass in this repository.
3. Put project-specific kernel changes in the overlay or project source tree.
4. Add a new AOT instantiation and source-oracle A/B test when expanding a
   supported configuration.
5. Reject wheel contents that include dependency headers or a second native
   extension.

See [VALIDATION.md](VALIDATION.md) for correctness, latency, and wheel gates.
