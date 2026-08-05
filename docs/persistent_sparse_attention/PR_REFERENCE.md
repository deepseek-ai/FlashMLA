# PR reference: SM100 persistent sparse attention for V4 CSA/HCA

> Reference text only. This description is written for a PR whose base is
> `b7643bd54521f563b839b98289b5cd048c062ba2`; revise it against the final
> destination template before submission.

## Title

Add SM100 persistent sparse-attention kernels and V4 CSA/HCA interfaces

## Summary

This PR adds an optional SM100 persistent sparse-attention component for V4
CSA/HCA pipelines. It combines persistent scheduling with kernel fusion and
exposes the required prefill, decode, frontend, indexing, readiness, and output
interfaces through one FlashMLA extension.

The default FlashMLA call path remains unchanged. Specialized behavior is
selected only when the caller supplies persistent-attention options or invokes
the component API directly.

## What changed

- Add SM100 sparse-prefill kernels with caller-owned output, bounded persistent
  launches, optional Q preprocessing, ready signaling, an additional KV pool,
  and fused output handling.
- Add the corresponding row-ready sparse-decode specialization for future
  CSA/HCA decode integration.
- Add the V4 CSA/HCA pipeline kernels, including persistent frontend, ROOT
  fusion, logits/WqB-ready adapters, index/cache operations, learned TopK,
  state/cache updates, and output quantization.
- Register the complete suite in
  `flash_mla.persistent_sparse_attention_cuda` and expose Python wrappers from
  `flash_mla.persistent_sparse_attention`.
- Compile fixed frontend, logits, and WqB-ready variants ahead of time; the
  installed wheel does not use runtime JIT, NVCC, or NVRTC.
- Add DeepGEMM as a pinned build-only submodule, matching the existing CUTLASS
  dependency model. Unmodified headers come directly from the submodules and
  project-specific adaptations remain in a small overlay.
- Add CUDA 13.x CCCL and SM100a build definitions.
- Consolidate implementation, tests, reference oracles, and documentation
  under the persistent sparse-attention component directories.

## Build

~~~bash
git submodule update --init csrc/cutlass csrc/deep_gemm
env FLASH_MLA_BUILD_PERSISTENT_SPARSE_ATTENTION_SUITE=1 MAX_JOBS=8 python setup.py bdist_wheel
~~~

The full V4 CSA/HCA suite is opt-in. Normal FlashMLA builds keep the existing
sparse-attention surface and do not include the additional pipeline kernels.

## Validation

- CUDA 13.2 full-suite build on B200: PASS
- Source-to-port output comparison: all qualified output fields bitwise exact
- Median target/source latency: frontend 0.991, logits 0.773, WqB-ready 0.997,
  packed WqB-ready 0.994 (25% acceptance band): PASS
- Persistent sparse-attention prefill/decode regression on B200: PASS
- Packaging contracts: 5/5 PASS
- Isolated wheel import and AOT kernel smoke on B200: PASS
- Wheel contains no dependency header snapshot, legacy `csa_cuda`, or runtime
  compiler dependency

Quick reproduction:

~~~bash
git submodule update --init csrc/cutlass csrc/deep_gemm
env FLASH_MLA_BUILD_PERSISTENT_SPARSE_ATTENTION_SUITE=1 FLASH_MLA_DISABLE_SM90=1 MAX_JOBS=8 python setup.py build_ext --inplace
python tests/persistent_sparse_attention/test_packaging_contract.py
python tests/persistent_sparse_attention/test_private_extension.py
~~~

Detailed architecture and dependency boundaries are documented in
`docs/persistent_sparse_attention/DESIGN.md`. Test coverage and summarized
results are documented in `docs/persistent_sparse_attention/VALIDATION.md`.
