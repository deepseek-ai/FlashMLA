# DeepGEMM/CUTLASS build dependency ledger

The full persistent sparse-attention suite is compiled ahead of time against
two build-only git submodules:

- DeepGEMM 559d79fb6994a58b8a15b4b93bf13ccc16edf247;
- CUTLASS 147f5673d0c1c3dcf66f78d677fd647e4a020219.

Unmodified headers are included directly from csrc/deep_gemm and
csrc/cutlass. Files under include_overlays/ are only the small set of
project-owned adaptations required by the persistent/fused kernels; overlay
search order leaves both submodules untouched.

DeepGEMM, CUTLASS, CuTe, NVCC, and NVRTC are not runtime dependencies. The
wheel contains the compiled extension and Python/config files only.
