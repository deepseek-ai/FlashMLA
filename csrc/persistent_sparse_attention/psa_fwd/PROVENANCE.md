# Kernel maintenance notes

This directory contains a self-contained SM100 sparse-prefill implementation
used by the optional private extension. It is kept in newly added files so the
stock FlashMLA kernels and parameter ABI remain unchanged.

The implementation uses private parameters, namespaces, and launch entry
points. Its numerical device code is kept separate from Python and Torch
registration details, which allows the extension boundary to follow the
target branch without rewriting the kernel body.

The source closure requires CUDA Toolkit SM100 support and the repository's
official CUTLASS/CuTe headers. Shared helper code remains covered by the
repository's Apache-2.0 license and NOTICE attribution.

Correctness and port-equivalence checks are summarized in
`docs/persistent_sparse_attention/VALIDATION.md`.
