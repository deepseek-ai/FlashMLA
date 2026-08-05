# ROOT fusion dependency ledger

ROOT is an AOT CUDA launch based on public DeepGEMM commit
`559d79fb6994a58b8a15b4b93bf13ccc16edf247` (MIT). The build never patches a
user or CI checkout.

The three small device-body patches under `patches/` document the adapted
DeepGEMM changes. Their project-owned results and the standalone C0/C4
epilogue live in
`csrc/persistent_sparse_attention/deepgemm/include_overlays/`; all unmodified
headers are included directly from the pinned `csrc/deep_gemm` submodule. ROOT
is compiled AOT with `FLASH_MLA_CSA_DEEPGEMM_ROOT_ADAPTED=1`, and no dependency
header snapshot is packaged in the wheel.

The scheduler patch adds an explicit logical CTA index. The GEMM patches factor
public global kernels into callable device bodies while preserving their public
wrappers. The reduced FP8/FP4 adapter also retains the final TMA-store wait. No
ready-publication, dynamic handoff, counter/clock, chunk, or steady-SM policy is
introduced into those GEMM bodies.

Imported standalone source truth:

| Standalone file | Accepted source SHA-256 |
| --- | --- |
| `root_fusion.cuh` | `01fb63eda8736b2a580814059a2574c6d28db333d842c8f857dc9b87003b968b` |
| `internal/projection_fusion.cuh` | `25d3d405759261d53f6bcdeb170070ac0e5935f86f65c1cfce8bdbe47a510628` |
| `internal/index_compress_fusion.cuh` | `0cd8ee5dffc3cc77199f7f2e38ddba0cee33f20b430e6298333319304b38f38b` |
| `internal/bf16_pair_gemm.cuh` | `ef8d0bb182c3c192682bd7d5d0c576f73d0aeee06a06ac64d2115829f1749b51` |
| `internal/c0_c4_epilogue.cuh` | `467926b0e5edf72875e95cacd01fdebdbcefec86f5dfdb97e3e93c017f3b8367` |

Release gates are the production-shape source-oracle comparison, the independent
nonzero PyTorch impulse oracle, CUDA Graph replay, canary guards, and B200/GB200
accepted grid validation. Results and reproduction commands are in
`docs/persistent_sparse_attention/VALIDATION.md`.
