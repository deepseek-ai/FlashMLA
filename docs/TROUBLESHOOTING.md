# FlashMLA Troubleshooting Guide

This guide summarizes frequent issues reported by users and provides quick fixes. If a problem persists, follow the linked GitHub issues for deeper context.

## GPU Architecture & Compatibility

- **Supported architectures**: SM90 (Hopper), SM100 (Blackwell). SM120 support is not available yet.
- **Sparse attention support**:
  - SM100: Sparse BF16/FP16 available.
  - SM90: Dense attention only; Sparse BF16 is **not supported** (see error below).
  - SM120: Not supported.
- **Performance reference** (dense decoding, H800): Dense MLA ~660 TFlops, Sparse ~410 TFlops.

## GPU Compatibility Matrix

| GPU Arch | Dense MLA | Sparse MLA | Notes |
|----------|-----------|------------|-------|
| SM90 (Hopper) | BF16/FP16 | Not supported | Use dense kernels only |
| SM100 (Blackwell) | BF16/FP16 | BF16/FP16 (config-dependent) | Ensure kernels are built with sparse enabled |
| SM120 | Unsupported | Unsupported | Not planned yet |

## Sparse vs Dense Attention Usage

**Symptom**: Users enable sparse attention on SM90 and get lower throughput or runtime errors.

**Fix**: On SM90, disable sparse attention (set `use_sparse=False` or equivalent flag). Use dense kernels only. On SM100, sparse attention requires correct build flags and runtime configuration. Ensure your model config selects sparse kernels only where supported.

## RuntimeError: Sparse BF16 MLA is not supported on SM90

**Error**:
```
RuntimeError: Sparse BF16 MLA is not supported on SM90
```

**Cause**: Sparse kernels are not compiled or supported on SM90.

**Fix**:
1. Switch to dense attention
2. Rebuild without sparse flags
3. Verify your model config does not request sparse kernels

## Build Errors by CUDA Version

**Requirements**: CUDA 12.8+, PyTorch 2.0+

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `nvcc fatal : Unsupported gpu architecture 'sm_120'` | Unsupported arch flag | Remove unsupported arch flags; limit to sm_90 or sm_100 |
| `undefined symbol: ... cudaMemcpyAsync` | CUDA runtime/toolkit mismatch | Align CUDA toolkit with driver/runtime version |
| `ptxas fatal : Value 'sm_90' is not defined` | Old CUDA toolkit | Ensure CUDA >= 12.8 |

### Build Checklist

1. Confirm `nvcc --version` reports 12.8 or newer
2. Clear `CMAKE_CUDA_ARCHITECTURES` or `TORCH_CUDA_ARCH_LIST` to only include `90` or `100`
3. Remove previous build artifacts (`build/`, `*.so`) before rebuilding
4. Ensure PyTorch is compiled for the same CUDA major/minor as your toolkit

## KV Cache Paging Mode Questions

**Symptom**: Unexpected memory usage or OOM when paging is enabled.

**Fixes**:
1. Verify paging is supported on your GPU (SM90/SM100 only)
2. Tune page size and eviction threshold; start with defaults provided by FlashMLA
3. If instability persists, disable paging to confirm the root cause, then re-enable with more conservative thresholds

## Windows / ARM64 Build Support

**Current status**: Official builds target Linux x86_64. Windows and ARM64 are not officially supported.

**Workarounds (community)**:
- **Windows**: Use WSL2 with CUDA 12.8+ and compatible drivers
- **ARM64** (e.g., Grace): Cross-compile is experimental; verify toolchain supports your GPU arch and CUDA 12.8+

## Metadata API Compatibility Issues

**Symptom**: Metadata API shape/dtype mismatches across releases.

**Fixes**:
1. Align FlashMLA version with the metadata API expectations in your host framework
2. Regenerate metadata after upgrading FlashMLA or PyTorch
3. Check for breaking changes called out in release notes; adjust field names or tensor layouts accordingly

## Sparse Attention Configuration Gotchas

- Ensure runtime flags match build capabilities: if built without sparse, disable sparse at runtime
- **Mixed precision**: Sparse BF16 on SM90 is unsupported; use dense BF16 or switch to FP16 where allowed
- **Fallbacks**: If sparse fails to dispatch, explicitly force dense kernels to avoid silent slow paths

## Advanced Help (Relevant Issues)

| Category | Related Issues |
|----------|----------------|
| GPU arch support | #101, #113, #124, #134 |
| Sparse vs dense confusion | #87, #116, #142 |
| CUDA version build failures | #115, #121, #160 |
| Sparse BF16 on SM90 runtime error | #93, #110, #178 |
| KV cache paging behavior | #121, #126 |
| Windows/ARM64 build requests | #109, #119, #151 |
| Metadata API compatibility | #108, #126, #173 |

---

If your issue is not covered here, please open a new GitHub issue with:
- GPU model
- CUDA/PyTorch versions
- Build flags
- Full error log
