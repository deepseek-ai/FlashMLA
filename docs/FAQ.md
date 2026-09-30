# FlashMLA FAQ

Frequently asked questions about FlashMLA, based on common GitHub issues and user inquiries.

## General Questions

### Does `flash_mla_with_kvcache` work only in paged mode?

No. It supports both paged and non-paged KV cache. Paged mode is recommended for long-context workloads and memory fragmentation control, but non-paged works for shorter, fixed-length workloads.

### Can MLA/MHA be used in the prefill stage?

Yes. Prefill supports both MLA and MHA:
- Use **MLA** for sparse/long contexts with grouped-query attention
- Use **MHA** when you need standard dense attention or compatibility with existing kernels

### What GPU architectures are supported?

| Architecture | Support |
|--------------|---------|
| SM90 (Hopper - H100/H800) | Full support |
| SM100 (Blackwell - B100/B200) | Full support |
| SM120 | Not supported |
| Older (SM80, SM70) | Not supported |

Use matching CUDA drivers (12.8+) and ensure your build targets these architectures.

## Sparse Attention

### How do I test sparse attention?

1. Enable MLA sparse mode in your config
2. Load a sparse pattern (block-sparse mask)
3. Run the sparse test suite: `python tests/test_flash_mla_sparse.py`
4. Compare outputs against dense runs to validate correctness
5. Check perf counters to confirm sparse paths are used

### What's the difference between dense and sparse MLA?

| Feature | Dense MLA | Sparse MLA |
|---------|-----------|------------|
| Computation | Full attention matrix | Pruned blocks |
| Performance (H800) | ~660 TFlops | ~410 TFlops |
| Use case | Short sequences, full accuracy | Long sequences, structured sparsity |
| SM90 support | Yes | No |
| SM100 support | Yes | Yes |

Dense MLA computes full attention (higher FLOPs, higher accuracy for dense tasks). Sparse MLA prunes blocks to reduce compute/memory, trading some fidelity for speed/throughput on long or structured-sparsity workloads.

## Integration

### How do I integrate FlashMLA with vLLM/SGLang/other frameworks?

1. Use the provided FlashMLA attention interface
2. Register it as the backend kernel for attention ops
3. Follow the framework's custom op/plugin hooks:
   - **vLLM**: Custom attention registry
   - **SGLang**: Extension points
4. Ensure paged KV cache shape/block alignment matches FlashMLA's layout (block size = 64)

Example for vLLM:
```python
from vllm.attention.backends import register_attention_backend
from flash_mla import FlashMLABackend

register_attention_backend("flash_mla", FlashMLABackend)
```

## Performance

### What performance should I expect on different GPUs?

| GPU | Dense MLA | Sparse MLA | Notes |
|-----|-----------|------------|-------|
| H800 | ~660 TFlops | ~410 TFlops | Reference config |
| H100 | ~600 TFlops | ~380 TFlops | Slightly lower than H800 |
| B200 | ~1460 TFlops (MHA prefill) | TBD | SM100 optimizations |

Actual numbers vary with sequence length, batch shape, paging configuration, and sparsity pattern. Expect higher throughput on SM100 vs SM90 when similarly configured.

## Memory & Batching

### How do I handle variable-length sequences in a batch?

1. Use paged KV cache with proper offsets/indirection per sequence
2. Provide per-sequence lengths via `seq_lens` tensor
3. Use masks to avoid reading/writing past valid tokens
4. Pad to block boundaries (64 tokens) only where required by the cache layout

Example:
```python
# Variable-length batch
seq_lens = torch.tensor([128, 256, 64, 512])  # Different lengths per batch
nblk_per_seq = (seq_lens + 63) // 64  # Blocks needed per sequence
max_nblk = nblk_per_seq.max()

# Allocate cache with max blocks
k_cache = torch.zeros(B, Hk, max_nblk, 64, D, dtype=torch.bfloat16)
```

### What's the block size for paged KV cache?

Block size is **64 tokens**. This is fixed and cannot be changed. Align all allocations and paging logic to 64-token blocks.

## Quantization

### How do I enable FP8 quantization for the KV cache?

1. Ensure GPU supports FP8 (SM90/SM100)
2. Build with FP8 KV cache enabled
3. Use the FP8-compatible KV layout
4. Apply calibration/scaling utilities provided by FlashMLA

```python
# FP8 KV cache setup
k_cache = torch.zeros(B, Hk, Nblk, 64, D, dtype=torch.float8_e4m3fn)
v_cache = torch.zeros(B, Hk, Nblk, 64, D, dtype=torch.float8_e4m3fn)

# Scaling factors (calibrated)
k_scale = torch.ones(B, Hk, 1, 1, 1)
v_scale = torch.ones(B, Hk, 1, 1, 1)
```

Fall back to FP16/BF16 if FP8 is unsupported on your GPU.

---

## Still have questions?

If your question isn't answered here:
1. Search [existing issues](https://github.com/deepseek-ai/FlashMLA/issues)
2. Open a new issue with your GPU model, CUDA version, and detailed question
