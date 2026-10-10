# FlashMLA 2.1.0 for PPU

## Introduction

FlashMLA 2.1.0 is an efficient MLA kernel designed specifically for PPUs, powering the DeepSeek-V4 models.

This repository contains the following implementations:

**Sparse Attention Kernels**

*These kernels power DeepSeek Sparse Attention (DSA), as introduced in [this paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)*

- Token-level sparse attention for the prefill stage, with BF16 KV cache
- Token-level sparse attention for the decoding stage,with BF16 / FP8 KV cache

**Dense Attention Kernels**

- Dense attention for the decoding stage with BF16 KV cache

**Attention Warp Interleave Kernels**

- Added an optimized BF16 Warp Interleave kernel for dense decode, sparse decode and sparse prefill on 810E and M890. The kernel partitions all warps into two warp groups and interleaves their execution to overlap softmax computation with QK/PV GEMM operations. 

#### Test & benchmark MLA decoding (Sparse & Dense):

```bash
# Dense:
python tests/test_flash_mla_dense_decoding.py
# Sparse:
python tests/test_flash_mla_sparse_decoding.py
python tests/test_flash_mla_sparse_prefill.py
# Compare test:
python bench_flash_mla.py --baseline torch --target flash_mla --compare
```

## PPU Backend Extensions and Optimizations

The PPU version includes targeted optimizations around data movement, shared storage layout, matrix computation, warp specialization, metadata scheduling, and compilation backend orchestration.

- **AIU + TSM Swizzle data path**: Tiled Q/K/V and KV cache data are routed through the AIU for asynchronous movement, and a swizzle layout is applied when writing to TSM / shared memory, so that data in shared storage aligns more closely with the operand read pattern of the subsequent MMA operations, reducing extra reshuffling and memory access conflicts.
- **PPU Tensor Cell mapping**: The QK and PV multiply-accumulate operations are organized as tiled MMA and mapped onto PPU Tensor Cell instructions, together with accumulator layout conversion, covering the FP16/BF16 compute paths as well as the BF16 compute path used after reading an FP8 KV cache.
- **Warp interleave / CrossCut scheduling**: The dense MLA decoding path selects CrossCut and warp interleave schemes according to sequence length, batch size, and device capability, advancing QK, softmax, PV, and output write-back in an interleaved manner across warp groups to improve parallelism and pipeline utilization during the decoding stage.
- **Tile and register residency tuning**: Block shape, warp count, stage count, MMA atom shape, and Q register residency strategy are selected for different `seqlen_q`, head counts, page block sizes, top-k values, and KV cache formats, matching the differing memory-access/compute ratios of dense decoding, sparse prefill, and sparse decoding.
- **Metadata + Split-KV load balancing**: Metadata precomputes the intra-batch tile partitioning and split count before execution, and the decoding kernel dispatches work according to this metadata, improving load balancing for long/short sequences, variable-length batches, top-k sparse attention, and multi-split scenarios.
- **Sparse / FP8 KV cache path optimizations**: The DSA decoding path supports token-level sparse attention, extra KV, attention sink, and top-k length scenarios; an FP8 KV cache is dequantized after being read and then computed in BF16, reducing KV cache bandwidth pressure.
- **Compilation options assisting backend orchestration**: The build parameters enable PPU/AIU support and configure backend optimization options such as register count, matrix address sinking, asynchronous address sinking, load/store address sinking, warpage, and virtual register reordering, helping the compiler better orchestrate memory access, register usage, and the compute pipeline.

## Requirements

- PPU SDK
- ZW 610 / 610E / 810 / 810E / M890
- PyTorch 2.0 or later (a Docker image matching the PPU SDK version is recommended).

## Installation

```bash
python setup.py install
```

## Usage

### MLA Decoding

To use the MLA decoding kernels, call get_mla_metadata once before the decoding loop to get the tile scheduler metadata. Then, call flash_mla_with_kvcache in each decoding step. For example:

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens,
    s_q * h_q // h_kv,
    h_kv,
    h_q,
    is_fp8,
    topk,
)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        is_causal, is_fp8_kvcache, indices,
    )
    ...
```

Where

- `s_q` is the number of q tokens per q sequence. If MTP (speculative decoding) is disabled, it should be 1.
- `h_kv` is the number of key-value heads.
- `h_q` is the number of query heads.

See `tests/test_flash_mla_dense_decoding.py` for a complete example.

**FP8 KV Cache:**
If `is_fp8_kvcache` is set to `True`, the kernel reads the KV cache in the "FP8 with scale" format (described below). It dequantizes the cache to bfloat16 and performs attention computation in bfloat16. The output is also in bfloat16.


**Sparse Attention (`indices` tensor):**
The `indices` tensor (if provided) enables token-level sparse attention by instructing the kernel to compute attention only for specified tokens.

-   **Shape:** `indices` should be a 3D tensor of shape `(batch_size, seq_len_q, topk)`.
-   **Format:** `indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * page_block_size + (the offset of token t within the page block)`, where `t` is the k-th token for the j-th query sequence in the i-th batch. Since the index of the page block has already been encoded into `indices_in_kvcache`, the kernel does not require the `block_table` parameter.
-   **Invalid entries:** Set invalid indices to `-1`.

**Return Values:**
The kernel returns `(out, lse)`, where:
-   `out` is the attention result.
-   `lse` is the log-sum-exp value of the attention scores for each query head.

See `tests/test_flash_mla_sparse_decoding.py` for a complete example.

### Sparse MLA Prefill

For the sparse MLA prefill kernel, call `flash_mla_sparse_fwd` directly with the following parameters:
-   `q`: Query tensor of shape `[s_q, h_q, d_qk]`
-   `kv`: Key-Value tensor of shape `[s_kv, h_kv, d_qk]`
-   `indices`: Indices tensor of shape `[s_q, h_kv, topk]`
-   `sm_scale`: A scalar value

**Note on batching:** This kernel does not support a batch dimension. For multi-batch inference, reshape the input tensors and adjust the `indices` parameter to simulate batch processing.

**Invalid indices:** Set invalid entries in `indices` to `-1` or any number `>= s_kv`.

**Return Values and Equivalent PyTorch Code:**
The kernel returns `(out, max_logits, lse)`. This is equivalent to the following PyTorch operations:

```python
Q: [s_q, h_q, d_qk], bfloat16
kv: [s_kv, h_kv, d_qk], bfloat16
indices: [s_q, h_kv, topk], int32

kv = kv.squeeze(1)  # [s_kv, d_qk], h_kv must be 1
indices = indices.squeeze(1)    # [s_q, topk]
focused_kv = kv[indices]    # For the i-th sequence (s_q), the corresponding KV tokens are selected from the KV cache based on indices[i, :]. This operation results in a tensor of shape [s_q, topk, d_qk].

P = (Q @ focused_kv.transpose(-1, -2)) * sm_scale * math.log2(math.e)    # [s_q, h_q, topk]
max_logits = P.max(dim=-1) # [s_q, h_q]
lse = log2sumexp2(P, dim=-1, base=2)   # [s_q, h_q], "log2sumexp2" means that the exponentiation and logarithm are base-2
S = exp2(P - lse)      # [s_q, h_q, topk]
out = S @ focused_kv  # [s_q, h_q, d_qk]

return (out, max_logits, lse)
```

See `tests/test_flash_mla_sparse_prefill.py` for a complete example.

## When you encounter issues
If you encounter bugs, please open a GitHub Issue!

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.
