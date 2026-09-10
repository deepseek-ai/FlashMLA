# FlashMLA

## Introduction

FlashMLA is DeepSeek's library of optimized attention kernels, powering the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) and [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) models. This repository contains the following implementations:

**Sparse Attention Kernels**

*These kernels power DeepSeek Sparse Attention (DSA), as introduced in [this paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).*

- Token-level sparse attention for the prefill stage
- Token-level sparse attention for the decoding stage, with FP8 KV cache

**Dense Attention Kernels**

- Dense attention for the prefill stage
- Dense attention for the decoding stage

## News

- **2026.09.10 Release of DeepSeek v4.1's Attention Kernels**: We've released attention kernels for [DeepSeek-V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), including both prefill and decoding (with FP8 or FP4 KV cache). We've also released a [fused-norm-rope-attn-rope-cast kernel](#fused-norm--rope--attn--rope--cast-kernel) which fuses Q-norm (only used in V4, not V4.1), Q-RoPE, core attention, O-RoPE (conjugate), and cast-to-fp8, while retaining the same performance.
- **2025.09.29 Release of Sparse Attention Kernels**: With the launch of [DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp), we are releasing the corresponding token-level sparse attention kernels. These kernels power the model's DeepSeek Sparse Attention (DSA) and achieve up to 640 TFlops during prefilling and 410 TFlops during decoding. We also release a deep-dive blog for our new FP8 sparse decoding kernel. Check it out [here](docs/20250929-hopper-fp8-sparse-deep-dive.md).
- **2025.08.01 Kernels for MHA on SM100**: Thanks to [NVIDIA's PR](https://github.com/deepseek-ai/FlashMLA/pull/76) for MHA forward / backward kernels on SM100!
- **2025.04.22 Deep-Dive Blog**: We'd love to share the technical details behind the new FlashMLA kernel! Check out our deep-dive write-up [here](docs/20250422-new-kernel-deep-dive.md).
- **2025.04.22 Performance Update**: We're excited to announce the new release of Flash MLA, which delivers 5% ~ 15% performance improvement for compute-bound workloads, achieving up to 660 TFlops on NVIDIA H800 SXM5 GPUs. The interface of the new version is fully compatible with the old one. Simply upgrade to the new version for an immediate performance boost! 🚀🚀🚀

## Performance

#### Test & benchmark MLA decoding (Sparse & Dense):

```bash
python tests/test_flash_mla_dense_decoding.py
python tests/test_flash_mla_sparse_decoding.py
```

The dense MLA decoding kernel achieves up to 3000 GB/s in memory-bound configuration and 660 TFLOPS in computation-bound configuration on H800 SXM5 with CUDA 12.8. The token-level sparse MLA decoding kernel (which uses an FP8 KV cache while performing the matrix multiplication in bfloat16) achieves 410 TFLOPS in compute-bound configuration on H800 SXM5 with CUDA 12.8, and achieves up to 700 TFlops on B200.

#### Test & benchmark MHA prefill (Dense):

```bash
python tests/test_fmha_sm100.py
```

It achieves up to 1460 TFlops in forward and 1000 TFlops in backward computation on B200, as reported by NVIDIA.

#### Test & benchmark MLA prefill (Sparse):

```bash
python tests/test_flash_mla_sparse_prefill.py
```

It achieves up to 640 TFlops in forward computation on H800 SXM5 with CUDA 12.8, and achieves up to 1450 TFlops on B200, CUDA 12.9.

#### Test & benchmark the fused norm RoPE attn RoPE cast kernel (Sparse):

```bash
python tests/test_fused_norm_rope_attn_rope_cast.py
```

[TileLang](https://github.com/tile-ai/tilelang), [Tile-Kernels](https://github.com/deepseek-ai/TileKernels/), and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) are required for running this test script.

This kernel fuses Q-norm (only used in V4, not V4.1), Q-RoPE, core attention, O-RoPE (conjugate) and cast-to-FP8 into a single kernel, saving times for those small kernels. Although it fuses many small operations, this kernel still keeps the same or even slightly higher TFlops (at the cost of having to permute the Q_b and Wv weights in advance). It achieves up to 1430 TFlops during prefill and 670 TFlops during decoding on B200.


## Requirements

- SM90 / SM100 (See the support matrix below)
- CUDA 12.8 and above (CUDA 12.9+ is required for SM100 kernels)
- PyTorch 2.0 and above

Support matrix:

| Kernel | GPU Architecture | MLA Mode [1] | Supported Models |
| :---: | :---: | :---: | :---: |
| Dense Decoding | SM90 | MQA | DeepSeek V3 / V3.1 |
| Sparse Decoding | SM90 & SM100 | MQA | DeepSeek V3.2 / V4 / V4.1 [2] |
| Dense Prefill | SM100 | MHA | DeepSeek V3 / V3.1 / V3.2 |
| Sparse Prefill | SM90 & SM100 | MQA | DeepSeek V3.2 / V4 / V4.1 |
| Fused Norm RoPE Attn RoPE Cast | SM100 | MQA | DeepSeek V4 / V4.1 |

[1]: Here "MLA Mode" refers to the mode used for MLA calculation. MQA stands for Multi-Query Attention mode (i.e. `head_dim_k` = 576 (for DeepSeek V3/V3.1/V3.2) or 512 (for DeepSeek V4/V4.1) with `head_dim_v` = 512), while MHA stands for Multi-Head Attention mode (i.e. `head_dim_k` = 192 / 128 with `head_dim_v` = 128). For a detailed explanation of these modes, please refer to the appendix of [DeepSeek V3.2's Paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).

[2] Sparse Decoding for DeepSeek V4.1 is only available on SM100

## Installation

```bash
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v .
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

**FP8 KV Cache:**
If `is_fp8_kvcache` is set to `True`, the kernel reads the KV cache in the "FP8 with scale" format (described below). It dequantizes the cache to bfloat16 and performs attention computation in bfloat16. The output is also in bfloat16. In this repository, `is_fp8_kvcache=True` is only supported together with `indices` (i.e. sparse attention); the dense decoding kernel reads bf16 / fp16 KV caches. 

In the "FP8 with scale" format for DeepSeek V3.2 (`head_dim` = 576, sparse attention), a page block is `page_block_size` token-major rows of 656 Bytes each:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512 `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values. The first `float32` is the scale for the first 128 `float8_e4m3` values, the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This part is not quantized for accuracy.

For DeepSeek V4 / V4.1 (`head_dim` = 512), the format is detected from the last dimension of `k_cache` (i.e. the bytes per token): 584 (V4), 528 (V4.1) or 288 (V4.1 fp4). In all three, a page block stores `page_block_size` data rows first and `page_block_size` scale rows afterwards:
-   **V4**: 584 Bytes per token. The data row is 448 Bytes of quantized NoPE (`float8_e4m3`) followed by 128 Bytes, i.e. the 64 `bfloat16` RoPE values (not quantized). The scale row is 8 Bytes, of which the first 7 are `float8_e8m0` scales and the 8th byte is padding; each scale covers 64 consecutive `float8_e4m3` values of the NoPE part.
-   **V4.1**: 528 Bytes per token. The data row is 512 Bytes of `float8_e4m3`, i.e. the 64 RoPE dimensions are quantized as well and there is no `bfloat16` part. The scale row is 16 Bytes of `float8_e8m0`, each scale covering 32 consecutive `float8_e4m3` values.
-   **V4.1 fp4**: 288 Bytes per token. The data row is 256 Bytes containing 512 `e2m1` values, 2 values per byte (the even-indexed one in the low nibble). The scale row is 32 Bytes of `float8_e4m3`, each scale covering 16 consecutive `e2m1` values. This format is only valid for `extra_k_cache`, and only when `k_cache` is in the V4.1 format; otherwise `extra_k_cache` must have the same format as `k_cache`. In pratice we expect the sliding window (SWA) kv cache to be in FP8 and the compress attention (CA) kv cache to be in FP4.

See `tests/quant.py` for quantization and dequantization details.

**Sparse Attention (`indices` tensor):**
The `indices` tensor (if provided) enables token-level sparse attention by instructing the kernel to compute attention only for specified tokens.

-   **Shape:** `indices` should be a 3D tensor of shape `(batch_size, seq_len_q, topk)`.
-   **Format:** `indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * page_block_size + (the offset of token t within the page block)`, where `t` is the k-th token for the j-th query sequence in the i-th batch. Since the index of the page block has already been encoded into `indices_in_kvcache`, the kernel does not require the `block_table` parameter.
-   **Invalid entries:** Set invalid indices to `-1`.

**Return Values:**
The kernel returns `(out, lse)`, where:
-   `out` is the attention result.
-   `lse` is the log-sum-exp value of the attention scores for each query head.

See `tests/test_flash_mla_dense_decoding.py` and `tests/test_flash_mla_sparse_decoding.py` for complete examples.

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
lse = log2sumexp2(P, dim=-1, base=2)   # [s_q, h_q]，"log2sumexp2" means that the exponentiation and logarithm are base-2
S = exp2(P - lse)      # [s_q, h_q, topk]
out = S @ focused_kv  # [s_q, h_q, d_qk]

return (out, max_logits, lse)
```

See `tests/test_flash_mla_sparse_prefill.py` for a complete example.

### Dense MHA Prefill

This kernel implements the standard dense Multi-Head Attention (MHA) forward and backward operations. It can be called using:
-   `flash_attn_varlen_func`
-   `flash_attn_varlen_qkvpacked_func`
-   `flash_attn_varlen_kvpacked_func`

The usage is similar to the `flash_attn` package. See `tests/test_fmha_sm100.py` for a complete example.

### Fused norm + RoPE + attn + RoPE + cast kernel

In the DeepSeek-V4.1 release, we also provide a fused kernel that combines Q-norm (only used in V4, not in V4.1), Q-RoPE, core attention, O-RoPE (conjugate) and the cast to FP8 into a single kernel. It removes the extra time spent on these small kernels while keeping the same or even slightly higher TFlops, at the cost of having to permute the Q_b and Wv weights in advance.

In DeepSeek-V4.1 attention, Q (`[hidden_size]`) is first projected to `[q_lora_rank]` (the Q_a projection) and then to `[num_attention_heads, head_dim]` (the Q_b projection). After core attention, the output (`[num_attention_heads, head_dim]`) is reshaped to `[o_groups, num_attention_heads // o_groups * head_dim]`, and each of its rows is projected to `[o_lora_rank]` (the Wv projection), giving an `[o_groups, o_lora_rank]` matrix. That matrix is reshaped to `[o_groups * o_lora_rank]` and finally projected to `[hidden_size]` (the Wo projection). This kernel requires the Q_b and Wv weights to be permuted.

To permute the Q_b weight:

```python
import torch
import tile_kernels
from flash_mla import fused_norm_rope_attn_rope_cast

h_q, d_q = 64, 512          # Q heads and Q head dimension
q_lora_rank = 1536
scale_gran = 128

# q_b_proj: [h_q * d_q, q_lora_rank], bfloat16
q_b_proj = torch.randn((h_q * d_q, q_lora_rank), dtype=torch.bfloat16, device='cuda')

# Quantize the weight to FP8 with per-token scale factors, in DeepGEMM's layout
q_b_proj_fp8, q_b_sf = tile_kernels.quant.per_token_cast(
    q_b_proj, 'e4m3', scale_gran,
    use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True,
)

# Permute the weight and its scale factors into the layout required by the fused kernel
q_b_proj_fp8, q_b_sf = fused_norm_rope_attn_rope_cast.permute_q_b_proj(
    (q_b_proj_fp8, q_b_sf), h_q, d_q,
)
```

To permute the Wv weight:

```python
import deep_gemm
import torch
import tile_kernels
from flash_mla import fused_norm_rope_attn_rope_cast

n_wv_group, wv_group_size, d_o = 8, 8, 512      # n_wv_group * wv_group_size == h_q
wv_proj_out_dim = 512                           # o_lora_rank
scale_gran = 32

# wv_proj: [n_wv_group, wv_proj_out_dim, wv_group_size * d_o], bfloat16
wv_proj = torch.randn((n_wv_group * wv_proj_out_dim, wv_group_size * d_o),
                      dtype=torch.bfloat16, device='cuda')

# Quantize the weight to FP8, and put its scale factors into the layout that DeepGEMM's einsum expects
wv_proj_fp8, wv_sf = tile_kernels.quant.per_token_cast(
    wv_proj, 'e4m3', scale_gran,
    use_tma_aligned_col_major_sf=False, round_sf=True, use_packed_ue8m0=False,
)
wv_sf = deep_gemm.transform_sf_into_required_layout(
    wv_sf.view(n_wv_group, wv_proj_out_dim, wv_group_size * d_o // scale_gran),
    wv_proj_out_dim, wv_group_size * d_o,
    num_groups=n_wv_group, recipe=(1, 1, scale_gran), is_sfa=False,
)
wv_proj_fp8 = wv_proj_fp8.view(n_wv_group, wv_proj_out_dim, wv_group_size * d_o)

# Permute the weight and its scale factors into the layout required by the fused kernel
wv_proj_fp8, wv_sf = fused_norm_rope_attn_rope_cast.permute_wv_proj(
    (wv_proj_fp8, wv_sf), wv_group_size, d_o,
)
```

And finally, to use the fused kernel:

```python
# q: [s_q, h_q, d_qk], bfloat16, i.e. the Q_b projection computed with the permuted weight above
out_fp8, out_sf, max_logits, lse = fused_norm_rope_attn_rope_cast.prefill(
    enable_q_norm,              # False for DeepSeek-V4.1
    rms_norm_eps,               # e.g. 1e-4
    token_positions,            # [s_q], int32
    False, 64, cos_sin_cache,   # non-neox RoPE with rope_dim = 64
    n_wv_group,                 # h_q // wv_group_size
    32,                         # num_per_channels
    True, True, True,           # use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0
    q, kv, indices,             # bf16 Q, bf16 KV [s_kv, h_kv, d_qk], int32 indices [s_q, h_kv, topk]
    sm_scale=sm_scale,
    attn_sink=attn_sink,        # optional, [h_q], float32
    topk_length=topk_length,    # optional, [s_q], int32
)

# For decoding, call `decode` instead, passing the paged quantized KV cache:
#   q: [s_q, h_q, d_qk], bf16
#   k_cache: [num_blocks, page_block_size, h_kv, bytes_per_token], fp8_e4m3
#   indices_in_kvcache: [s_q, topk], int32
out_fp8, out_sf, lse = fused_norm_rope_attn_rope_cast.decode(
    enable_q_norm, rms_norm_eps,
    token_positions, False, 64, cos_sin_cache,
    n_wv_group, 32, True, True, True,
    q, k_cache, indices_in_kvcache,
    sm_scale=sm_scale,
    attn_sink=attn_sink,
    topk_length=topk_length,
    extra_k_cache=extra_k_cache,                        # optional, same layout as k_cache
    extra_indices_in_kvcache=extra_indices_in_kvcache,  # optional, [s_q, extra_topk], int32
    extra_topk_length=extra_topk_length,                # optional, [s_q], int32
)

# The FP8 output is consumed directly by the Wv projection, using the permuted Wv weight
wv_out = torch.empty((s_q, n_wv_group, wv_proj_out_dim), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_einsum("bhr,hdr->bhd", (out_fp8, out_sf), (wv_proj_fp8, wv_sf), wv_out, recipe=(1, 1, 32))
```

You may refer to the fused kernel's test script ([tests/test_fused_norm_rope_attn_rope_cast.py](tests/test_fused_norm_rope_attn_rope_cast.py)) for a complete example.

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.

## Community Support

### MetaX
For MetaX GPUs, visit the official website: [MetaX](https://www.metax-tech.com).

The corresponding FlashMLA version can be found at: [MetaX-MACA/FlashMLA](https://github.com/MetaX-MACA/FlashMLA)


### Moore Threads
For the Moore Threads GPU, visit the official website: [Moore Threads](https://www.mthreads.com/).

The corresponding FlashMLA version is available on GitHub: [MooreThreads/MT-flashMLA](https://github.com/MooreThreads/MT-flashMLA).


### Hygon DCU
For the Hygon DCU, visit the official website: [Hygon Developer](https://developer.sourcefind.cn/).

The corresponding FlashMLA version is available here: [OpenDAS/MLAttention](https://developer.sourcefind.cn/codes/OpenDAS/MLAttention).


### Intellifusion
For the Intellifusion NNP, visit the official website: [Intellifusion](https://www.intellif.com).

The corresponding FlashMLA version is available on Gitee: [Intellifusion/tyllm](https://gitee.com/Intellifusion_2025/tyllm/blob/master/python/tylang/flash_mla.py).


### Iluvatar Corex
For Iluvatar Corex GPUs, visit the official website: [Iluvatar Corex](https://www.iluvatar.com).

The corresponding FlashMLA version is available on GitHub: [Deep-Spark/FlashMLA](https://github.com/Deep-Spark/FlashMLA/tree/iluvatar_flashmla)


### AMD Instinct
For AMD Instinct GPUs, visit the official website: [AMD Instinct](https://www.amd.com/en/products/accelerators/instinct.html).

The corresponding FlashMLA version can be found at: [AITER/MLA](https://github.com/ROCm/aiter/blob/main/aiter/mla.py)

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient Multi-head Latent Attention Kernels},
      author={Jiashi Li, Shengyu Liu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
