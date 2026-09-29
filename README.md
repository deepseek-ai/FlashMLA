# FlashMLA

> Breaking change notice (2026.09.30): In the 2026.09.30 release, we removed support for the Hopper architecture and for earlier models (including DeepSeek V3 / V3.2 / V4.0), and we changed the FP8 / FP4 KV cache format. This version is therefore not compatible with previous ones. If you need to run those models or use the old KV cache format, please switch to [this commit](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7).

## Introduction

FlashMLA is DeepSeek's library of optimized attention kernels, powering inference of the [DeepSeek-V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) model on NVIDIA GPUs and Huawei Ascend NPUs. This repo contains token-level sparse attention kernels for prefill and for decoding with an FP8 / FP4 KV cache, a fused kernel that combines Q-norm, Q-RoPE, attention, O-RoPE (conjugate) and the cast to FP8, and dense attention kernels for prefill and backward. The sparse kernels power DeepSeek Sparse Attention (DSA), as introduced in [this paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).

FlashMLA 是 DeepSeek 的高性能注意力算子库，支撑着 [DeepSeek-V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 模型在 NVIDIA GPU 与华为昇腾 NPU 上的推理。该仓库支持 prefill 与使用了 FP8 或 FP4 格式的 KV 缓存的 decoding 计算，以及一个高性能的融合了 Attention 前后的小操作的融合 kernel。此外，该仓库还提供了稠密 Attention 的 prefill 与反向传播算子。

## News

- **2026.09.30 Release of Ascend Attention Kernels**: We've released sparse attention prefill and decoding kernels for the Huawei Ascend 950 NPU, which achieve up to 410 TFlops (95% hardware peak) and 360 TFlops (83% of hardware peak) during prefill and decoding, respectively. We've also released [a deep-dive technical report](docs/20260930-ascend-prefill-deep-dive.md) on the algorithms and optimization techniques behind these kernels.

  我们发布了华为昇腾 NPU 平台上的稀疏注意力 Prefill 和 Decoding 算子，其在 Prefill 时能够达到 410 TFlops（95% 硬件极限），在 Decoding 时能达到 360 TFlops（83% 硬件极限）。此外，我们还发布了 [一篇简短的技术报告](docs/20260930-ascend-prefill-deep-dive-zh.md)，介绍了这个算子背后所使用到的算法与优化技术。

  We also optimize the performance of the fused norm-RoPE-attn-RoPE-cast kernel under decoding settings by around 10% - 15%.

  我们还将 fused norm-RoPE-attn-RoPE-cast kernel 的 decoding 的性能优化了 10% - 15%。

  Note that this release removes support for the Hopper architecture and for earlier models (including DeepSeek V3 / V3.2 / V4.0), and changes the FP8 / FP4 KV cache format, so it is not compatible with previous versions. If you need to run those models or use the old KV cache format, please switch to [this commit](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7).

  请注意，这个版本移除了对 NVIDIA Hopper 架构显卡与前代模型（包括 DeepSeek V3 / V3.2 / V4.0）的支持，并且改变了 KV cache 的格式。如果您希望使用 Hopper 显卡、推理前代模型或者使用旧的 KV cache 格式，请切换至 [这个 commit](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7)。
  
- **2026.09.10 Release of DeepSeek v4.1's Attention Kernels**: We've released attention kernels for [DeepSeek-V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), including both prefill and decoding (with FP8 or FP4 KV cache). We've also released a [fused-norm-rope-attn-rope-cast kernel](#fused-norm--rope--attn--rope--cast-kernel) which fuses Q-norm (only used in V4, not V4.1), Q-RoPE, core attention, O-RoPE (conjugate), and cast-to-fp8, while retaining the same performance.
- **2025.09.29 Release of Sparse Attention Kernels**: With the launch of [DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp), we are releasing the corresponding token-level sparse attention kernels. These kernels power the model's DeepSeek Sparse Attention (DSA) and achieve up to 640 TFlops during prefilling and 410 TFlops during decoding. We also release a deep-dive blog for our new FP8 sparse decoding kernel. Check it out [here](docs/20250929-hopper-fp8-sparse-deep-dive.md).
- **2025.08.01 Kernels for MHA on SM100**: Thanks to [NVIDIA's PR](https://github.com/deepseek-ai/FlashMLA/pull/76) for MHA forward / backward kernels on SM100!
- **2025.04.22 Deep-Dive Blog**: We'd love to share the technical details behind the new FlashMLA kernel! Check out our deep-dive write-up [here](docs/20250422-new-kernel-deep-dive.md).
- **2025.04.22 Performance Update**: We're excited to announce the new release of Flash MLA, which delivers 5% ~ 15% performance improvement for compute-bound workloads, achieving up to 660 TFlops on NVIDIA H800 SXM5 GPUs. The interface of the new version is fully compatible with the old one. Simply upgrade to the new version for an immediate performance boost! 🚀🚀🚀

## Performance

### Test & benchmark the fused norm RoPE attn RoPE cast kernel (Sparse):

```bash
python tests/test-fused-norm-rope-attn-rope-cast.py
```

[TileLang](https://github.com/tile-ai/tilelang), [Tile-Kernels](https://github.com/deepseek-ai/TileKernels/), and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) are required to run this test script.

This kernel fuses Q-norm (only used in V4, not V4.1), Q-RoPE, core attention, O-RoPE (conjugate) and the cast to FP8 into a single kernel, which saves the overhead of those small kernels. Although it fuses many small operations, it still achieves the same or even slightly higher TFlops, at the cost of having to permute the Q_b and Wv weights in advance. It achieves up to 1460 TFlops during prefill and 950 TFlops during decoding on B200 with CUDA 13.3.

This kernel currently supports CUDA only; Ascend is not supported.

### Test & benchmark MLA prefill (Sparse):

```bash
python tests/test-sparse-prefill.py
```

It achieves up to 1350 TFlops on B200 with CUDA 13.3. In practice, we highly recommend using the fused-norm-rope-attn-rope-cast kernel for better performance.

On the Huawei Ascend 950 NPU, it achieves up to 410 TFlops, which is 95% of the theoretical hardware peak.

### Test & benchmark MLA decoding (Sparse):

```bash
python tests/test-sparse-decode.py
```

It achieves up to 1024 TFlops on B200 with CUDA 13.3. In practice, we highly recommend using the fused-norm-rope-attn-rope-cast kernel for better performance.

On the Huawei Ascend 950 NPU, it achieves up to 360 TFlops, which is 83% of the theoretical hardware peak.

### Test & benchmark MHA prefill (Dense):

```bash
python tests/test_fmha_sm100.py
```

It achieves up to 1460 TFlops in forward and 1000 TFlops in backward computation on B200, as reported by NVIDIA.

This kernel currently supports CUDA only; Ascend is not supported.

## Requirements

For the NVIDIA platform:

- NVIDIA SM100 / SM103 GPU
- CUDA 13.1 and above (CUDA 13.2 and above is recommended, since it provides the native `cvt.rn.bf16x2` conversions that the quantized decoding kernels use; on CUDA 13.1 they fall back to a slower widening through FP32)
- PyTorch 2.0 and above

For the Huawei platform:

- Huawei Ascend 950 NPU
- CANN 9.2.0 and above (provides the `bisheng` compiler and the Ascend C headers used at build time)
- `torch_npu`
- PyTorch 2.0 and above

## Installation

```bash
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v . --no-build-isolation
```

`--no-build-isolation` is required: `setup.py` imports `torch` (and `torch_npu` on the Ascend platform) while building, and this repository does not declare them as PEP 518 build requirements.

The build target platform is detected automatically (`/dev/davinci_manager` means Ascend, anything else means CUDA) and can be overridden with `FLASH_MLA_BUILD_TARGET_PLATFORM=CUDA` or `FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND`.

On the Ascend platform, `ASCEND_HOME_PATH` must point to the root of the CANN installation (default: `/usr/local/Ascend/ascend-toolkit/latest`).

After the CUDA extension is built, a register-spill check runs over the produced `.so`. The CUTLASS FMHA kernels are exempt, but if any kernel defined by this repository spills registers the build fails with "Register spilling detected. Build failed!"; export `FLASH_MLA_SKIP_REG_SPILL_CHECK=1` to skip the check.

## Usage

### DeepSeek Sparse Attention (DSA) Decoding

To use the DSA decoding kernels, call get_mla_metadata once before the decoding loop to get the tile scheduler metadata. Then, call flash_mla_with_kvcache in each decoding step. For example:

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata()    # A placeholder only. The actual scheduling metadata is generated on the first call to flash_mla_with_kvcache

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        indices=indices,
    )
    ...
```

Where

- `s_q` is the number of q tokens per q sequence. If MTP (speculative decoding) is disabled, it should be 1.
- `h_kv` is the number of key-value heads.
- `h_q` is the number of query heads.

**FP8 / FP4 KV Cache Format**

For decoding, this kernel currently supports only the FP8 and FP4 KV cache formats. Unquantized (bfloat16) KV cache format is not supported.

For DeepSeek V4.1 (`head_dim` = 512), the format is detected from the last dimension of `k_cache` (i.e. the bytes per token): 528 (V4.1) or 288 (V4.1 fp4). In both formats, each **token** stores its quantized raw data first, followed immediately by its scales:

- **V4.1 (528 bytes per token):** the first 512 bytes are `float8_e4m3` values covering all 512 dimensions. The 64 RoPE dimensions are quantized as well, so there is no `bfloat16` part. The last 16 bytes are `float8_e8m0` scales, one for every 32 consecutive `float8_e4m3` values.
- **V4.1 fp4 (288 bytes per token):** the first 256 bytes contain 512 `e2m1` values, 2 values per byte, the even-indexed one in the low nibble. The last 32 bytes are `float8_e4m3` scales, one for every 16 consecutive `e2m1` values. This format is only valid for `extra_k_cache` when `k_cache` is in the V4.1 format.

See `tests/quant.py` for quantization and dequantization details.

**`indices` Tensor**

The `indices` tensor enables token-level sparse attention by instructing the kernel to compute attention only for specified tokens.

- **Shape:** `indices` should be a 3D tensor of shape `(batch_size, seq_len_q, topk)`.
- **Format:** `indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * page_block_size + (the offset of token t within the page block)`, where `t` is the k-th token for the j-th query sequence in the i-th batch. Since the index of the page block has already been encoded into `indices_in_kvcache`, the kernel does not use the `block_table` parameter; you still have to pass it, since it is a required positional argument of `flash_mla_with_kvcache` (`None` is fine).
- **Invalid entries:** Set invalid indices to `-1`.

**Return Values**

The kernel returns `(out, lse)`, where:
- `out` is the attention result, `[batch_size, seq_len_q, h_q, head_dim_v]`, bfloat16.
- `lse` is the log-sum-exp value of the attention scores for each query head, `[batch_size, h_q, seq_len_q]`, float32. Its shape is transposed with respect to `out`.

See `tests/test-sparse-decode.py` for complete examples.

### DeepSeek Sparse Attention (DSA) Prefill

For the DSA prefill kernel, call `flash_mla_sparse_fwd` directly with the following parameters:
- `q`: Query tensor of shape `[s_q, h_q, d_qk]`
- `kv`: Key-Value tensor of shape `[s_kv, h_kv, d_qk]`
- `indices`: Indices tensor of shape `[s_q, h_kv, topk]`
- `sm_scale`: A scalar value
- `d_v`: Optional, the dimension of the value vectors. Can only be 512, which is also the default.
- `attn_sink`: Optional, `[h_q]`, float32. If provided, the output is additionally scaled by `exp(lse) / (exp(lse) + exp(attn_sink))`. It has no effect on the returned `lse` and `max_logits`.
- `topk_length`: Optional, `[s_q]`, int32. If provided, the i-th query token only attends to the first `topk_length[i]` indices.

**Note on batching:** This kernel does not support a batch dimension. For multi-batch inference, reshape the input tensors and adjust the `indices` parameter to simulate batch processing.

**Invalid indices:** Set invalid entries in `indices` to `-1` or any number `>= s_kv`.

**Return Values and Equivalent PyTorch Code:**
The kernel returns `(out, max_logits, lse)`, where `max_logits` and `lse` are in the natural logarithm. This is equivalent to the following PyTorch operations:

```python
Q: [s_q, h_q, d_qk], bfloat16
kv: [s_kv, h_kv, d_qk], bfloat16
indices: [s_q, h_kv, topk], int32

kv = kv.squeeze(1)  # [s_kv, d_qk], h_kv must be 1
indices = indices.squeeze(1)    # [s_q, topk]
invalid = (indices < 0) | (indices >= s_kv)     # [s_q, topk], the kernel ignores these entries
indices = indices.masked_fill(invalid, 0)       # So that the gather below stays in range
focused_kv = kv[indices]    # For the i-th sequence (s_q), the corresponding KV tokens are selected from the KV cache based on indices[i, :]. This operation results in a tensor of shape [s_q, topk, d_qk].

P = (Q @ focused_kv.transpose(-1, -2)) * sm_scale    # [s_q, h_q, topk]
P = P.masked_fill(invalid.unsqueeze(1), float('-inf'))
max_logits = P.max(dim=-1).values   # [s_q, h_q]
lse = torch.logsumexp(P, dim=-1)    # [s_q, h_q]
S = torch.softmax(P, dim=-1)        # [s_q, h_q, topk]
out = S @ focused_kv  # [s_q, h_q, d_qk]

return (out, max_logits, lse)
```

A query token that has no valid index is an edge case on top of the code above: the kernel returns `max_logits = -inf`, `lse = +inf` and an all-zero output for it, whereas the pseudo-code would produce a NaN output.

See `tests/test-sparse-prefill.py` for a complete example.

### Dense MHA Prefill

This kernel implements the standard dense Multi-Head Attention (MHA) forward and backward operations. It can be called using:
- `flash_attn_varlen_func`
- `flash_attn_varlen_qkvpacked_func`
- `flash_attn_varlen_kvpacked_func`

The usage is similar to the `flash_attn` package, with two differences: the two packed variants take an additional required `head_dim_qk` argument, and all three return `(out, lse)` instead of only `out`. See `tests/test_fmha_sm100.py` for a complete example of `flash_attn_varlen_func`.

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

# wv_proj: [n_wv_group * wv_proj_out_dim, wv_group_size * d_o], bfloat16
# (it is viewed as [n_wv_group, wv_proj_out_dim, wv_group_size * d_o] further below)
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

You may refer to the fused kernel's test script ([tests/test-fused-norm-rope-attn-rope-cast.py](tests/test-fused-norm-rope-attn-rope-cast.py)) for a complete example.

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
      author={Jiashi Li and Shengyu Liu and Yuanhang Sun},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
