# FlashMLA

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

> **重大变更说明（2026.09.30）**：在 2026.09.30 版本中，我们移除了对 Hopper 架构以及早期模型（包括 DeepSeek V3 / V3.2 / V4.0）的支持，并调整了 FP8 / FP4 KV cache 格式。因此该版本与先前版本不兼容。如果您需要运行这些旧模型或使用旧版 KV cache 格式，请切换至 [该 Commit](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7)。

## 简介

FlashMLA 是 DeepSeek 的高性能注意力算子库，支撑着 [DeepSeek-V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 模型在 NVIDIA GPU 与华为昇腾 NPU 上的极致推理。该代码库包含面向 Prefill 以及采用 FP8 / FP4 KV cache 的 Decoding 计算的 Token 级稀疏注意力（Sparse Attention）算子、一个深度融合了 Q-norm、Q-RoPE、核心 Attention、O-RoPE（共轭）以及 FP8 转换的高性能融合算子，以及用于 Prefill 和反向传播的稠密注意力算子。稀疏算子正是支撑 DeepSeek 稀疏注意力（DSA，详见[该论文](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)）的核心引擎。

## 最新动态

- **2026.09.30 发布昇腾平台注意力算子**：我们发布了面向华为昇腾 950 NPU 平台的稀疏注意力 Prefill 和 Decoding 算子，在 Prefill 阶段可达到最高 410 TFlops（硬件理论峰值的 95%），在 Decoding 阶段可达到最高 360 TFlops（硬件理论峰值的 83%）。此外，我们还发布了一篇[深度技术报告（英文版）](docs/20260930-ascend-prefill-deep-dive.md)与[技术报告（中文版）](docs/20260930-ascend-prefill-deep-dive-zh.md)，深入介绍了这个算子背后的算法原理与底层优化技术。

  我们还将融合算子（norm-RoPE-attn-RoPE-cast）在 Decoding 场景下的性能优化了约 10% - 15%。

  请注意，这个版本移除了对 NVIDIA Hopper 架构显卡与前代模型（包括 DeepSeek V3 / V3.2 / V4.0）的支持，并且调整了 KV cache 格式，因此与历史版本不兼容。如果您希望使用 Hopper 显卡、推理前代模型或者使用旧的 KV cache 格式，请切换至 [这个 Commit](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7)。

- **2026.09.10 发布 DeepSeek v4.1 注意力算子**：我们发布了面向 [DeepSeek-V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 的注意力算子，涵盖 Prefill 与 Decoding（支持 FP8 或 FP4 KV cache）。同时发布了[融合 norm-rope-attn-rope-cast 算子](#融合-norm--rope--attn--rope--cast-算子)，将 Q-norm（仅在 V4 中使用，V4.1 中未采用）、Q-RoPE、核心 Attention、O-RoPE（共轭）与 Cast-to-FP8 深度融为单个算子，在保持同等极致算力吞吐的同时省去了碎片化小算子的调度开销。
- **2025.09.29 发布稀疏注意力算子**：伴随 [DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) 正式发布，我们开源了对应的 Token 级稀疏注意力算子。这些算子强力驱动模型的 DeepSeek 稀疏注意力（DSA），在 Prefill 阶段达到最高 640 TFlops，在 Decoding 阶段达到最高 410 TFlops。我们还为全新的 FP8 稀疏解码算子发布了深度技术博客，欢迎[点击查阅](docs/20250929-hopper-fp8-sparse-deep-dive.md)。
- **2025.08.01 支持 SM100 架构 MHA 算子**：特别鸣谢 [NVIDIA 贡献的 PR](https://github.com/deepseek-ai/FlashMLA/pull/76)，实现了面向 SM100 的 MHA 前向与反向传播算子！
- **2025.04.22 深度技术博客**：我们非常高兴与大家分享全新 FlashMLA 算子背后的底层技术细节，欢迎查阅[深度技术文章](docs/20250422-new-kernel-deep-dive.md)。
- **2025.04.22 性能升级公告**：我们很高兴宣布 FlashMLA 新版本发布，针对计算密集型负载带来 5% ~ 15% 的性能提升，在 NVIDIA H800 SXM5 GPU 上实现了高达 660 TFlops 的算力吞吐。新版本接口完全向下兼容旧版本，升级即可直接获得性能提升！🚀🚀🚀

## 性能表现

### 测试与评测融合算子（norm RoPE attn RoPE cast，稀疏）：

```bash
python tests/test-fused-norm-rope-attn-rope-cast.py
```

运行此测试脚本需要依赖 [TileLang](https://github.com/tile-ai/tilelang)、[Tile-Kernels](https://github.com/deepseek-ai/TileKernels/) 以及 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。

该算子将 Q-norm（仅在 V4 中使用，V4.1 不使用）、Q-RoPE、核心 Attention、O-RoPE（共轭）以及转换到 FP8 的操作深度融合为单个 Kernel，有效避免了多个小算子的额外调度与访存开销。虽然融合了众多小操作，但在提前对 Q_b 和 Wv 权重进行重排的前提下，它依然能够达到相同甚至略高的 TFlops。在 B200 GPU 与 CUDA 13.3 环境下，Prefill 阶段可达到最高 1460 TFlops，Decoding 阶段可达到最高 950 TFlops。

该算子目前仅支持 CUDA 平台，暂不支持昇腾。

### 测试与评测 MLA Prefill（稀疏）：

```bash
python tests/test-sparse-prefill.py
```

在 B200 GPU 与 CUDA 13.3 下可达最高 1350 TFlops。在实际业务中，我们强烈推荐使用融合 norm-rope-attn-rope-cast 算子以获得更优性能。

在华为昇腾 950 NPU 上，可达最高 410 TFlops，达到硬件理论极限的 95%。

### 测试与评测 MLA Decoding（稀疏）：

```bash
python tests/test-sparse-decode.py
```

在 B200 GPU 与 CUDA 13.3 下可达最高 1024 TFlops。在实际业务中，我们强烈推荐使用融合 norm-rope-attn-rope-cast 算子以获得更优性能。

在华为昇腾 950 NPU 上，可达最高 360 TFlops，达到硬件理论极限的 83%。

### 测试与评测 MHA Prefill（稠密）：

```bash
python tests/test_fmha_sm100.py
```

根据 NVIDIA 官方测试报告，在 B200 上前向计算可达最高 1460 TFlops，反向计算可达最高 1000 TFlops。

该算子目前仅支持 CUDA 平台，暂不支持昇腾。

## 环境依赖

**NVIDIA 平台：**

- NVIDIA SM100 / SM103 GPU
- CUDA 13.1 及以上（推荐 CUDA 13.2 及以上，因为其提供了量化解码算子所需的原生 `cvt.rn.bf16x2` 转换指令；在 CUDA 13.1 下会回退到较慢的经由 FP32 扩宽转换）
- PyTorch 2.0 及以上

**华为昇腾平台：**

- 华为昇腾 950 NPU (Huawei Ascend 950)
- CANN 9.2.0 及以上（提供构建时所需的 `bisheng` 编译器与 Ascend C 头文件）
- `torch_npu`
- PyTorch 2.0 及以上

## 安装指南

```bash
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v . --no-build-isolation
```

必须添加 `--no-build-isolation` 参数：构建过程中 `setup.py` 会导入 `torch`（昇腾平台上还会导入 `torch_npu`），而本仓库未在 PEP 518 构建依赖中声明它们。

构建目标平台会自动检测（存在 `/dev/davinci_manager` 即判定为昇腾 Ascend，否则判定为 CUDA），可通过环境变量手动覆盖：`FLASH_MLA_BUILD_TARGET_PLATFORM=CUDA` 或 `FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND`。

在昇腾平台上，`ASCEND_HOME_PATH` 必须指向 CANN 安装的根目录（默认：`/usr/local/Ascend/ascend-toolkit/latest`）。

当 CUDA 扩展构建完成后，系统会对生成的 `.so` 文件执行寄存器溢出（register spilling）检查。CUTLASS FMHA 算子豁免此项检查，但如果本仓库定义的任何算子发生寄存器溢出，构建将报错终止并提示 "Register spilling detected. Build failed!"；可通过设置环境变量 `FLASH_MLA_SKIP_REG_SPILL_CHECK=1` 跳过该检查。

## 使用指南

### DeepSeek 稀疏注意力 (DSA) 解码 (Decoding)

使用 DSA 解码算子时，在进入解码循环前先调用一次 `get_mla_metadata` 以获取 Tile 调度元数据。随后在每一步解码中调用 `flash_mla_with_kvcache`。示例如下：

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata()    # 仅作为占位符。实际调度元数据在首次调用 flash_mla_with_kvcache 时生成

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        indices=indices,
    )
    ...
```

其中：

- `s_q` 为每个 query 序列中的 q token 数量。若禁用 MTP（多 Token 预测 / 投机推测解码），该值应为 1。
- `h_kv` 为 key-value 头数。
- `h_q` 为 query 头数。

**FP8 / FP4 KV 缓存格式**

对于解码阶段，该算子目前仅支持 FP8 和 FP4 格式的 KV 缓存，不支持非量化（bfloat16）格式。

对于 DeepSeek V4.1（`head_dim` = 512），格式由 `k_cache` 的最后一维（即每个 token 的字节数）自动识别：528 字节（V4.1）或 288 字节（V4.1 fp4）。在这两种格式中，每个 **token** 都会优先存储其量化后的原始数据，紧随其后存储缩放系数（scales）：

- **V4.1（每 token 528 字节）：** 前 512 字节为覆盖所有 512 个维度的 `float8_e4m3` 数值。64 维的 RoPE 同样经过量化，因此不存在 `bfloat16` 部分。最后 16 字节为 `float8_e8m0` 缩放系数，每 32 个连续的 `float8_e4m3` 数值对应一个缩放系数。
- **V4.1 fp4（每 token 288 字节）：** 前 256 字节包含 512 个 `e2m1` 数值，每个字节存放 2 个数值，偶数索引位于低 4 位（low nibble）。最后 32 字节为 `float8_e4m3` 缩放系数，每 16 个连续的 `e2m1` 数值对应一个缩放系数。该格式仅在 `k_cache` 采用 V4.1 格式时作为 `extra_k_cache` 有效。

量化与反量化细节请参阅 `tests/quant.py`。

**`indices` 张量**

`indices` 张量通过指示算子仅对指定的 Token 计算注意力，从而实现 Token 级稀疏注意力。

- **形状（Shape）：** `indices` 应为一个形状为 `(batch_size, seq_len_q, topk)` 的 3D 张量。
- **格式（Format）：** `indices_in_kvcache[i][j][k] = (token t 所在页块 page block 的索引) * page_block_size + (token t 在该页块内的偏移量)`，其中 `t` 为第 i 个批次中第 j 个 query 序列的第 k 个 token。由于页块索引已经编码进 `indices_in_kvcache`，算子不会实际使用 `block_table` 参数；但您仍需传入该参数，因为它是 `flash_mla_with_kvcache` 的必需位置参数（传入 `None` 即可）。
- **无效条目：** 将无效索引设为 `-1`。

**返回值**

算子返回 `(out, lse)`，其中：
- `out` 为注意力计算结果，形状为 `[batch_size, seq_len_q, h_q, head_dim_v]`，类型为 bfloat16。
- `lse` 为每个 query 头的注意力分数的 log-sum-exp 值，形状为 `[batch_size, h_q, seq_len_q]`，类型为 float32。其形状相对于 `out` 进行了转置。

完整示例请参阅 `tests/test-sparse-decode.py`。

### DeepSeek 稀疏注意力 (DSA) 预填充 (Prefill)

对于 DSA 预填充算子，直接调用 `flash_mla_sparse_fwd` 并传入以下参数：
- `q`：Query 张量，形状为 `[s_q, h_q, d_qk]`
- `kv`：Key-Value 张量，形状为 `[s_kv, h_kv, d_qk]`
- `indices`：索引张量，形状为 `[s_q, h_kv, topk]`
- `sm_scale`：标量缩放因子
- `d_v`：可选，Value 向量维度。仅能为 512（亦为默认值）。
- `attn_sink`：可选，`[h_q]`，float32。若提供，输出将额外按 `exp(lse) / (exp(lse) + exp(attn_sink))` 进行缩放。对返回的 `lse` 与 `max_logits` 无影响。
- `topk_length`：可选，`[s_q]`，int32。若提供，第 i 个 query token 仅会关注前 `topk_length[i]` 个索引。

**批处理注意事项：** 该算子不支持 Batch 维度。对于多 Batch 推理，请通过重排（reshape）输入张量并调整 `indices` 参数来模拟批处理过程。

**无效索引：** 将 `indices` 中的无效条目设为 `-1` 或任意 `>= s_kv` 的数值。

**返回值与等效 PyTorch 参考代码：**
算子返回 `(out, max_logits, lse)`，其中 `max_logits` 与 `lse` 均采用自然对数底。这等效于以下 PyTorch 操作：

```python
Q: [s_q, h_q, d_qk], bfloat16
kv: [s_kv, h_kv, d_qk], bfloat16
indices: [s_q, h_kv, topk], int32

kv = kv.squeeze(1)  # [s_kv, d_qk], h_kv 必须为 1
indices = indices.squeeze(1)    # [s_q, topk]
invalid = (indices < 0) | (indices >= s_kv)     # [s_q, topk], 算子会忽略这些条目
indices = indices.masked_fill(invalid, 0)       # 确保下方的 gather 操作在合法索引范围内
focused_kv = kv[indices]    # 对于第 i 个序列 (s_q)，根据 indices[i, :] 从 KV 缓存中选出对应的 KV token。该操作生成形状为 [s_q, topk, d_qk] 的张量。

P = (Q @ focused_kv.transpose(-1, -2)) * sm_scale    # [s_q, h_q, topk]
P = P.masked_fill(invalid.unsqueeze(1), float('-inf'))
max_logits = P.max(dim=-1).values   # [s_q, h_q]
lse = torch.logsumexp(P, dim=-1)    # [s_q, h_q]
S = torch.softmax(P, dim=-1)        # [s_q, h_q, topk]
out = S @ focused_kv  # [s_q, h_q, d_qk]

return (out, max_logits, lse)
```

在上述代码逻辑之外存在一个边界情况：对于没有任何有效索引的 query token，算子会为其返回 `max_logits = -inf`、`lse = +inf` 以及全零的输出；而上述伪代码则会产生 NaN 输出。

完整示例请参阅 `tests/test-sparse-prefill.py`。

### 稠密 MHA 预填充 (Dense MHA Prefill)

该算子实现了标准稠密多头注意力（MHA）的前向与反向传播操作。可通过以下接口调用：
- `flash_attn_varlen_func`
- `flash_attn_varlen_qkvpacked_func`
- `flash_attn_varlen_kvpacked_func`

用法与社区 `flash_attn` 库类似，存在两点差异：两种 packed 变体需要额外传入必需的 `head_dim_qk` 参数；并且三者均返回 `(out, lse)`，而非仅返回 `out`。完整示例请参阅 `tests/test_fmha_sm100.py`。

### 融合 norm + RoPE + attn + RoPE + cast 算子

在 DeepSeek-V4.1 发布中，我们还提供了一个融合算子，将 Q-norm（仅在 V4 中使用，V4.1 中未采用）、Q-RoPE、核心 Attention、O-RoPE（共轭）与 Cast-to-FP8 融为单个 Kernel。它彻底消除了在这些小算子上的额外调度开销，同时保持相同甚至略高的 TFlops，其前提是必须预先对 Q_b 和 Wv 权重进行布局重排。

在 DeepSeek-V4.1 的注意力机制中，Q（`[hidden_size]`）首先投影到 `[q_lora_rank]`（Q_a 投影），然后投影到 `[num_attention_heads, head_dim]`（Q_b 投影）。在核心 Attention 计算之后，输出（`[num_attention_heads, head_dim]`）被重排为 `[o_groups, num_attention_heads // o_groups * head_dim]`，其每一行被投影到 `[o_lora_rank]`（Wv 投影），得到一个 `[o_groups, o_lora_rank]` 的矩阵。该矩阵被重排为 `[o_groups * o_lora_rank]` 并最终投影到 `[hidden_size]`（Wo 投影）。该融合算子要求预先对 Q_b 与 Wv 权重进行重排。

**重排 Q_b 权重：**

```python
import torch
import tile_kernels
from flash_mla import fused_norm_rope_attn_rope_cast

h_q, d_q = 64, 512          # Q 头数与 Q 头维度
q_lora_rank = 1536
scale_gran = 128

# q_b_proj: [h_q * d_q, q_lora_rank], bfloat16
q_b_proj = torch.randn((h_q * d_q, q_lora_rank), dtype=torch.bfloat16, device='cuda')

# 将权重转换为 FP8，并使用 per-token 缩放系数，符合 DeepGEMM 内存布局
q_b_proj_fp8, q_b_sf = tile_kernels.quant.per_token_cast(
    q_b_proj, 'e4m3', scale_gran,
    use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True,
)

# 将权重及其缩放系数重排为融合算子所需的布局
q_b_proj_fp8, q_b_sf = fused_norm_rope_attn_rope_cast.permute_q_b_proj(
    (q_b_proj_fp8, q_b_sf), h_q, d_q,
)
```

**重排 Wv 权重：**

```python
import deep_gemm
import torch
import tile_kernels
from flash_mla import fused_norm_rope_attn_rope_cast

n_wv_group, wv_group_size, d_o = 8, 8, 512      # n_wv_group * wv_group_size == h_q
wv_proj_out_dim = 512                           # o_lora_rank
scale_gran = 32

# wv_proj: [n_wv_group * wv_proj_out_dim, wv_group_size * d_o], bfloat16
# (下方会将其 view 为 [n_wv_group, wv_proj_out_dim, wv_group_size * d_o])
wv_proj = torch.randn((n_wv_group * wv_proj_out_dim, wv_group_size * d_o),
                      dtype=torch.bfloat16, device='cuda')

# 将权重转换为 FP8，并将其缩放系数转换为 DeepGEMM einsum 所需的布局
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

# 将权重及其缩放系数重排为融合算子所需的布局
wv_proj_fp8, wv_sf = fused_norm_rope_attn_rope_cast.permute_wv_proj(
    (wv_proj_fp8, wv_sf), wv_group_size, d_o,
)
```

最后，调用融合算子：

```python
# q: [s_q, h_q, d_qk], bfloat16, 即使用上述重排权重计算得到的 Q_b 投影
out_fp8, out_sf, max_logits, lse = fused_norm_rope_attn_rope_cast.prefill(
    enable_q_norm,              # DeepSeek-V4.1 为 False
    rms_norm_eps,               # 例如 1e-4
    token_positions,            # [s_q], int32
    False, 64, cos_sin_cache,   # non-neox RoPE，rope_dim = 64
    n_wv_group,                 # h_q // wv_group_size
    32,                         # num_per_channels
    True, True, True,           # use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0
    q, kv, indices,             # bf16 Q, bf16 KV [s_kv, h_kv, d_qk], int32 indices [s_q, h_kv, topk]
    sm_scale=sm_scale,
    attn_sink=attn_sink,        # 可选, [h_q], float32
    topk_length=topk_length,    # 可选, [s_q], int32
)

# 对于解码阶段，调用 `decode` 接口并传入分页量化 KV cache：
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
    extra_k_cache=extra_k_cache,                        # 可选，布局与 k_cache 相同
    extra_indices_in_kvcache=extra_indices_in_kvcache,  # 可选, [s_q, extra_topk], int32
    extra_topk_length=extra_topk_length,                # 可选, [s_q], int32
)

# FP8 输出直接提供给 Wv 投影，使用上述重排后的 Wv 权重
wv_out = torch.empty((s_q, n_wv_group, wv_proj_out_dim), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_einsum("bhr,hdr->bhd", (out_fp8, out_sf), (wv_proj_fp8, wv_sf), wv_out, recipe=(1, 1, 32))
```

完整示例可参阅融合算子的测试脚本（[tests/test-fused-norm-rope-attn-rope-cast.py](tests/test-fused-norm-rope-attn-rope-cast.py)）。

## 致谢

FlashMLA 灵感源自 [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) 与 [cutlass](https://github.com/nvidia/cutlass) 项目。

## 社区硬件生态支持

### 沐曦 (MetaX)
针对沐曦 GPU，请访问官方网站：[MetaX](https://www.metax-tech.com)。

对应的 FlashMLA 适配版本请访问：[MetaX-MACA/FlashMLA](https://github.com/MetaX-MACA/FlashMLA)

### 摩尔线程 (Moore Threads)
针对摩尔线程 GPU，请访问官方网站：[Moore Threads](https://www.mthreads.com/)。

对应的 FlashMLA 适配版本可在 GitHub 获取：[MooreThreads/MT-flashMLA](https://github.com/MooreThreads/MT-flashMLA)。

### 海光 DCU (Hygon DCU)
针对海光 DCU，请访问官方网站：[Hygon Developer](https://developer.sourcefind.cn/)。

对应的 FlashMLA 适配版本可在开源社区获取：[OpenDAS/MLAttention](https://developer.sourcefind.cn/codes/OpenDAS/MLAttention)。

### 酷芯 / 智微智能 / 中科天玑 (Intellifusion)
针对 Intellifusion NNP 芯片，请访问官方网站：[Intellifusion](https://www.intellif.com)。

对应的 FlashMLA 适配版本可在 Gitee 获取：[Intellifusion/tyllm](https://gitee.com/Intellifusion_2025/tyllm/blob/master/python/tylang/flash_mla.py)。

### 天数智芯 (Iluvatar Corex)
针对天数智芯 GPU，请访问官方网站：[Iluvatar Corex](https://www.iluvatar.com)。

对应的 FlashMLA 适配版本可在 GitHub 获取：[Deep-Spark/FlashMLA](https://github.com/Deep-Spark/FlashMLA/tree/iluvatar_flashmla)。

### AMD Instinct
针对 AMD Instinct 系列 GPU，请访问官方网站：[AMD Instinct](https://www.amd.com/en/products/accelerators/instinct.html)。

对应的 FlashMLA 适配版本可在 GitHub 获取：[AITER/MLA](https://github.com/ROCm/aiter/blob/main/aiter/mla.py)。

## 论文引用

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient Multi-head Latent Attention Kernels},
      author={Jiashi Li and Shengyu Liu and Yuanhang Sun},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年10月15日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
