#pragma once
#include <ATen/Tensor.h>
#include <cstdint>
#include <optional>

// Block-sparse KV-outer MLA prefill backward (sm100). Builds the k2q CSR from
// per-Q-block selection then runs Sm100BlockSparseBwd (192/128, TileShape
// <64,128,192,128>, CausalForBackwardMask). q,k,v,o,d_o: [S,H,D]; lse: [S,H]
// with stride(0)==1; q2k_blocks: int32 [num_q_blocks, topk] (KV-block ids,
// -1 pad). dq,dk,dv outputs. Single sequence (B=1).
// attn_sink: optional per-head [H] fp32 gpt-oss sink (its d_sink gradient -> d_sink output, [H] fp32,
// pre-zeroed). dQ/dK/dV are sink-correct via the sink-aware LSE from the fwd. std::nullopt disables.
void block_sparse_prefill_bwd(at::Tensor workspace, at::Tensor d_o, at::Tensor q, at::Tensor k,
                              at::Tensor v, at::Tensor o, at::Tensor lse, at::Tensor q2k_blocks,
                              at::Tensor dq, at::Tensor dk, at::Tensor dv,
                              double softmax_scale, int64_t window_size,
                              int64_t kv_block_size, int64_t q_block_size,
                              std::optional<at::Tensor> attn_sink = std::nullopt,
                              std::optional<at::Tensor> d_sink = std::nullopt);

int64_t block_sparse_bwd_workspace_size(int64_t Q, int64_t H, int64_t B, int64_t D);
