#pragma once
#include <ATen/Tensor.h>
#include <cstdint>
#include <optional>

// Block-sparse MLA prefill forward (sm100, 192/128 non-absorbed / training path).
// Forks the dense MLA fwd; each q-block (TileShape Q = 256 rows) attends only its
// q2k-selected K-blocks (kv_block_size = 128). Emits O + LSE (natural-log) in the
// dense fwd convention so it feeds the matched 192/128 block-sparse KV-outer bwd.
// q: [S_q,H,192]; k: [S_k,H,192]; v: [S_k,H,128]; o: [S_q,H,128] (out);
// lse: [S_q,H] fp32 with stride(0)==1 (out); q2k_blocks: int32 [num_q_blocks, topk]
// (KV-block ids, -1 pad), SHARED across heads. Single sequence (B=1).
// attn_sink: optional per-head [H] fp32 gpt-oss attention sink, folded into the softmax denominator
// in-kernel (O + LSE sink-aware). std::nullopt disables (DSA/global block-sparse layers).
void block_sparse_prefill_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
                              at::Tensor lse, at::Tensor q2k_blocks, double softmax_scale,
                              std::optional<at::Tensor> attn_sink = std::nullopt);
