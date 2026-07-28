#pragma once

#include <ATen/Tensor.h>
#include <optional>

void FMHACutlassSM100FwdRun(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
                            at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                            at::Tensor o, at::Tensor lse,
                            int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen,
                            int window_size = -1, std::optional<at::Tensor> attn_sink = std::nullopt);

void FMHACutlassSM100BwdRun(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                            at::Tensor v, at::Tensor o, at::Tensor lse,
                            at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                            at::Tensor dq, at::Tensor dk, at::Tensor dv,
                            int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen,
                            int window_size = -1, std::optional<at::Tensor> attn_sink = std::nullopt,
                            std::optional<at::Tensor> d_sink = std::nullopt);
