#pragma once

#include <ATen/Tensor.h>

void FMHACutlassSM100FwdRun(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
                            at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                            at::Tensor o, at::Tensor lse,
                            int64_t mask_mode_code, double softmax_scale, int64_t max_seqlen_q, int64_t max_seqlen_kv, bool is_varlen);

void FMHACutlassSM100BwdRun(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                            at::Tensor v, at::Tensor o, at::Tensor lse,
                            at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                            at::Tensor dq, at::Tensor dk, at::Tensor dv,
                            int64_t mask_mode_code, double softmax_scale, int64_t max_seqlen_q, int64_t max_seqlen_kv, bool is_varlen);
