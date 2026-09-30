#pragma once

#include <torch/csrc/stable/tensor.h>

void FMHACutlassSM100FwdRun(torch::stable::Tensor workspace_buffer, torch::stable::Tensor q, torch::stable::Tensor k, torch::stable::Tensor v,
                            torch::stable::Tensor cumulative_seqlen_q, torch::stable::Tensor cumulative_seqlen_kv,
                            torch::stable::Tensor o, torch::stable::Tensor lse,
                            int64_t mask_mode_code, double softmax_scale, int64_t max_seqlen_q, int64_t max_seqlen_kv, bool is_varlen);

void FMHACutlassSM100BwdRun(torch::stable::Tensor workspace_buffer, torch::stable::Tensor d_o, torch::stable::Tensor q, torch::stable::Tensor k,
                            torch::stable::Tensor v, torch::stable::Tensor o, torch::stable::Tensor lse,
                            torch::stable::Tensor cumulative_seqlen_q, torch::stable::Tensor cumulative_seqlen_kv,
                            torch::stable::Tensor dq, torch::stable::Tensor dk, torch::stable::Tensor dv,
                            int64_t mask_mode_code, double softmax_scale, int64_t max_seqlen_q, int64_t max_seqlen_kv, bool is_varlen);
