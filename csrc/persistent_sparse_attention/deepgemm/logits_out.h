#pragma once

#include <cstdint>

#include <torch/extension.h>

namespace flash_mla::csa::deepgemm_adapters {

void deepgemm_fp8_fp4_mqa_logits_out(
    const torch::Tensor& q_fp,
    const torch::Tensor& q_scale,
    const torch::Tensor& kv_fp,
    const torch::Tensor& kv_scale,
    const torch::Tensor& weights,
    const torch::Tensor& row_starts,
    const torch::Tensor& row_ends,
    const torch::Tensor& logits_out,
    int64_t num_sms);

}  // namespace flash_mla::csa::deepgemm_adapters
