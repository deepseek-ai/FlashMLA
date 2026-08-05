#pragma once

#include <torch/extension.h>

namespace flash_mla::csa::deepgemm_adapters {

void wqb_ready_out(
    const torch::Tensor& values,
    const torch::Tensor& scales,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scales,
    const torch::Tensor& output,
    const torch::Tensor& counters,
    const torch::Tensor& ready,
    const torch::Tensor& clocks,
    int64_t chunk_size,
    int64_t steady_sms,
    int64_t head_tiles,
    int64_t num_sms);

void wqb_ready_packed_out(
    const torch::Tensor& values,
    const torch::Tensor& packed_scales,
    const torch::Tensor& weight,
    const torch::Tensor& packed_weight_scales,
    const torch::Tensor& output,
    const torch::Tensor& counters,
    const torch::Tensor& ready,
    const torch::Tensor& clocks,
    int64_t chunk_size,
    int64_t steady_sms,
    int64_t head_tiles,
    int64_t num_sms);

}  // namespace flash_mla::csa::deepgemm_adapters
