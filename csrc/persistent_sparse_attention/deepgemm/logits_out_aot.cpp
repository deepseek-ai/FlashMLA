// SPDX-License-Identifier: MIT
// Caller-owned MXFP4 logits adapter over fixed SM100 AOT instantiations.

#include "logits_out.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

#include "aot_launchers.h"
#include "tma_utils.h"

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
    int64_t num_sms) {
  constexpr int64_t kHeads = 64;
  constexpr int64_t kPhysicalHeadBytes = 64;
  constexpr int64_t kLogicalHeadDim = 128;
  constexpr int64_t kSplitKv = 256;
  constexpr int64_t kBlockQ = 2;

  TORCH_CHECK(q_fp.is_cuda() && q_fp.scalar_type() == torch::kInt8 &&
                  q_fp.dim() == 3 && q_fp.size(1) == kHeads &&
                  q_fp.size(2) == kPhysicalHeadBytes && q_fp.is_contiguous(),
              "q_fp must be contiguous packed-MXFP4 int8 [rows,64,64]");
  const int64_t rows = q_fp.size(0);
  TORCH_CHECK(q_scale.is_cuda() && q_scale.scalar_type() == torch::kInt32 &&
                  q_scale.sizes() == torch::IntArrayRef({rows, kHeads}) &&
                  q_scale.is_contiguous(),
              "q_scale must be contiguous int32 [rows,64]");
  TORCH_CHECK(kv_fp.is_cuda() && kv_fp.scalar_type() == torch::kInt8 &&
                  kv_fp.dim() == 2 && kv_fp.size(1) == kPhysicalHeadBytes &&
                  kv_fp.is_contiguous(),
              "kv_fp must be contiguous packed-MXFP4 int8 [kv_rows,64]");
  const int64_t kv_rows = kv_fp.size(0);
  TORCH_CHECK(kv_scale.is_cuda() &&
                  kv_scale.scalar_type() == torch::kInt32 &&
                  kv_scale.sizes() == torch::IntArrayRef({kv_rows}) &&
                  kv_scale.is_contiguous(),
              "kv_scale must be contiguous int32 [kv_rows]");
  TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat32 &&
                  weights.sizes() == torch::IntArrayRef({rows, kHeads}) &&
                  weights.stride(1) == 1,
              "weights must be float32 [rows,64] with contiguous heads");
  for (const auto* metadata : {&row_starts, &row_ends}) {
    TORCH_CHECK(metadata->is_cuda() &&
                    metadata->scalar_type() == torch::kInt32 &&
                    metadata->sizes() == torch::IntArrayRef({rows}) &&
                    metadata->is_contiguous(),
                "row start/end metadata must be contiguous int32 [rows]");
  }
  TORCH_CHECK(logits_out.is_cuda() &&
                  logits_out.scalar_type() == torch::kFloat32 &&
                  logits_out.dim() == 2 && logits_out.size(0) == rows &&
                  logits_out.size(1) > 0 && logits_out.stride(1) == 1,
              "logits_out must be float32 [rows,max_seqlen_k]");
  TORCH_CHECK(logits_out.stride(0) % 256 == 0,
              "logits_out row stride must be 1024-byte aligned");
  TORCH_CHECK(rows <= std::numeric_limits<uint32_t>::max() &&
                  kv_rows <= std::numeric_limits<uint32_t>::max() &&
                  logits_out.stride(0) <=
                      std::numeric_limits<uint32_t>::max(),
              "logits dimensions exceed uint32");
  for (const auto* tensor : {&q_scale, &kv_fp, &kv_scale, &weights,
                             &row_starts, &row_ends, &logits_out}) {
    TORCH_CHECK(tensor->device() == q_fp.device(),
                "all logits tensors must share one CUDA device");
  }

  c10::cuda::CUDAGuard guard(q_fp.device());
  const auto* properties =
      at::cuda::getDeviceProperties(q_fp.get_device());
  if (num_sms == 0) {
    num_sms = properties->multiProcessorCount;
  }
  // Both grids are instantiated (see `logits_aot.cu`), so a caller may ask
  // for either as long as it fits the device. Pinning `num_sms` to the full
  // device grid was over-strict: a persistent kernel deliberately sized below
  // the device leaves SMs for a concurrent node, which is what the CSA
  // schedule does -- its GB200 profile runs this at 148 on a 152-SM part.
  TORCH_CHECK(properties->major == 10 &&
                  (properties->multiProcessorCount == 148 ||
                   properties->multiProcessorCount == 152) &&
                  (num_sms == 148 || num_sms == 152) &&
                  num_sms <= properties->multiProcessorCount,
              "AOT logits supports a 148- or 152-SM grid, not larger than the "
              "device: got ", num_sms, " on ",
              properties->multiProcessorCount, " SMs");

  const auto q_map = tma::make_2d(
      q_fp, tma::Element::kPackedFloat4E2M1,
      kLogicalHeadDim, rows * kHeads,
      kLogicalHeadDim, kBlockQ * kHeads,
      q_fp.stride(1), kPhysicalHeadBytes, false);
  const auto kv_map = tma::make_2d(
      kv_fp, tma::Element::kPackedFloat4E2M1,
      kLogicalHeadDim, kv_rows,
      kLogicalHeadDim, kSplitKv,
      kv_fp.stride(0), kPhysicalHeadBytes, false);
  const auto sf_kv_map = tma::make_2d(
      kv_scale, tma::Element::kInt32,
      tma::aligned_size(kv_rows, 4), 1,
      kSplitKv, 1, 0, 0);
  const auto sf_q_map = tma::make_2d(
      q_scale, tma::Element::kInt32,
      kHeads, rows, kHeads, kBlockQ,
      q_scale.stride(0), 0);
  const auto weights_map = tma::make_2d(
      weights, tma::Element::kFloat32,
      kHeads, rows, kHeads, kBlockQ,
      weights.stride(0), 0);
  const LogitsLaunchParams params{
      static_cast<uint32_t>(rows),
      static_cast<uint32_t>(kv_rows),
      static_cast<uint32_t>(logits_out.stride(0)),
      reinterpret_cast<const uint32_t*>(row_starts.const_data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(row_ends.const_data_ptr<int32_t>()),
      logits_out.data_ptr<float>(),
      q_map,
      sf_q_map,
      kv_map,
      sf_kv_map,
      weights_map,
  };
  const cudaError_t status = launch_index_logits_aot(
      params,
      static_cast<uint32_t>(num_sms),
      at::cuda::getCurrentCUDAStream(q_fp.get_device()).stream());
  TORCH_CHECK(status == cudaSuccess,
              "AOT logits launch failed: ", cudaGetErrorString(status));
}

}  // namespace flash_mla::csa::deepgemm_adapters
