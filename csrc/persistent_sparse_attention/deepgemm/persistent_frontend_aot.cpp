// SPDX-License-Identifier: MIT
// Persistent P0/P1/F1'/P4 frontend over fixed SM100 AOT instantiations.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <initializer_list>

#include "aot_launchers.h"
#include "tma_utils.h"

namespace flash_mla::csa::deepgemm_adapters {
namespace {

constexpr int64_t kTokens = 8192;
constexpr int64_t kHidden = 7168;
constexpr int64_t kQKV = 2048;
constexpr int64_t kQRank = 1536;
constexpr int64_t kKV = 512;
constexpr int64_t kQuery = 65536;

void check_frontend_same_device(
    const torch::Tensor& first,
    std::initializer_list<const torch::Tensor*> rest) {
  for (const auto* tensor : rest) {
    TORCH_CHECK(tensor->device() == first.device(),
                "all frontend tensors must share one CUDA device");
  }
}

void check_matrix(
    const torch::Tensor& tensor,
    c10::ScalarType dtype,
    int64_t rows,
    int64_t columns,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == dtype &&
                  tensor.sizes() == torch::IntArrayRef({rows, columns}) &&
                  tensor.is_contiguous(),
              name,
              " has the wrong shape, dtype, or layout");
}

void check_scale(
    const torch::Tensor& tensor,
    int64_t rows,
    int64_t columns,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda() &&
                  tensor.scalar_type() == torch::kInt32 &&
                  tensor.sizes() == torch::IntArrayRef({rows, columns}) &&
                  tensor.stride(0) == 1 &&
                  tensor.stride(1) == tma::aligned_size(rows, 4),
              name,
              " must be MN-major packed UE8M0 int32");
}

}  // namespace

void persistent_attention_frontend(
    const torch::Tensor& hidden,
    const torch::Tensor& hidden_fp8,
    const torch::Tensor& hidden_sf,
    const std::pair<torch::Tensor, torch::Tensor>& w_qkv_a,
    const torch::Tensor& qkv,
    const torch::Tensor& q_weight,
    const torch::Tensor& kv_weight,
    const torch::Tensor& q_norm_fp8,
    const torch::Tensor& q_norm_sf,
    const torch::Tensor& kv_out,
    const std::pair<torch::Tensor, torch::Tensor>& w_q_b,
    const torch::Tensor& q_2d,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& workspace,
    float rms_eps,
    int64_t num_sms) {
  check_matrix(hidden, torch::kBFloat16, kTokens, kHidden, "hidden");
  check_matrix(hidden_fp8, torch::kFloat8_e4m3fn,
               kTokens, kHidden, "hidden_fp8");
  check_scale(hidden_sf, kTokens, 14, "hidden_sf");
  check_matrix(w_qkv_a.first, torch::kFloat8_e4m3fn,
               kQKV, kHidden, "w_qkv_a.values");
  check_scale(w_qkv_a.second, kQKV, 14, "w_qkv_a.scales");
  check_matrix(qkv, torch::kBFloat16, kTokens, kQKV, "qkv");
  TORCH_CHECK(q_weight.is_cuda() &&
                  q_weight.scalar_type() == torch::kBFloat16 &&
                  q_weight.sizes() == torch::IntArrayRef({kQRank}) &&
                  q_weight.is_contiguous(),
              "q_weight must be contiguous BF16 [1536]");
  TORCH_CHECK(kv_weight.is_cuda() &&
                  kv_weight.scalar_type() == torch::kBFloat16 &&
                  kv_weight.sizes() == torch::IntArrayRef({kKV}) &&
                  kv_weight.is_contiguous(),
              "kv_weight must be contiguous BF16 [512]");
  check_matrix(q_norm_fp8, torch::kFloat8_e4m3fn,
               kTokens, kQRank, "q_norm_fp8");
  check_scale(q_norm_sf, kTokens, 3, "q_norm_sf");
  TORCH_CHECK(kv_out.is_cuda() &&
                  kv_out.scalar_type() == torch::kBFloat16 &&
                  kv_out.sizes() == torch::IntArrayRef({kTokens, 1, kKV}) &&
                  kv_out.is_contiguous(),
              "kv_out must be contiguous BF16 [8192,1,512]");
  check_matrix(w_q_b.first, torch::kFloat8_e4m3fn,
               kQuery, kQRank, "w_q_b.values");
  check_scale(w_q_b.second, kQuery, 3, "w_q_b.scales");
  check_matrix(q_2d, torch::kBFloat16, kTokens, kQuery, "q_2d");
  TORCH_CHECK(positions.is_cuda() &&
                  positions.scalar_type() == torch::kInt64 &&
                  positions.sizes() == torch::IntArrayRef({kTokens}) &&
                  positions.is_contiguous(),
              "positions must be contiguous int64 [8192]");
  TORCH_CHECK(cos_sin_cache.is_cuda() &&
                  cos_sin_cache.scalar_type() == torch::kFloat32 &&
                  cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.size(1) >= 64 &&
                  cos_sin_cache.stride(1) == 1,
              "cos_sin_cache must have contiguous float32 rows of width >=64");
  TORCH_CHECK(workspace.is_cuda() &&
                  workspace.scalar_type() == torch::kInt32 &&
                  workspace.is_contiguous(),
              "workspace must be contiguous CUDA int32");
  check_frontend_same_device(
      hidden,
      {&hidden_fp8, &hidden_sf, &w_qkv_a.first, &w_qkv_a.second, &qkv,
       &q_weight, &kv_weight, &q_norm_fp8, &q_norm_sf, &kv_out,
       &w_q_b.first, &w_q_b.second, &q_2d, &positions, &cos_sin_cache,
       &workspace});

  c10::cuda::CUDAGuard guard(hidden.device());
  const auto* properties =
      at::cuda::getDeviceProperties(hidden.get_device());
  if (num_sms == 0) {
    num_sms = properties->multiProcessorCount;
  }
  TORCH_CHECK(properties->major == 10 &&
                  (properties->multiProcessorCount == 148 ||
                   properties->multiProcessorCount == 152) &&
                  num_sms == properties->multiProcessorCount,
              "AOT frontend supports only the full 148-SM B200 or "
              "152-SM GB200 grid");
  TORCH_CHECK(workspace.numel() >= num_sms,
              "frontend workspace must provide one counter per CTA");

  constexpr int kBlockM = 128;
  constexpr int kBlockN = 224;
  constexpr int kBlockK = 128;
  constexpr int kLoadM = 128;
  constexpr int kLoadN = 112;
  constexpr int kStoreM = 128;
  constexpr int kStoreN = 224;
  constexpr int kSwizzleAB = 128;
  constexpr int kSwizzleD = 64;

  const FrontendLaunchParams params{
      static_cast<uint32_t>(kTokens),
      tma::make_a(hidden_fp8, tma::Element::kFloat8E4M3,
                  kTokens, kHidden, kLoadM, kBlockK,
                  hidden_fp8.stride(0), kSwizzleAB),
      tma::make_scale(hidden_sf, kTokens, kHidden, kBlockM, 128),
      tma::make_b(w_qkv_a.first, tma::Element::kFloat8E4M3,
                  kQKV, kHidden, kLoadN, kBlockK,
                  w_qkv_a.first.stride(0), kSwizzleAB),
      tma::make_scale(w_qkv_a.second, kQKV, kHidden, kBlockN, 128),
      tma::make_d(qkv, tma::Element::kBFloat16,
                  kTokens, kQKV, kStoreM, kStoreN,
                  qkv.stride(0), kSwizzleD),
      tma::make_a(q_norm_fp8, tma::Element::kFloat8E4M3,
                  kTokens, kQRank, kLoadM, kBlockK,
                  q_norm_fp8.stride(0), kSwizzleAB),
      tma::make_scale(q_norm_sf, kTokens, kQRank, kBlockM, 128),
      tma::make_b(w_q_b.first, tma::Element::kFloat8E4M3,
                  kQuery, kQRank, kLoadN, kBlockK,
                  w_q_b.first.stride(0), kSwizzleAB),
      tma::make_scale(w_q_b.second, kQuery, kQRank, kBlockN, 128),
      tma::make_d(q_2d, tma::Element::kBFloat16,
                  kTokens, kQuery, kStoreM, kStoreN,
                  q_2d.stride(0), kSwizzleD),
      hidden.const_data_ptr(),
      static_cast<uint32_t>(hidden.stride(0)),
      hidden_fp8.data_ptr(),
      static_cast<uint32_t>(hidden_fp8.stride(0)),
      hidden_sf.data_ptr(),
      static_cast<uint32_t>(hidden_sf.stride(1)),
      qkv.const_data_ptr(),
      static_cast<uint32_t>(qkv.stride(0)),
      q_weight.const_data_ptr(),
      kv_weight.const_data_ptr(),
      q_norm_fp8.data_ptr(),
      static_cast<uint32_t>(q_norm_fp8.stride(0)),
      q_norm_sf.data_ptr(),
      static_cast<uint32_t>(q_norm_sf.stride(1)),
      kv_out.data_ptr(),
      static_cast<uint32_t>(kv_out.stride(0)),
      positions.const_data_ptr<int64_t>(),
      cos_sin_cache.const_data_ptr<float>(),
      static_cast<uint32_t>(cos_sin_cache.stride(0)),
      rms_eps,
      reinterpret_cast<uint32_t*>(workspace.data_ptr<int32_t>()),
      0,
      0,
      0,
  };
  const cudaError_t status = launch_persistent_frontend_aot(
      params,
      static_cast<uint32_t>(num_sms),
      at::cuda::getCurrentCUDAStream(hidden.get_device()).stream());
  TORCH_CHECK(status == cudaSuccess,
              "AOT frontend launch failed: ", cudaGetErrorString(status));
}

}  // namespace flash_mla::csa::deepgemm_adapters
