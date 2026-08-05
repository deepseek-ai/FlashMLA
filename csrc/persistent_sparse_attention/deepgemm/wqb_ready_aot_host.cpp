// SPDX-License-Identifier: MIT
// Caller-owned WqB projection with AOT readiness publication.

#include "wqb_ready.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "aot_launchers.h"
#include "tma_utils.h"

namespace flash_mla::csa::deepgemm_adapters {
namespace {

constexpr int64_t kAcceptedM = 8192;
constexpr int64_t kAcceptedN = 65536;
constexpr int64_t kAcceptedK = 1536;

void check_wqb_same_device(
    const torch::Tensor& first,
    std::initializer_list<const torch::Tensor*> rest) {
  for (const auto* tensor : rest) {
    TORCH_CHECK(tensor->device() == first.device(),
                "all WqB tensors must share one CUDA device");
  }
}

void check_scale(
    const torch::Tensor& scale,
    int64_t mn,
    int64_t k,
    int64_t granularity_k,
    const char* name) {
  const int64_t columns =
      (k + granularity_k * 4 - 1) / (granularity_k * 4);
  TORCH_CHECK(scale.is_cuda() &&
                  scale.scalar_type() == torch::kInt32 &&
                  scale.dim() == 2 && scale.size(0) == mn &&
                  scale.size(1) == columns && scale.stride(0) == 1 &&
                  scale.stride(1) == tma::aligned_size(mn, 4),
              name,
              " must be MN-major packed UE8M0 int32 [",
              mn,
              ",",
              columns,
              "]");
}

void wqb_ready_common(
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
    int64_t num_sms,
    bool fp4_weight) {
  TORCH_CHECK(values.is_cuda() &&
                  values.scalar_type() == torch::kFloat8_e4m3fn &&
                  values.dim() == 2 && values.is_contiguous(),
              "values must be contiguous FP8 E4M3 [M,K]");
  const int64_t m = values.size(0);
  const int64_t k = values.size(1);
  TORCH_CHECK(weight.is_cuda() && weight.dim() == 2 &&
                  weight.is_contiguous(),
              "weight must be a contiguous two-dimensional CUDA tensor");
  TORCH_CHECK(
      fp4_weight
          ? (weight.scalar_type() == torch::kInt8 &&
             weight.size(1) * 2 == k)
          : (weight.scalar_type() == torch::kFloat8_e4m3fn &&
             weight.size(1) == k),
      fp4_weight
          ? "MXFP4 weight must be packed int8 [N,K/2]"
          : "FP8 weight must be E4M3 [N,K]");
  const int64_t n = weight.size(0);
  TORCH_CHECK(m == kAcceptedM && n == kAcceptedN && k == kAcceptedK,
              "AOT WqB accepts only [8192,1536] @ [65536,1536].T");
  check_scale(scales, m, k, 128, "scales");
  check_scale(weight_scales, n, k, fp4_weight ? 32 : 128,
              "weight_scales");
  TORCH_CHECK(output.is_cuda() &&
                  output.scalar_type() == torch::kBFloat16 &&
                  output.sizes() == torch::IntArrayRef({m, n}) &&
                  output.is_contiguous(),
              "output must be caller-owned contiguous BF16 [M,N]");
  TORCH_CHECK(chunk_size == 128,
              "accepted WqB chunk_size is 128");
  TORCH_CHECK(head_tiles == 4688,
              "accepted WqB head_tiles is 4688");
  const int64_t chunks = (m + chunk_size - 1) / chunk_size;
  TORCH_CHECK(counters.is_cuda() && ready.is_cuda() && clocks.is_cuda() &&
                  counters.scalar_type() == torch::kInt32 &&
                  ready.scalar_type() == torch::kInt32 &&
                  clocks.scalar_type() == torch::kInt64 &&
                  counters.is_contiguous() && ready.is_contiguous() &&
                  clocks.is_contiguous() && counters.numel() >= chunks &&
                  ready.numel() >= chunks && clocks.numel() >= chunks + 1,
              "counters/ready/clocks must cover at least N, N, N+1 entries");
  check_wqb_same_device(values, {&scales, &weight, &weight_scales, &output,
                             &counters, &ready, &clocks});

  c10::cuda::CUDAGuard guard(values.device());
  const auto* properties =
      at::cuda::getDeviceProperties(values.get_device());
  if (num_sms == 0) {
    num_sms = properties->multiProcessorCount;
  }
  TORCH_CHECK(properties->major == 10 &&
                  (properties->multiProcessorCount == 148 ||
                   properties->multiProcessorCount == 152) &&
                  num_sms == properties->multiProcessorCount,
              "AOT WqB supports only the full 148-SM B200 or "
              "152-SM GB200 grid");
  if (steady_sms == 0) {
    steady_sms = num_sms == 148 ? 36 : 38;
  }
  TORCH_CHECK(steady_sms > 0 && steady_sms < num_sms &&
                  steady_sms % 2 == 0,
              "steady_sms must be an even value strictly inside the grid");

  const auto a_map = tma::make_a(
      values, tma::Element::kFloat8E4M3,
      m, k, 128, 128, values.stride(0), 128);
  const auto b_map = tma::make_b(
      weight,
      fp4_weight ? tma::Element::kPackedFloat4E2M1
                 : tma::Element::kFloat8E4M3,
      n, k, 112, 128, weight.stride(0), 128);
  const auto d_map = tma::make_d(
      output, tma::Element::kBFloat16,
      m, n, 128, 224, output.stride(0), 64);
  const auto sfa_map = tma::make_scale(scales, m, k, 128, 128);
  const auto sfb_map = tma::make_scale(
      weight_scales, n, k, 224, fp4_weight ? 32 : 128);
  const WqBLaunchParams params{
      static_cast<uint32_t>(m),
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(k),
      a_map,
      b_map,
      sfa_map,
      sfb_map,
      d_map,
      reinterpret_cast<uint32_t*>(counters.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(ready.data_ptr<int32_t>()),
      reinterpret_cast<uint64_t*>(clocks.data_ptr<int64_t>()),
      static_cast<uint32_t>(chunk_size),
      static_cast<uint32_t>(chunks),
      static_cast<uint32_t>(steady_sms),
      static_cast<uint32_t>(head_tiles),
  };
  const cudaError_t status = launch_wqb_ready_aot(
      params,
      fp4_weight,
      static_cast<uint32_t>(num_sms),
      at::cuda::getCurrentCUDAStream(values.get_device()).stream());
  TORCH_CHECK(status == cudaSuccess,
              "AOT WqB launch failed: ", cudaGetErrorString(status));
}

}  // namespace

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
    int64_t num_sms) {
  wqb_ready_common(
      values, scales, weight, weight_scales, output,
      counters, ready, clocks, chunk_size, steady_sms,
      head_tiles, num_sms, true);
}

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
    int64_t num_sms) {
  wqb_ready_common(
      values, packed_scales, weight, packed_weight_scales, output,
      counters, ready, clocks, chunk_size, steady_sms,
      head_tiles, num_sms, false);
}

}  // namespace flash_mla::csa::deepgemm_adapters
