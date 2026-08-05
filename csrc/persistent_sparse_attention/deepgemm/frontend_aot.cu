#include "aot_launchers.h"

#include <cuda_bf16.h>

#include <deep_gemm/impls/sm100_persistent_frontend_fused.cuh>

namespace flash_mla::csa::deepgemm_adapters {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kCluster = 2;
constexpr uint32_t kDynamicSharedBytes = 210732 + 256;

template <uint32_t kNumSMs>
using FrontendKernel = decltype(
    &deep_gemm::sm100_persistent_frontend_fused_impl<
        7168,
        2048,
        1536,
        512,
        448,
        64,
        65536,
        128,
        224,
        224,
        128,
        128,
        128,
        64,
        6,
        128,
        128,
        kNumSMs,
        cutlass::float_e4m3_t,
        cutlass::float_e4m3_t,
        cutlass::bfloat16_t>);

template <uint32_t kNumSMs>
cudaError_t launch(
    const FrontendLaunchParams& p,
    cudaStream_t stream) {
  FrontendKernel<kNumSMs> kernel =
      &deep_gemm::sm100_persistent_frontend_fused_impl<
          7168,
          2048,
          1536,
          512,
          448,
          64,
          65536,
          128,
          224,
          224,
          128,
          128,
          128,
          64,
          6,
          128,
          128,
          kNumSMs,
          cutlass::float_e4m3_t,
          cutlass::float_e4m3_t,
          cutlass::bfloat16_t>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kDynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = kCluster;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kNumSMs, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSharedBytes;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;

  return cudaLaunchKernelEx(
      &config,
      kernel,
      p.shape_m,
      p.p1_a,
      p.p1_sfa,
      p.p1_b,
      p.p1_sfb,
      p.p1_cd,
      p.p4_a,
      p.p4_sfa,
      p.p4_b,
      p.p4_sfb,
      p.p4_cd,
      static_cast<const __nv_bfloat16*>(p.hidden),
      p.hidden_stride,
      static_cast<uint8_t*>(p.hidden_fp8),
      p.hidden_fp8_stride,
      static_cast<uint32_t*>(p.hidden_sf),
      p.hidden_sf_stride_k,
      static_cast<const __nv_bfloat16*>(p.qkv),
      p.qkv_stride,
      static_cast<const __nv_bfloat16*>(p.q_weight),
      static_cast<const __nv_bfloat16*>(p.kv_weight),
      static_cast<uint8_t*>(p.q_norm_fp8),
      p.q_norm_fp8_stride,
      static_cast<uint32_t*>(p.q_norm_sf),
      p.q_norm_sf_stride_k,
      static_cast<__nv_bfloat16*>(p.kv_out),
      p.kv_out_stride,
      p.positions,
      p.cos_sin_cache,
      p.cos_sin_stride,
      p.rms_eps,
      p.grid_sync_counters,
      p.debug_m_tile_begin,
      p.debug_m_tile_end,
      p.debug_skip_mask);
}

}  // namespace

cudaError_t launch_persistent_frontend_aot(
    const FrontendLaunchParams& params,
    uint32_t num_sms,
    cudaStream_t stream) {
  switch (num_sms) {
    case 148:
      return launch<148>(params, stream);
    case 152:
      return launch<152>(params, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace flash_mla::csa::deepgemm_adapters
