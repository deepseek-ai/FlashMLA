#include "aot_launchers.h"

#include <flash_mla/persistent_sparse_attention/wqb_ready.cuh>

namespace flash_mla::csa::deepgemm_adapters {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kCluster = 2;
constexpr uint32_t kDynamicSharedBytes = 210732;

template <uint32_t kNumSMs, typename Weight>
cudaError_t launch(const WqBLaunchParams& p, cudaStream_t stream) {
  constexpr uint32_t kWeightGranularity =
      cute::is_same_v<Weight, cutlass::float_e4m3_t> ? 128 : 32;
  auto kernel = &deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<
      cute::UMMA::Major::K,
      cute::UMMA::Major::K,
      128,
      kWeightGranularity,
      128,
      0,
      65536,
      1536,
      128,
      224,
      128,
      1,
      128,
      128,
      64,
      6,
      128,
      128,
      2,
      false,
      kNumSMs,
      false,
      true,
      deep_gemm::GemmType::Normal,
      false,
      cutlass::float_e4m3_t,
      Weight,
      cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity,
      true>;
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
      nullptr,
      p.shape_m,
      p.shape_n,
      p.shape_k,
      p.a,
      p.b,
      p.sfa,
      p.sfb,
      p.d,
      p.chunk_counters,
      p.chunk_ready,
      p.chunk_clocks,
      p.chunk_size,
      p.num_chunks,
      p.steady_sms,
      p.head_tiles);
}

template <uint32_t kNumSMs>
cudaError_t dispatch_weight(
    const WqBLaunchParams& params,
    bool fp4_weight,
    cudaStream_t stream) {
  if (fp4_weight) {
    return launch<kNumSMs, cutlass::detail::float_e2m1_unpacksmem_t>(
        params, stream);
  }
  return launch<kNumSMs, cutlass::float_e4m3_t>(params, stream);
}

}  // namespace

cudaError_t launch_wqb_ready_aot(
    const WqBLaunchParams& params,
    bool fp4_weight,
    uint32_t num_sms,
    cudaStream_t stream) {
  switch (num_sms) {
    case 148:
      return dispatch_weight<148>(params, fp4_weight, stream);
    case 152:
      return dispatch_weight<152>(params, fp4_weight, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace flash_mla::csa::deepgemm_adapters
