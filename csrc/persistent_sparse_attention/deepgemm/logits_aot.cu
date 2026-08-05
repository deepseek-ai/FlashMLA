#include "aot_launchers.h"

#include <deep_gemm/impls/sm100_mqa_logits.cuh>

namespace flash_mla::csa::deepgemm_adapters {
namespace {

constexpr uint32_t kThreads = 128 + 256;
using SharedStorage = deep_gemm::layout::MQALogitsSharedStorage<
    64,
    128,
    true,
    2,
    256,
    3,
    10,
    3,
    cutlass::float_e2m1_t>;
constexpr uint32_t kDynamicSharedBytes = sizeof(SharedStorage);

template <uint32_t kNumSMs>
cudaError_t launch(const LogitsLaunchParams& p, cudaStream_t stream) {
  auto kernel = &deep_gemm::sm100_mqa_logits<
      64,
      128,
      true,
      true,
      2,
      256,
      3,
      10,
      kNumSMs,
      128,
      256,
      cutlass::float_e2m1_t,
      float,
      float>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kDynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  kernel<<<kNumSMs, kThreads, kDynamicSharedBytes, stream>>>(
      p.num_q_tokens,
      p.num_kv_tokens,
      p.logits_stride,
      p.row_starts,
      p.row_ends,
      p.logits,
      p.q,
      p.sf_q,
      p.kv,
      p.sf_kv,
      p.weights);
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_index_logits_aot(
    const LogitsLaunchParams& params,
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
