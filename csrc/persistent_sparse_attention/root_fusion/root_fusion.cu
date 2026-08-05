#include "root_fusion.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "root_fusion.cuh"

namespace {

constexpr int kTokens = 8192;
constexpr int kHidden = 7168;
constexpr int kProjectionWidth = 2048;
constexpr int kIndexWidth = 512;
constexpr int kIndexWeightWidth = 64;
constexpr int kThreads = 512;
constexpr int kCluster = 2;
constexpr int kDynamicSharedBytes = 230188;

cudaError_t encode_tiled(
    CUtensorMap* descriptor,
    CUtensorMapDataType dtype,
    void* pointer,
    int element_size,
    int global_inner,
    int global_outer,
    int shared_inner,
    int shared_outer,
    int global_outer_stride,
    int swizzle_bytes) {
  if (swizzle_bytes != 0) {
    shared_inner = swizzle_bytes / element_size;
  }
  const cuuint64_t global_dimensions[2] = {
      static_cast<cuuint64_t>(global_inner),
      static_cast<cuuint64_t>(global_outer)};
  const cuuint64_t global_strides[1] = {
      static_cast<cuuint64_t>(global_outer_stride) *
      static_cast<cuuint64_t>(element_size)};
  const cuuint32_t box_dimensions[2] = {
      static_cast<cuuint32_t>(shared_inner),
      static_cast<cuuint32_t>(shared_outer)};
  const cuuint32_t element_strides[2] = {1, 1};
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  if (swizzle_bytes == 32) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  } else if (swizzle_bytes == 64) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (swizzle_bytes == 128) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  } else if (swizzle_bytes != 0) {
    return cudaErrorInvalidValue;
  }
  const CUresult status = cuTensorMapEncodeTiled(
      descriptor,
      dtype,
      2,
      pointer,
      global_dimensions,
      global_strides,
      box_dimensions,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return status == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

template <typename... Args>
cudaError_t encode_all(Args... statuses) {
  const cudaError_t values[] = {statuses...};
  for (const cudaError_t status : values) {
    if (status != cudaSuccess) {
      return status;
    }
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t launch_root_fusion(
    const RootFusionParams& params,
    uint32_t grid_sms,
    uint32_t projection_sms,
    cudaStream_t stream) {
  CUtensorMap projection_a;
  CUtensorMap projection_b;
  CUtensorMap projection_scale_a;
  CUtensorMap projection_scale_b;
  CUtensorMap projection_d;
  CUtensorMap index_a;
  CUtensorMap index_b;
  CUtensorMap index_d;
  CUtensorMap weight_a;
  CUtensorMap weight_b;
  CUtensorMap weight_d;

  const cudaError_t descriptor_status = encode_all(
      encode_tiled(
          &projection_a, CU_TENSOR_MAP_DATA_TYPE_UINT8,
          params.hidden_fp8, 1,
          kHidden, kTokens, 128, 112, kHidden, 128),
      encode_tiled(
          &projection_b, CU_TENSOR_MAP_DATA_TYPE_UINT8,
          const_cast<void*>(params.fused_wqa_wkv), 1,
          kHidden, kProjectionWidth, 128, 128, kHidden, 128),
      encode_tiled(
          &projection_scale_a, CU_TENSOR_MAP_DATA_TYPE_INT32,
          params.hidden_scale, 4,
          kTokens, kHidden / 512, 224, 1,
          params.hidden_scale_stride_k, 0),
      encode_tiled(
          &projection_scale_b, CU_TENSOR_MAP_DATA_TYPE_INT32,
          const_cast<uint32_t*>(params.fused_wqa_wkv_scale), 4,
          kProjectionWidth, kHidden / 512, 128, 1,
          params.fused_wqa_wkv_scale_stride_k, 0),
      encode_tiled(
          &projection_d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          params.qkv, 2,
          kProjectionWidth, kTokens, 128, 16,
          kProjectionWidth, 128),
      encode_tiled(
          &index_a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          const_cast<void*>(params.hidden), 2,
          kHidden, kTokens, 64, 128, kHidden, 128),
      encode_tiled(
          &index_b, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          const_cast<void*>(params.index_compressor_weight), 2,
          kHidden, kIndexWidth, 64, 64, kHidden, 128),
      encode_tiled(
          &index_d, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
          params.index_kv_score, 4,
          kIndexWidth, kTokens, 128, 128,
          kIndexWidth, 128),
      encode_tiled(
          &weight_a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          const_cast<void*>(params.hidden), 2,
          kHidden, kTokens, 64, 128, kHidden, 128),
      encode_tiled(
          &weight_b, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          const_cast<void*>(params.index_weight_projection), 2,
          kHidden, kIndexWeightWidth, 64, 32, kHidden, 128),
      encode_tiled(
          &weight_d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          params.index_weights, 2,
          kIndexWeightWidth, kTokens, 64, 128,
          kIndexWeightWidth, 128));
  if (descriptor_status != cudaSuccess) {
    return descriptor_status;
  }

#define FLASH_MLA_CSA_ROOT_KERNEL(left_sms, right_sms) \
  deep_gemm::root_fusion_kernel<                     \
      left_sms, right_sms, 4,                        \
      224, 128, 128, 128, 128, 128, 6,                \
      128, 128, 2, true, true,                        \
      3, false, false,                                \
      128, 128, 64, 128, 128, 128, 8,                \
      384, 128, 2, false, 128, false,                 \
      128, 64, 64, 128, 128, 128, 9,                 \
      128, 128, 2, false, 128, false>

  using RootKernel = decltype(&FLASH_MLA_CSA_ROOT_KERNEL(104, 44));
  RootKernel kernel = nullptr;
  if (grid_sms == 148) {
    switch (projection_sms) {
      case 96: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(96, 52); break;
      case 98: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(98, 50); break;
      case 100: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(100, 48); break;
      case 102: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(102, 46); break;
      case 104: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(104, 44); break;
      case 106: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(106, 42); break;
      case 108: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(108, 40); break;
      default: return cudaErrorInvalidValue;
    }
  } else if (grid_sms == 152) {
    switch (projection_sms) {
      case 104: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(104, 48); break;
      case 106: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(106, 46); break;
      case 108: kernel = &FLASH_MLA_CSA_ROOT_KERNEL(108, 44); break;
      default: return cudaErrorInvalidValue;
    }
  } else {
    return cudaErrorInvalidValue;
  }
#undef FLASH_MLA_CSA_ROOT_KERNEL

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
  config.gridDim = dim3(grid_sms, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSharedBytes;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;

  status = cudaLaunchKernelEx(
      &config,
      kernel,
      projection_a,
      projection_b,
      projection_scale_a,
      projection_scale_b,
      projection_d,
      static_cast<const __nv_bfloat16*>(params.hidden),
      static_cast<uint8_t*>(params.hidden_fp8),
      params.hidden_scale,
      params.hidden_scale_stride_k,
      static_cast<const __nv_bfloat16*>(params.q_norm_weight),
      static_cast<const __nv_bfloat16*>(params.kv_norm_weight),
      static_cast<const __nv_bfloat16*>(params.qkv),
      static_cast<__nv_bfloat16*>(params.q_norm),
      static_cast<__nv_bfloat16*>(params.kv_norm),
      static_cast<uint8_t*>(params.q_norm_fp8),
      params.q_norm_scale,
      params.q_norm_scale_stride_k,
      params.projection_barrier,
      4u,
      index_a,
      index_b,
      index_d,
      weight_a,
      weight_b,
      weight_d,
      params.index_barrier,
      params.row_ready,
      params.num_state_slots,
      params.c4_workspace,
      4u,
      params.index_kv_score,
      params.state_cache,
      params.state_stride0,
      params.state_stride1,
      params.token_to_request,
      params.positions,
      params.state_slot_mapping,
      params.state_block_table,
      params.state_block_table_stride,
      params.rms_weight,
      params.cos_sin_cache,
      params.cos_sin_stride,
      params.kv_cache,
      params.kv_slot_mapping,
      params.kv_cache_stride0,
      params.ape);
  return status;
}
