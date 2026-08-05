#pragma once

#include <cuda.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace flash_mla::csa::deepgemm_adapters::tma {

enum class Element {
  kInt32,
  kFloat32,
  kBFloat16,
  kFloat8E4M3,
  kPackedFloat4E2M1,
};

inline int element_bytes(Element element) {
  switch (element) {
    case Element::kInt32:
    case Element::kFloat32:
      return 4;
    case Element::kBFloat16:
      return 2;
    case Element::kFloat8E4M3:
    case Element::kPackedFloat4E2M1:
      return 1;
  }
  return 0;
}

inline CUtensorMapDataType driver_type(
    Element element,
    bool packed_fp4_unpacked_in_shared) {
  switch (element) {
    case Element::kInt32:
      return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case Element::kFloat32:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case Element::kBFloat16:
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case Element::kFloat8E4M3:
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case Element::kPackedFloat4E2M1:
      return packed_fp4_unpacked_in_shared
          ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
          : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
  }
  return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

inline CUtensorMapSwizzle swizzle(int mode) {
  switch (mode) {
    case 0:
    case 16:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      TORCH_CHECK(false, "unsupported TMA swizzle mode: ", mode);
  }
}

inline int aligned_size(int elements, int bytes_per_element) {
  constexpr int kTmaAlignmentBytes = 16;
  const int bytes = elements * bytes_per_element;
  return ((bytes + kTmaAlignmentBytes - 1) / kTmaAlignmentBytes) *
         kTmaAlignmentBytes / bytes_per_element;
}

inline CUtensorMap make_2d(
    const torch::Tensor& tensor,
    Element element,
    int global_inner,
    int global_outer,
    int shared_inner,
    int shared_outer,
    int global_outer_stride,
    int swizzle_mode,
    bool packed_fp4_unpacked_in_shared = true) {
  const int bytes = element_bytes(element);
  if (swizzle_mode != 0) {
    shared_inner = swizzle_mode / bytes;
  }
  if (element == Element::kPackedFloat4E2M1) {
    TORCH_CHECK(
        !packed_fp4_unpacked_in_shared || global_inner % 128 == 0,
        "unpacked MXFP4 TMA inner dimension must be divisible by 128");
    if (!packed_fp4_unpacked_in_shared && swizzle_mode != 0) {
      shared_inner = swizzle_mode * 2;
    }
  }

  CUtensorMap result{};
  const cuuint64_t global_dimensions[2] = {
      static_cast<cuuint64_t>(global_inner),
      static_cast<cuuint64_t>(global_outer),
  };
  const cuuint64_t global_strides[1] = {
      static_cast<cuuint64_t>(global_outer_stride * bytes),
  };
  const cuuint32_t box_dimensions[2] = {
      static_cast<cuuint32_t>(shared_inner),
      static_cast<cuuint32_t>(shared_outer),
  };
  constexpr cuuint32_t element_strides[2] = {1, 1};
  const CUresult status = cuTensorMapEncodeTiled(
      &result,
      driver_type(element, packed_fp4_unpacked_in_shared),
      2,
      tensor.data_ptr(),
      global_dimensions,
      global_strides,
      box_dimensions,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle(swizzle_mode),
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  const char* error_name = nullptr;
  if (status != CUDA_SUCCESS) {
    cuGetErrorName(status, &error_name);
  }
  TORCH_CHECK(
      status == CUDA_SUCCESS,
      "cuTensorMapEncodeTiled failed: ",
      error_name == nullptr ? "unknown CUDA driver error" : error_name,
      " element=", static_cast<int>(element),
      " global=[", global_inner, ",", global_outer,
      "] box=[", shared_inner, ",", shared_outer,
      "] outer_stride=", global_outer_stride,
      " swizzle=", swizzle_mode,
      " ptr=", tensor.data_ptr());
  return result;
}

inline CUtensorMap make_a(
    const torch::Tensor& tensor,
    Element element,
    int shape_m,
    int shape_k,
    int block_m,
    int block_k,
    int outer_stride,
    int swizzle_mode) {
  return make_2d(
      tensor, element, shape_k, shape_m, block_k, block_m, outer_stride,
      swizzle_mode);
}

inline CUtensorMap make_b(
    const torch::Tensor& tensor,
    Element element,
    int shape_n,
    int shape_k,
    int block_n,
    int block_k,
    int outer_stride,
    int swizzle_mode) {
  return make_2d(
      tensor, element, shape_k, shape_n, block_k, block_n, outer_stride,
      swizzle_mode);
}

inline CUtensorMap make_d(
    const torch::Tensor& tensor,
    Element element,
    int shape_m,
    int shape_n,
    int block_m,
    int block_n,
    int outer_stride,
    int swizzle_mode) {
  return make_2d(
      tensor, element, shape_n, shape_m, block_n, block_m, outer_stride,
      swizzle_mode);
}

inline CUtensorMap make_scale(
    const torch::Tensor& tensor,
    int shape_mn,
    int shape_k,
    int block_mn,
    int granularity_k) {
  shape_mn = aligned_size(shape_mn, 4);
  return make_2d(
      tensor,
      Element::kInt32,
      shape_mn,
      (shape_k + granularity_k * 4 - 1) / (granularity_k * 4),
      block_mn,
      1,
      shape_mn,
      0);
}

}  // namespace flash_mla::csa::deepgemm_adapters::tma
