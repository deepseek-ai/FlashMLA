#pragma once

#include <torch/extension.h>
#include "cutlass/numeric_types.h"
#include "helper.h"

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <>
struct cutlass_dtype<__nv_fp8_e4m3> {
  using type = cutlass::float_e4m3_t;
};

template <>
struct cutlass_dtype<__nv_fp8_e5m2> {
  using type = cutlass::float_e5m2_t;
};

template <typename T>
using cutlass_dtype_t = typename cutlass_dtype<T>::type;

template<typename T>
struct DeviceAllocation {
  T* ptr_ = nullptr;
  size_t offset_ = 0;
  size_t size_ = 0;
  torch::Tensor tensor;

  DeviceAllocation(DeviceAllocation const&) = delete;
  DeviceAllocation& operator=(DeviceAllocation const&) = delete;

  DeviceAllocation() = default;
  DeviceAllocation(size_t size) { reset(size); }
  ~DeviceAllocation() {}

  void reset(size_t size, size_t offset=0) {
    size_t num_element = sizeof(T) * (size + offset);
    auto options = torch::TensorOptions()
          .dtype(torch::kInt8)
          .device(torch::kCUDA);

    tensor = torch::empty(num_element, options);
    ptr_ = tensor.data_ptr();
    size_ = size;
    offset_ = offset;
  }

  T* get() {
    return ptr_ + offset_;
  }

  const T* get() const {
    return ptr_ + offset_;
  }

  size_t size() const { return size_; }

  size_t get_storage_size() const { return (size_ + offset_) * sizeof(T); }

  void copy_from_host(const T* ptr, size_t sz) {
    auto ret = cudaMemcpy(ptr_, ptr, sz * sizeof(T), cudaMemcpyDefault);
    assert(ret == cudaSuccess);
  }

  void copy_from_device(const T* ptr, size_t sz) {
    auto ret = cudaMemcpy(ptr_, ptr, sz * sizeof(T), cudaMemcpyDefault);
    assert(ret == cudaSuccess);
  }
};