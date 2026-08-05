#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace flash_mla_csa {

inline cudaStream_t current_cuda_stream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

inline void check_last_cuda_error() {
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace flash_mla_csa
