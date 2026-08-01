#pragma once

#include <cuda_runtime.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>

namespace kerutils {

// Helper to return the current CUDA stream for the device that `t` lives on, as
// a raw cudaStream_t. Semantically equivalent to at::cuda::getCurrentCUDAStream().
inline cudaStream_t get_current_cuda_stream(const torch::stable::Tensor &t) {
    void *stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream_ptr));
    return static_cast<cudaStream_t>(stream_ptr);
}

}
