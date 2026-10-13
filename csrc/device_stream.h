#pragma once

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>
#endif

#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND
#include <acl/acl.h>

extern "C" aclrtStream flash_mla_get_current_npu_stream();
#endif

// For CUDA GPU, return the current `cudaStream_t`; for Ascend NPU, return the
// current `aclrtStream`.
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
inline cudaStream_t get_current_stream() {
    void *stream = nullptr;
    // -1 preserves getCurrentCUDAStream()'s current-device behavior.
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(-1, &stream));
    return static_cast<cudaStream_t>(stream);
}
#endif

#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND
inline aclrtStream get_current_stream() {
    return flash_mla_get_current_npu_stream();
}
#endif
