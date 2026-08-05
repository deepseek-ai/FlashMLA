#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Fixed production specialization: head_size=state_width=1024,
// compress_ratio=4. Strides are measured in elements.
cudaError_t launch_state_save_fp32(
    const float*, uint64_t, const float*, uint64_t, const float*, uint64_t,
    const int64_t*, float*, uint64_t, uint64_t, const int64_t*, uint32_t,
    uint32_t, uint32_t, cudaStream_t);

cudaError_t launch_state_save_bf16(
    const void*, uint64_t, const void*, uint64_t, const void*, uint64_t,
    const int64_t*, void*, uint64_t, uint64_t, const int64_t*, uint32_t,
    uint32_t, uint32_t, cudaStream_t);
