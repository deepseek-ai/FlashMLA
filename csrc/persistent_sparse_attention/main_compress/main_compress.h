#pragma once

#include <cuda_runtime.h>
#include <cstdint>

cudaError_t launch_main_compress(
    uint32_t, const float*, uint32_t, uint32_t, uint32_t, const int32_t*,
    const int64_t*, const int64_t*, const int32_t*, uint32_t, const void*,
    float, const float*, uint32_t, uint8_t*, const int64_t*, uint32_t,
    uint32_t, bool, cudaStream_t);
