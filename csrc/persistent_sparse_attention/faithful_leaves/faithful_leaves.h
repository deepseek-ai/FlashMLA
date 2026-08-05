// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

extern "C" {

cudaError_t launch_index_query_quant(const int64_t*, const void*, const float*, uint8_t*, uint8_t*, uint32_t, uint32_t, cudaStream_t);
cudaError_t launch_index_weight_scale(const void*, float*, uint32_t, float, float, uint32_t, cudaStream_t);
cudaError_t launch_index_cache_pack(const uint8_t*, uint8_t*, int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, cudaStream_t);
cudaError_t launch_kv_cache_gather(void*, const uint8_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, int32_t, int32_t, int32_t, uint32_t, cudaStream_t);

}  // extern "C"
