// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
void launch_swa_cache_insert(const void* kv, int64_t kv_stride0,
    uint8_t* cache, int64_t cache_stride0, const int64_t* slot_mapping,
    int64_t slot_stride0, const int64_t* position_ids,
    int64_t position_stride0, const float* cos_sin_cache,
    int64_t cos_sin_stride0, int32_t num_tokens, int32_t cache_block_size,
    int32_t num_sms, cudaStream_t stream);
