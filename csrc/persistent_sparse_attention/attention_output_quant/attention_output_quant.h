// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

cudaError_t launch_attention_output_quant(
    const void* o, const int64_t* positions, const float* cos_sin_cache,
    void* fp8_output, int32_t* scale_output, int32_t num_tokens,
    int64_t o_stride_token, int64_t o_stride_head, int64_t o_stride_dim,
    int64_t position_stride, int64_t cache_stride_position,
    int64_t cache_stride_dim, int64_t fp8_stride_token,
    int64_t fp8_stride_group, int64_t fp8_stride_dim,
    int64_t scale_stride_token, int64_t scale_stride_group,
    int64_t scale_stride_head, int32_t num_sms, cudaStream_t stream);
