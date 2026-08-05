// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
void launch_combine_indices(const int32_t* topk, int64_t topk_stride,
    int32_t* combined, int64_t combined_stride, int32_t* lengths,
    int64_t lengths_stride, int32_t compressed_rows, int32_t num_tokens,
    int32_t num_sms, cudaStream_t stream);
