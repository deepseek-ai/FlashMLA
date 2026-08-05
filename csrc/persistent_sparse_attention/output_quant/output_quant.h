// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// Fixed production shape: contiguous BF16 [8192, 16384] to contiguous E4M3
// bytes plus physical int32 scale backing [32, 8192].
cudaError_t launch_output_quant(
    const void* input, uint8_t* output, int32_t* scales,
    uint32_t num_sms, cudaStream_t stream);
