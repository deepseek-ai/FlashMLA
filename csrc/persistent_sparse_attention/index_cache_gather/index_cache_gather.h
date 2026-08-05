// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
cudaError_t launch_index_cache_gather(const uint8_t* kv_cache,
    uint8_t* dst_k, uint8_t* dst_scale, const int32_t* block_table,
    const int32_t* cu_seq_lens, int64_t cache_block_stride,
    int64_t dst_k_row_stride, int64_t dst_scale_row_stride,
    int64_t block_table_row_stride, uint32_t batch_size,
    uint32_t output_capacity, uint32_t num_cache_blocks,
    uint32_t block_table_width, uint32_t num_sms, bool vectorized,
    cudaStream_t stream);
cudaError_t query_index_cache_gather_residency(int device,
    int* active_blocks_per_sm, int* threads_per_cta,
    int* dynamic_smem_bytes);
