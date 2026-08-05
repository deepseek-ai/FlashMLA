// SPDX-License-Identifier: Apache-2.0
// Derived from accepted CSA K6; vendored to avoid source-tree dependencies.

#include "index_cache_gather.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kCacheBlockSize = 64;
constexpr uint32_t kValueBytes = 64;
constexpr uint32_t kScaleBytes = 4;
constexpr uint32_t kThreads = 1024;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kRowsPerWarp = 6;
constexpr uint32_t kLanesPerRow = 5;
constexpr uint32_t kRowsPerCta = kWarps * kRowsPerWarp;

struct KernelParams {
    const uint8_t* kv_cache;
    uint8_t* dst_k;
    uint8_t* dst_scale;
    const int32_t* block_table;
    const int32_t* cu_seq_lens;
    int64_t cache_block_stride;
    int64_t dst_k_row_stride;
    int64_t dst_scale_row_stride;
    int64_t block_table_row_stride;
    uint32_t batch_size;
    uint32_t output_capacity;
    uint32_t num_cache_blocks;
    uint32_t block_table_width;
};

__device__ __forceinline__ int find_batch(
        const int32_t* cu_seq_lens,
        const uint32_t batch_size,
        const uint32_t token_idx) {
    uint32_t low = 0;
    uint32_t high = batch_size;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2;
        if (static_cast<uint32_t>(cu_seq_lens[middle + 1]) <= token_idx) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low >= batch_size) {
        return -1;
    }
    const int32_t start = cu_seq_lens[low];
    const int32_t end = cu_seq_lens[low + 1];
    if (start < 0 || end < start ||
        token_idx < static_cast<uint32_t>(start) ||
        token_idx >= static_cast<uint32_t>(end)) {
        return -1;
    }
    return static_cast<int>(low);
}

template <bool kVectorized>
__global__ __launch_bounds__(kThreads, 1)
void index_cache_gather_kernel(KernelParams params) {
    extern __shared__ volatile uint8_t residency_guard[];
    if (threadIdx.x == 0) {
        residency_guard[0] = 0;
    }

    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = threadIdx.x / 32;
    if (lane >= kRowsPerWarp * kLanesPerRow) {
        return;
    }
    const uint32_t row_in_warp = lane / kLanesPerRow;
    const uint32_t lane_in_row = lane % kLanesPerRow;
    const uint64_t row_stride =
        static_cast<uint64_t>(gridDim.x) * kRowsPerCta;
    for (uint64_t token_idx =
             static_cast<uint64_t>(blockIdx.x) * kRowsPerCta +
                 warp * kRowsPerWarp + row_in_warp;
         token_idx < params.output_capacity;
         token_idx += row_stride) {
        const int batch = find_batch(
            params.cu_seq_lens,
            params.batch_size,
            static_cast<uint32_t>(token_idx));
        if (batch < 0) {
            continue;
        }
        const uint32_t sequence_start =
            static_cast<uint32_t>(params.cu_seq_lens[batch]);
        const uint32_t sequence_token =
            static_cast<uint32_t>(token_idx) - sequence_start;
        const uint32_t logical_block =
            sequence_token / kCacheBlockSize;
        if (logical_block >= params.block_table_width) {
            continue;
        }
        const int32_t physical_block =
            params.block_table[
                static_cast<int64_t>(batch) *
                    params.block_table_row_stride +
                logical_block];
        if (physical_block < 0 ||
            static_cast<uint32_t>(physical_block) >=
                params.num_cache_blocks) {
            continue;
        }
        const uint32_t token_in_block =
            sequence_token % kCacheBlockSize;
        const uint8_t* page =
            params.kv_cache +
            static_cast<int64_t>(physical_block) *
                params.cache_block_stride;
        const uint8_t* source_value =
            page + static_cast<int64_t>(token_in_block) * kValueBytes;
        uint8_t* destination_value =
            params.dst_k +
            static_cast<int64_t>(token_idx) *
                params.dst_k_row_stride;
        const uint8_t* source_scale =
            page +
            static_cast<int64_t>(kCacheBlockSize) * kValueBytes +
            static_cast<int64_t>(token_in_block) * kScaleBytes;
        uint8_t* destination_scale =
            params.dst_scale +
            static_cast<int64_t>(token_idx) *
                params.dst_scale_row_stride;

        if constexpr (kVectorized) {
            if (lane_in_row < kValueBytes / sizeof(uint4)) {
                reinterpret_cast<uint4*>(destination_value)[lane_in_row] =
                    reinterpret_cast<const uint4*>(source_value)[lane_in_row];
            }
            if (lane_in_row == kValueBytes / sizeof(uint4)) {
                *reinterpret_cast<uint32_t*>(destination_scale) =
                    *reinterpret_cast<const uint32_t*>(source_scale);
            }
        } else {
            for (uint32_t byte = lane_in_row;
                 byte < kValueBytes;
                 byte += kLanesPerRow) {
                destination_value[byte] = source_value[byte];
            }
            if (lane_in_row < kScaleBytes) {
                destination_scale[lane_in_row] =
                    source_scale[lane_in_row];
            }
        }
    }
}

cudaError_t configure_kernel(
        const int device,
        int* dynamic_smem_bytes,
        int* active_blocks_per_sm) {
    int occupancy = 0;
    cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        index_cache_gather_kernel<true>,
        kThreads,
        0);
    if (error != cudaSuccess) {
        return error;
    }
    if (occupancy == 1) {
        *dynamic_smem_bytes = 0;
        *active_blocks_per_sm = occupancy;
        return cudaSuccess;
    }

    int max_smem_per_sm = 0;
    int max_smem_per_block = 0;
    error = cudaDeviceGetAttribute(
        &max_smem_per_sm,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor,
        device);
    if (error != cudaSuccess) {
        return error;
    }
    error = cudaDeviceGetAttribute(
        &max_smem_per_block,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device);
    if (error != cudaSuccess) {
        return error;
    }
    const int residency_smem = max_smem_per_sm / 2 + 256;
    if (residency_smem > max_smem_per_block) {
        return cudaErrorInvalidConfiguration;
    }
    error = cudaFuncSetAttribute(
        index_cache_gather_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        residency_smem);
    if (error != cudaSuccess) {
        return error;
    }
    error = cudaFuncSetAttribute(
        index_cache_gather_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        residency_smem);
    if (error != cudaSuccess) {
        return error;
    }
    occupancy = 0;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        index_cache_gather_kernel<true>,
        kThreads,
        residency_smem);
    if (error != cudaSuccess) {
        return error;
    }
    if (occupancy != 1) {
        return cudaErrorInvalidConfiguration;
    }
    *dynamic_smem_bytes = residency_smem;
    *active_blocks_per_sm = occupancy;
    return cudaSuccess;
}

}  // namespace

cudaError_t launch_index_cache_gather(
    const uint8_t* kv_cache,
    uint8_t* dst_k,
    uint8_t* dst_scale,
    const int32_t* block_table,
    const int32_t* cu_seq_lens,
    int64_t cache_block_stride,
    int64_t dst_k_row_stride,
    int64_t dst_scale_row_stride,
    int64_t block_table_row_stride,
    uint32_t batch_size,
    uint32_t output_capacity,
    uint32_t num_cache_blocks,
    uint32_t block_table_width,
    uint32_t num_sms,
    bool vectorized,
    cudaStream_t stream) {
    if (output_capacity == 0 || batch_size == 0) {
        return cudaSuccess;
    }
    int device = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        return error;
    }
    int dynamic_smem_bytes = 0;
    int active_blocks_per_sm = 0;
    error = configure_kernel(
        device,
        &dynamic_smem_bytes,
        &active_blocks_per_sm);
    if (error != cudaSuccess || active_blocks_per_sm != 1) {
        return error == cudaSuccess ? cudaErrorInvalidConfiguration : error;
    }

    const KernelParams params = {
        kv_cache,
        dst_k,
        dst_scale,
        block_table,
        cu_seq_lens,
        cache_block_stride,
        dst_k_row_stride,
        dst_scale_row_stride,
        block_table_row_stride,
        batch_size,
        output_capacity,
        num_cache_blocks,
        block_table_width,
    };
    if (vectorized) {
        index_cache_gather_kernel<true>
            <<<num_sms, kThreads, dynamic_smem_bytes, stream>>>(params);
    } else {
        index_cache_gather_kernel<false>
            <<<num_sms, kThreads, dynamic_smem_bytes, stream>>>(params);
    }
    return cudaGetLastError();
}

cudaError_t query_index_cache_gather_residency(
    int device,
    int* active_blocks_per_sm,
    int* threads_per_cta,
    int* dynamic_smem_bytes) {
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) {
        return error;
    }
    error = configure_kernel(
        device,
        dynamic_smem_bytes,
        active_blocks_per_sm);
    if (error != cudaSuccess) {
        return error;
    }
    *threads_per_cta = static_cast<int>(kThreads);
    return cudaSuccess;
}
