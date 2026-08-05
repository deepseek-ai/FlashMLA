// SPDX-License-Identifier: MIT
//
// Warp-specialized persistent gather for the CSA Indexer K cache.
//
// Physical source layout, per page:
//   [64 rows * 64 value bytes][64 rows * 4 scale bytes][unused padding]
//
// WG0 is the dispatch warpgroup.  Warp 0 publishes double-buffered batches of
// four page descriptors while the remaining WG0 warps stay out of the data
// path.  WG1 through WG7 are complete consumer warpgroups.  Their 28 warps
// copy four pages concurrently, which supplies enough memory-level parallelism
// even for the diagnostic 40-SM launch.  Every CTA walks page batches with a
// grid-stride loop, so the launch grid is exactly the SM budget assigned by the
// caller (40, 108, 132, or 148), independent of row count.

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kNumRowsPerPage = 64;
constexpr uint32_t kNumValueBytesPerRow = 64;
constexpr uint32_t kNumScaleBytesPerRow = 4;
constexpr uint32_t kNumValueBytesPerPage =
    kNumRowsPerPage * kNumValueBytesPerRow;
constexpr uint32_t kNumProducerWarps = 4;
constexpr uint32_t kNumConsumerWarps = 28;
constexpr uint32_t kNumThreads =
    (kNumProducerWarps + kNumConsumerWarps) * 32;
constexpr uint32_t kNumPagesPerBatch = 4;
constexpr uint32_t kNumDescriptorStages = 2;

struct PageBatch {
    uint32_t page_indices[kNumPagesPerBatch];
    uint32_t active_count;
};

struct KernelParams {
    const uint8_t* src;
    uint8_t* values_out;
    int32_t* scales_out;
    int64_t src_page_stride;
    int64_t src_byte_stride;
    int64_t values_row_stride;
    int64_t values_byte_stride;
    int64_t scales_row_stride;
    uint32_t first_page_idx;
    uint32_t last_page_idx;
    uint32_t src_row_offset;
    uint32_t src_row_end;
    uint32_t dst_row_offset;
};

__device__ __forceinline__ void publish_page_batch(
        PageBatch* batches,
        const KernelParams& params,
        const uint32_t batch_idx,
        const uint32_t descriptor_stage,
        const bool is_dispatch_lane) {
    if (not is_dispatch_lane) {
        return;
    }

    uint32_t active_count = 0;
#pragma unroll
    for (uint32_t page_slot = 0;
         page_slot < kNumPagesPerBatch;
         ++page_slot) {
        const uint64_t grid_iteration =
            static_cast<uint64_t>(batch_idx) * kNumPagesPerBatch +
            page_slot;
        const uint64_t page_idx =
            static_cast<uint64_t>(params.first_page_idx) +
            blockIdx.x + grid_iteration * gridDim.x;
        batches[descriptor_stage].page_indices[page_slot] =
            static_cast<uint32_t>(page_idx);
        active_count += page_idx <= params.last_page_idx;
    }
    // Publish the count last.  The CTA barrier makes the entire descriptor
    // stage visible to consumers.
    batches[descriptor_stage].active_count = active_count;
}

template <bool kVectorized>
__global__ __launch_bounds__(kNumThreads, 1)
void index_cache_pack_persistent_kernel(KernelParams params) {
    __shared__ PageBatch batches[kNumDescriptorStages];

    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t lane_idx = threadIdx.x % 32;
    const bool is_dispatch_lane = warp_idx == 0 and lane_idx == 0;
    const bool is_consumer_warp =
        warp_idx >= kNumProducerWarps and
        warp_idx < kNumProducerWarps + kNumConsumerWarps;
    const uint32_t consumer_warp_idx = warp_idx - kNumProducerWarps;

    publish_page_batch(batches, params, 0, 0, is_dispatch_lane);
    __syncthreads();

    for (uint32_t batch_idx = 0;; ++batch_idx) {
        const uint32_t descriptor_stage =
            batch_idx & (kNumDescriptorStages - 1);
        const uint32_t active_count =
            batches[descriptor_stage].active_count;

        if (active_count == 0) {
            break;
        }

        // The producer prepares the next descriptor stage while all consumer
        // warpgroups copy the current one.
        publish_page_batch(
            batches,
            params,
            batch_idx + 1,
            descriptor_stage ^ 1U,
            is_dispatch_lane);

        if (is_consumer_warp) {
            // Three page slots use eight warps each.  Four warps cover the
            // fourth slot in two row groups.  This keeps all 28 consumer
            // warps active while preserving a complete-warp copy pattern.
            const bool is_compact_page_slot =
                consumer_warp_idx < 24;
            const uint32_t page_slot = is_compact_page_slot
                ? consumer_warp_idx / 8
                : 3;
            const uint32_t warp_in_page = is_compact_page_slot
                ? consumer_warp_idx % 8
                : consumer_warp_idx - 24;
            const uint32_t row_group_count =
                is_compact_page_slot ? 1 : 2;

            if (page_slot < active_count) {
                const uint32_t page_idx =
                    batches[descriptor_stage].page_indices[page_slot];

#pragma unroll
                for (uint32_t row_group_idx = 0;
                     row_group_idx < 2;
                     ++row_group_idx) {
                    if (row_group_idx >= row_group_count) {
                        break;
                    }
                    const uint32_t row_group =
                        warp_in_page + row_group_idx * 4;
                    const uint32_t row_base = row_group * 8;
                    const uint32_t row_in_group = lane_idx / 4;
                    const uint32_t vector_idx = lane_idx % 4;
                    const uint32_t row_in_page =
                        row_base + row_in_group;
                    const uint32_t src_row_idx =
                        page_idx * kNumRowsPerPage + row_in_page;
                    const uint32_t dst_row_idx =
                        params.dst_row_offset +
                        src_row_idx - params.src_row_offset;
                    const int64_t src_page_offset =
                        static_cast<int64_t>(page_idx) *
                        params.src_page_stride;
                    const int64_t src_value_offset =
                        static_cast<int64_t>(
                            row_in_page * kNumValueBytesPerRow +
                            vector_idx * 16) *
                            params.src_byte_stride;
                    if constexpr (kVectorized) {
                        const auto src_vector =
                            reinterpret_cast<const uint4*>(
                                params.src + src_page_offset +
                                src_value_offset);
                        const auto dst_vector =
                            reinterpret_cast<uint4*>(
                                params.values_out +
                                static_cast<int64_t>(dst_row_idx) *
                                    params.values_row_stride +
                                vector_idx * 16);
                        *dst_vector = *src_vector;

                        // Each of lanes 0 and 1 moves four adjacent raw
                        // int32 scale payloads as one 16-byte transaction.
                        if (lane_idx < 2) {
                            const uint32_t scale_row =
                                row_base + lane_idx * 4;
                            const auto src_scales =
                                reinterpret_cast<const uint4*>(
                                    params.src + src_page_offset +
                                    kNumValueBytesPerPage +
                                    scale_row * kNumScaleBytesPerRow);
                            const uint32_t dst_scale_row =
                                params.dst_row_offset +
                                page_idx * kNumRowsPerPage +
                                scale_row -
                                params.src_row_offset;
                            const auto dst_scales =
                                reinterpret_cast<uint4*>(
                                    params.scales_out + dst_scale_row);
                            *dst_scales = *src_scales;
                        }
                    } else {
                        if (src_row_idx >= params.src_row_offset and
                            src_row_idx < params.src_row_end) {
                            const int64_t dst_value_offset =
                                static_cast<int64_t>(dst_row_idx) *
                                    params.values_row_stride +
                                static_cast<int64_t>(vector_idx * 16) *
                                    params.values_byte_stride;
#pragma unroll
                            for (uint32_t byte_idx = 0;
                                 byte_idx < 16;
                                 ++byte_idx) {
                                params.values_out[
                                    dst_value_offset +
                                    static_cast<int64_t>(byte_idx) *
                                        params.values_byte_stride] =
                                    params.src[
                                        src_page_offset + src_value_offset +
                                        static_cast<int64_t>(byte_idx) *
                                            params.src_byte_stride];
                            }

                            // One lane per row reconstructs the raw four-byte
                            // scale payload for arbitrary byte strides.
                            if (vector_idx == 0) {
                                const int64_t src_scale_offset =
                                    src_page_offset +
                                    static_cast<int64_t>(
                                        kNumValueBytesPerPage +
                                        row_in_page *
                                            kNumScaleBytesPerRow) *
                                        params.src_byte_stride;
                                const uint32_t scale_bits =
                                    static_cast<uint32_t>(
                                        params.src[src_scale_offset]) |
                                    (static_cast<uint32_t>(params.src[
                                         src_scale_offset +
                                         params.src_byte_stride])
                                     << 8) |
                                    (static_cast<uint32_t>(params.src[
                                         src_scale_offset +
                                         2 * params.src_byte_stride])
                                     << 16) |
                                    (static_cast<uint32_t>(params.src[
                                         src_scale_offset +
                                         3 * params.src_byte_stride])
                                     << 24);
                                params.scales_out[
                                    static_cast<int64_t>(dst_row_idx) *
                                    params.scales_row_stride] =
                                    static_cast<int32_t>(scale_bits);
                            }
                        }
                    }
                }
            }
        }

        // Consumers release the current descriptor stage after copying.  The
        // next stage was prepared concurrently by the producer.
        __syncthreads();
    }
}

}  // namespace

extern "C" cudaError_t launch_index_cache_pack(
    const uint8_t* src,
    uint8_t* values_out,
    int32_t* scales_out,
    int64_t src_page_stride,
    int64_t src_byte_stride,
    int64_t values_row_stride,
    int64_t values_byte_stride,
    int64_t scales_row_stride,
    uint32_t src_row_offset,
    uint32_t num_rows,
    uint32_t dst_row_offset,
    uint32_t num_sms,
    bool vectorized,
    cudaStream_t stream) {
    if (num_rows == 0) {
        return cudaSuccess;
    }

    const uint32_t first_page_idx = src_row_offset / kNumRowsPerPage;
    const uint32_t src_row_end = src_row_offset + num_rows;
    const uint32_t last_page_idx =
        (src_row_end - 1) / kNumRowsPerPage;
    const KernelParams params = {
        src,
        values_out,
        scales_out,
        src_page_stride,
        src_byte_stride,
        values_row_stride,
        values_byte_stride,
        scales_row_stride,
        first_page_idx,
        last_page_idx,
        src_row_offset,
        src_row_end,
        dst_row_offset,
    };

    if (vectorized) {
        index_cache_pack_persistent_kernel<true>
            <<<num_sms, kNumThreads, 0, stream>>>(params);
    } else {
        index_cache_pack_persistent_kernel<false>
            <<<num_sms, kNumThreads, 0, stream>>>(params);
    }
    return cudaGetLastError();
}
