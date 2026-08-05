// SPDX-License-Identifier: MIT
// Derived from accepted CSA K9; vendored to avoid source-tree dependencies.

#include "combine_indices.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kTopK = 1024;
constexpr int kSwa = 128;
constexpr int kCombined = kTopK + kSwa;
constexpr int kProductionTokens = 8192;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerWarpgroup = 4;
constexpr int kThreadsPerWarpgroup = kWarpSize * kWarpsPerWarpgroup;
constexpr int kNumConsumerWarps = 28;
constexpr int kNumWarpgroups = 8;
constexpr int kNumThreads = kNumWarpgroups * kThreadsPerWarpgroup;
constexpr int kNumPipelineThreads =
    kWarpSize + kNumConsumerWarps * kWarpSize;
constexpr int kNumPipelineSlots = 2;
constexpr int kMaxTasksPerConsumerWarp = 8;

static_assert(kNumThreads == 1024);
static_assert(kNumPipelineThreads == 928);
static_assert(kTopK % 4 == 0);

__device__ __forceinline__ void producer_arrive_ready(int32_t slot) {
    if (slot == 0) {
        asm volatile("bar.arrive 1, 928;" ::: "memory");
    } else {
        asm volatile("bar.arrive 2, 928;" ::: "memory");
    }
}

__device__ __forceinline__ void consumer_wait_ready(int32_t slot) {
    if (slot == 0) {
        asm volatile("bar.sync 1, 928;" ::: "memory");
    } else {
        asm volatile("bar.sync 2, 928;" ::: "memory");
    }
}

__device__ __forceinline__ void consumer_arrive_consumed(int32_t slot) {
    if (slot == 0) {
        asm volatile("bar.arrive 3, 928;" ::: "memory");
    } else {
        asm volatile("bar.arrive 4, 928;" ::: "memory");
    }
}

__device__ __forceinline__ void producer_wait_consumed(int32_t slot) {
    if (slot == 0) {
        asm volatile("bar.sync 3, 928;" ::: "memory");
    } else {
        asm volatile("bar.sync 4, 928;" ::: "memory");
    }
}

__device__ __forceinline__ bool vector_aligned(
    const int32_t* source,
    const int32_t* destination) {
    const auto source_address = reinterpret_cast<std::uintptr_t>(source);
    const auto destination_address =
        reinterpret_cast<std::uintptr_t>(destination);
    return ((source_address | destination_address) &
            (alignof(int4) - 1)) == 0;
}

__device__ __forceinline__ void copy_topk_one(
    const int32_t* source,
    int32_t* destination,
    int32_t lane,
    bool is_vector_aligned) {
    if (is_vector_aligned) {
        const auto source_vector = reinterpret_cast<const int4*>(source);
        const auto destination_vector = reinterpret_cast<int4*>(destination);
#pragma unroll
        for (int32_t vector_idx = lane;
             vector_idx < kTopK / 4;
             vector_idx += kWarpSize) {
            destination_vector[vector_idx] = source_vector[vector_idx];
        }
    } else {
#pragma unroll
        for (int32_t element_idx = lane;
             element_idx < kTopK;
             element_idx += kWarpSize) {
            destination[element_idx] = source[element_idx];
        }
    }
}

__device__ __forceinline__ void copy_topk_pair_aligned(
    const int32_t* source0,
    int32_t* destination0,
    const int32_t* source1,
    int32_t* destination1,
    int32_t lane) {
    const auto source_vector0 = reinterpret_cast<const int4*>(source0);
    const auto destination_vector0 = reinterpret_cast<int4*>(destination0);
    const auto source_vector1 = reinterpret_cast<const int4*>(source1);
    const auto destination_vector1 = reinterpret_cast<int4*>(destination1);
#pragma unroll
    for (int32_t vector_idx = lane;
         vector_idx < kTopK / 4;
         vector_idx += kWarpSize) {
        const int4 values0 = source_vector0[vector_idx];
        const int4 values1 = source_vector1[vector_idx];
        destination_vector0[vector_idx] = values0;
        destination_vector1[vector_idx] = values1;
    }
}

__device__ __forceinline__ void write_swa(
    int32_t* destination,
    int32_t swa_base,
    int32_t lane,
    bool is_vector_aligned) {
    const int32_t offset = lane * 4;
    if (is_vector_aligned) {
        auto destination_vectors =
            reinterpret_cast<int4*>(destination + kTopK);
        destination_vectors[lane] = make_int4(
            swa_base + offset,
            swa_base + offset + 1,
            swa_base + offset + 2,
            swa_base + offset + 3);
    } else {
        destination[kTopK + offset] = swa_base + offset;
        destination[kTopK + offset + 1] = swa_base + offset + 1;
        destination[kTopK + offset + 2] = swa_base + offset + 2;
        destination[kTopK + offset + 3] = swa_base + offset + 3;
    }
}

__device__ __forceinline__ void write_swa_aligned(
    int32_t* destination,
    int32_t swa_base,
    int32_t lane) {
    const int32_t offset = lane * 4;
    auto destination_vectors =
        reinterpret_cast<int4*>(destination + kTopK);
    destination_vectors[lane] = make_int4(
        swa_base + offset,
        swa_base + offset + 1,
        swa_base + offset + 2,
        swa_base + offset + 3);
}

__device__ __forceinline__ void process_one_token(
    const int32_t* topk,
    int64_t topk_stride,
    int32_t* combined,
    int64_t combined_stride,
    int32_t* lens,
    int64_t lens_stride,
    int32_t compressed_rows,
    int32_t token,
    int32_t lane) {
    const int32_t* source =
        topk + static_cast<int64_t>(token) * topk_stride;
    int32_t* destination =
        combined + static_cast<int64_t>(token) * combined_stride;
    const bool is_aligned = vector_aligned(source, destination);
    copy_topk_one(source, destination, lane, is_aligned);
    write_swa(destination, compressed_rows + token, lane, is_aligned);
    if (lane == 0) {
        lens[static_cast<int64_t>(token) * lens_stride] = kCombined;
    }
}

__device__ __forceinline__ void process_two_tokens(
    const int32_t* topk,
    int64_t topk_stride,
    int32_t* combined,
    int64_t combined_stride,
    int32_t* lens,
    int64_t lens_stride,
    int32_t compressed_rows,
    int32_t token0,
    int32_t token1,
    int32_t lane) {
    const int32_t* source0 =
        topk + static_cast<int64_t>(token0) * topk_stride;
    int32_t* destination0 =
        combined + static_cast<int64_t>(token0) * combined_stride;
    const int32_t* source1 =
        topk + static_cast<int64_t>(token1) * topk_stride;
    int32_t* destination1 =
        combined + static_cast<int64_t>(token1) * combined_stride;
    const bool aligned0 = vector_aligned(source0, destination0);
    const bool aligned1 = vector_aligned(source1, destination1);
    if (aligned0 and aligned1) {
        copy_topk_pair_aligned(
            source0, destination0, source1, destination1, lane);
    } else {
        copy_topk_one(source0, destination0, lane, aligned0);
        copy_topk_one(source1, destination1, lane, aligned1);
    }
    write_swa(destination0, compressed_rows + token0, lane, aligned0);
    write_swa(destination1, compressed_rows + token1, lane, aligned1);
    if (lane == 0) {
        lens[static_cast<int64_t>(token0) * lens_stride] = kCombined;
        lens[static_cast<int64_t>(token1) * lens_stride] = kCombined;
    }
}

__device__ __forceinline__ void process_one_token_contiguous(
    const int32_t* topk,
    int32_t* combined,
    int32_t compressed_rows,
    int32_t token,
    int32_t lane) {
    const int32_t* source =
        topk + static_cast<int64_t>(token) * kTopK;
    int32_t* destination =
        combined + static_cast<int64_t>(token) * kCombined;
    copy_topk_one(source, destination, lane, true);
    write_swa_aligned(destination, compressed_rows + token, lane);
}

__device__ __forceinline__ void process_two_tokens_contiguous(
    const int32_t* topk,
    int32_t* combined,
    int32_t compressed_rows,
    int32_t token0,
    int32_t token1,
    int32_t lane) {
    const int32_t* source0 =
        topk + static_cast<int64_t>(token0) * kTopK;
    int32_t* destination0 =
        combined + static_cast<int64_t>(token0) * kCombined;
    const int32_t* source1 =
        topk + static_cast<int64_t>(token1) * kTopK;
    int32_t* destination1 =
        combined + static_cast<int64_t>(token1) * kCombined;
    copy_topk_pair_aligned(
        source0, destination0, source1, destination1, lane);
    write_swa_aligned(destination0, compressed_rows + token0, lane);
    write_swa_aligned(destination1, compressed_rows + token1, lane);
}

template <int32_t kGridSize, int32_t kTasksPerConsumerWarp>
__global__ __launch_bounds__(kNumThreads, 1)
void combine_indices_fast_kernel(
    const int32_t* __restrict__ topk,
    int32_t* __restrict__ combined,
    int32_t* __restrict__ lens,
    int32_t compressed_rows) {
    static_assert(
        kGridSize * kNumConsumerWarps * kTasksPerConsumerWarp >=
        kProductionTokens);

    const int32_t thread_idx = static_cast<int32_t>(threadIdx.x);
    const int32_t warp_idx = thread_idx / kWarpSize;
    const int32_t lane = thread_idx % kWarpSize;
    if (warp_idx == 0) {
        // The dedicated producer warp publishes the independent lens
        // metadata. Consumers use the compile-time task schedule directly,
        // so no producer/consumer hand-off is required for this layout.
#pragma unroll
        for (int32_t token =
                 static_cast<int32_t>(blockIdx.x) * kWarpSize + lane;
             token < kProductionTokens;
             token += kWarpSize * kGridSize) {
            lens[token] = kCombined;
        }
        return;
    }
    if (warp_idx < kWarpsPerWarpgroup) {
        return;
    }

    const int32_t consumer_warp = warp_idx - kWarpsPerWarpgroup;
    if constexpr (kGridSize == 148 and kTasksPerConsumerWarp == 2) {
        const int32_t worker =
            static_cast<int32_t>(blockIdx.x) * kNumConsumerWarps +
            consumer_warp;
        const int32_t token0 = worker * 2;
        if (token0 < kProductionTokens) {
            process_two_tokens_contiguous(
                topk,
                combined,
                compressed_rows,
                token0,
                token0 + 1,
                lane);
        }
    } else {
#pragma unroll
        for (int32_t task_round = 0;
             task_round < kTasksPerConsumerWarp;
             task_round += 2) {
            const int32_t local_task0 =
                consumer_warp + task_round * kNumConsumerWarps;
            const int32_t local_task1 =
                local_task0 + kNumConsumerWarps;
            const int32_t token0 =
                static_cast<int32_t>(blockIdx.x) +
                local_task0 * kGridSize;
            const int32_t token1 =
                static_cast<int32_t>(blockIdx.x) +
                local_task1 * kGridSize;
            const bool valid0 = token0 < kProductionTokens;
            const bool valid1 =
                task_round + 1 < kTasksPerConsumerWarp and
                token1 < kProductionTokens;
            if (valid0 and valid1) {
                process_two_tokens_contiguous(
                    topk,
                    combined,
                    compressed_rows,
                    token0,
                    token1,
                    lane);
            } else if (valid0) {
                process_one_token_contiguous(
                    topk,
                    combined,
                    compressed_rows,
                    token0,
                    lane);
            }
        }
    }
}

__global__ __launch_bounds__(kNumThreads, 1)
void combine_indices_kernel(
    const int32_t* __restrict__ topk,
    int64_t topk_stride,
    int32_t* __restrict__ combined,
    int64_t combined_stride,
    int32_t* __restrict__ lens,
    int64_t lens_stride,
    int32_t compressed_rows,
    int32_t num_tokens) {
    __shared__ alignas(16) int64_t round_starts[kNumPipelineSlots];

    const int32_t thread_idx = static_cast<int32_t>(threadIdx.x);
    const int32_t warp_idx = thread_idx / kWarpSize;
    const int32_t lane = thread_idx % kWarpSize;
    const bool is_producer_warp = warp_idx == 0;
    const bool is_consumer_warp = warp_idx >= kWarpsPerWarpgroup;
    if (not is_producer_warp and not is_consumer_warp) {
        return;
    }
    const int32_t consumer_warp = warp_idx - kWarpsPerWarpgroup;
    const int32_t tasks_per_consumer_warp =
        gridDim.x == 40 ? 8 : (gridDim.x == 108 ? 3 : 2);
    const int64_t tasks_per_cta =
        static_cast<int64_t>(kNumConsumerWarps) *
        tasks_per_consumer_warp;
    const int64_t round_stride =
        tasks_per_cta * static_cast<int64_t>(gridDim.x);

    int32_t round_idx = 0;
    for (int64_t round_start = 0;
         round_start + static_cast<int64_t>(blockIdx.x) < num_tokens;
         round_start += round_stride, ++round_idx) {
        const int32_t slot = round_idx % kNumPipelineSlots;
        if (is_producer_warp) {
            if (round_idx >= kNumPipelineSlots) {
                producer_wait_consumed(slot);
            }
            if (lane == 0) {
                round_starts[slot] = round_start;
            }
            producer_arrive_ready(slot);
        }

        if (is_consumer_warp) {
            consumer_wait_ready(slot);
            const int64_t published_round_start = round_starts[slot];
#pragma unroll
            for (int32_t task_round = 0;
                 task_round < kMaxTasksPerConsumerWarp;
                 task_round += 2) {
                if (task_round >= tasks_per_consumer_warp) {
                    break;
                }
                const int64_t local_task0 =
                    consumer_warp +
                    static_cast<int64_t>(task_round) * kNumConsumerWarps;
                const int64_t local_task1 =
                    local_task0 + kNumConsumerWarps;
                const int64_t token0 =
                    published_round_start +
                    static_cast<int64_t>(blockIdx.x) +
                    local_task0 * static_cast<int64_t>(gridDim.x);
                const int64_t token1 =
                    published_round_start +
                    static_cast<int64_t>(blockIdx.x) +
                    local_task1 * static_cast<int64_t>(gridDim.x);
                const bool valid0 = token0 < num_tokens;
                const bool valid1 =
                    task_round + 1 < tasks_per_consumer_warp and
                    token1 < num_tokens;
                if (valid0 and valid1) {
                    process_two_tokens(
                        topk,
                        topk_stride,
                        combined,
                        combined_stride,
                        lens,
                        lens_stride,
                        compressed_rows,
                        static_cast<int32_t>(token0),
                        static_cast<int32_t>(token1),
                        lane);
                } else if (valid0) {
                    process_one_token(
                        topk,
                        topk_stride,
                        combined,
                        combined_stride,
                        lens,
                        lens_stride,
                        compressed_rows,
                        static_cast<int32_t>(token0),
                        lane);
                }
            }
            if (round_start + 2 * round_stride +
                    static_cast<int64_t>(blockIdx.x) <
                num_tokens) {
                consumer_arrive_consumed(slot);
            }
        }
    }
}

}  // namespace

void launch_combine_indices(
    const int32_t* topk,
    int64_t topk_stride,
    int32_t* combined,
    int64_t combined_stride,
    int32_t* lens,
    int64_t lens_stride,
    int32_t compressed_rows,
    int32_t num_tokens,
    int32_t num_sms,
    cudaStream_t stream) {
    if (num_tokens == 0) {
        return;
    }

    const bool fixed_shape_fast_path =
        num_tokens == kProductionTokens and
        topk_stride == kTopK and
        combined_stride == kCombined and
        lens_stride == 1 and
        (reinterpret_cast<std::uintptr_t>(topk) &
         (alignof(int4) - 1)) == 0 and
        (reinterpret_cast<std::uintptr_t>(combined) &
         (alignof(int4) - 1)) == 0;
    if (fixed_shape_fast_path) {
        if (num_sms == 40) {
            combine_indices_fast_kernel<40, 8><<<
                40, kNumThreads, 0, stream>>>(
                topk, combined, lens, compressed_rows);
        } else if (num_sms == 108) {
            combine_indices_fast_kernel<108, 3><<<
                108, kNumThreads, 0, stream>>>(
                topk, combined, lens, compressed_rows);
        } else if (num_sms == 152) {
            combine_indices_fast_kernel<152, 2><<<
                152, kNumThreads, 0, stream>>>(
                topk, combined, lens, compressed_rows);
        } else {
            combine_indices_fast_kernel<148, 2><<<
                148, kNumThreads, 0, stream>>>(
                topk, combined, lens, compressed_rows);
        }
        return;
    }

    combine_indices_kernel<<<
        static_cast<uint32_t>(num_sms),
        kNumThreads,
        0,
        stream>>>(
        topk,
        topk_stride,
        combined,
        combined_stride,
        lens,
        lens_stride,
        compressed_rows,
        num_tokens);
}
