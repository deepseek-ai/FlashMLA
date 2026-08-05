#include "state_save.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

#define STATE_SAVE_DEVICE_ASSERT(CONDITION) \
    do { if (!(CONDITION)) asm volatile("trap;"); } while (false)

namespace flash_mla_csa {

__device__ __forceinline__ void named_barrier_sync(
        uint32_t thread_count, uint32_t barrier_idx) {
    asm volatile("bar.sync %0, %1;" ::
        "r"(barrier_idx), "r"(thread_count) : "memory");
}

// One producer warp publishes 28 independent token tasks.  Seven complete
// consumer warpgroups consume one token per warp.  Warps 1--3 in WG0 stay out
// of the named barriers.  The wider CTA is required to saturate a 40-SM lane:
// the previous single-consumer-warpgroup body exposed only four memory warps
// per resident SM and was 10x slower than the one-program-per-token oracle.
constexpr uint32_t kStateSaveNumThreads = 1024;
constexpr uint32_t kStateSaveNumProducerThreads = 32;
constexpr uint32_t kStateSaveNumConsumerThreads = 896;
constexpr uint32_t kStateSaveNumConsumerWarps = 28;
constexpr uint32_t kStateSaveTasksPerConsumerWarp = 2;
constexpr uint32_t kStateSaveNumTasksPerIteration =
    kStateSaveNumConsumerWarps * kStateSaveTasksPerConsumerWarp;
constexpr uint32_t kStateSaveNumBarrierThreads =
    kStateSaveNumProducerThreads + kStateSaveNumConsumerThreads;
constexpr uint32_t kStateSaveTaskBarrierIdx = 0;

struct alignas(16) StateSaveTask {
    uint64_t state_offset;
    uint32_t token_idx;
    uint32_t ape_row_idx;
    uint32_t valid;
    uint32_t padding;
};

template <typename Element>
__device__ __forceinline__ void state_save_copy_and_add(
        const Element* kv_row,
        const Element* score_row,
        const Element* ape_row,
        Element* state_row,
        const uint32_t lane_idx,
        const uint32_t head_size,
        const uint32_t state_width) {
    if constexpr (std::is_same_v<Element, float>) {
        STATE_SAVE_DEVICE_ASSERT(head_size % 4 == 0);
        const auto* kv_vec = reinterpret_cast<const uint4*>(kv_row);
        const auto* score_vec = reinterpret_cast<const float4*>(score_row);
        const auto* ape_vec = reinterpret_cast<const float4*>(ape_row);
        auto* state_kv_vec = reinterpret_cast<uint4*>(state_row);
        auto* state_score_vec = reinterpret_cast<float4*>(
            state_row + state_width);

        const auto num_vector_idx = head_size / 4;
        for (uint32_t vector_idx = lane_idx;
             vector_idx < num_vector_idx;
             vector_idx += 32) {
            // The KV path is a raw 128-bit copy, preserving every FP32 bit.
            state_kv_vec[vector_idx] = kv_vec[vector_idx];

            const auto score = score_vec[vector_idx];
            const auto ape = ape_vec[vector_idx];
            state_score_vec[vector_idx] = make_float4(
                score.x + ape.x,
                score.y + ape.y,
                score.z + ape.z,
                score.w + ape.w);
        }
    } else {
        static_assert(std::is_same_v<Element, __nv_bfloat16>);
        STATE_SAVE_DEVICE_ASSERT(head_size % 8 == 0);
        const auto* kv_vec = reinterpret_cast<const uint4*>(kv_row);
        auto* state_kv_vec = reinterpret_cast<uint4*>(state_row);
        auto* state_score_vec = reinterpret_cast<uint4*>(
            state_row + state_width);

        const auto num_vector_idx = head_size / 8;
        for (uint32_t vector_idx = lane_idx;
             vector_idx < num_vector_idx;
             vector_idx += 32) {
            // The KV path is a raw 128-bit copy, preserving every BF16 bit.
            state_kv_vec[vector_idx] = kv_vec[vector_idx];

            uint4 packed_sum;
            auto* sum = reinterpret_cast<__nv_bfloat16*>(&packed_sum);
            const auto element_idx = vector_idx * 8;
#pragma unroll
            for (uint32_t value_idx = 0; value_idx < 8; ++value_idx) {
                // Triton promotes BF16 arithmetic to FP32 and rounds once at
                // the BF16 store.  Spell both conversions out to retain that
                // exact rounding point.
                const auto score = __bfloat162float(
                    score_row[element_idx + value_idx]);
                const auto ape = __bfloat162float(
                    ape_row[element_idx + value_idx]);
                sum[value_idx] = __float2bfloat16_rn(score + ape);
            }
            state_score_vec[vector_idx] = packed_sum;
        }
    }
}

__device__ __forceinline__ void state_save_copy_and_add_pair_fp32(
        const float* kv_row0,
        const float* score_row0,
        const float* ape_row0,
        float* state_row0,
        const float* kv_row1,
        const float* score_row1,
        const float* ape_row1,
        float* state_row1,
        const uint32_t lane_idx,
        const uint32_t head_size,
        const uint32_t state_width) {
    STATE_SAVE_DEVICE_ASSERT(head_size % 4 == 0);
    const auto* kv_vec0 = reinterpret_cast<const uint4*>(kv_row0);
    const auto* score_vec0 = reinterpret_cast<const float4*>(score_row0);
    const auto* ape_vec0 = reinterpret_cast<const float4*>(ape_row0);
    auto* state_kv_vec0 = reinterpret_cast<uint4*>(state_row0);
    auto* state_score_vec0 = reinterpret_cast<float4*>(
        state_row0 + state_width);
    const auto* kv_vec1 = reinterpret_cast<const uint4*>(kv_row1);
    const auto* score_vec1 = reinterpret_cast<const float4*>(score_row1);
    const auto* ape_vec1 = reinterpret_cast<const float4*>(ape_row1);
    auto* state_kv_vec1 = reinterpret_cast<uint4*>(state_row1);
    auto* state_score_vec1 = reinterpret_cast<float4*>(
        state_row1 + state_width);

    const auto num_vector_idx = head_size / 4;
    for (uint32_t vector_idx = lane_idx;
         vector_idx < num_vector_idx;
         vector_idx += 32) {
        // Load both independent rows before either store.  This keeps two
        // memory streams in flight per consumer warp without changing the
        // per-row arithmetic or rounding order.
        const auto kv0 = kv_vec0[vector_idx];
        const auto score0 = score_vec0[vector_idx];
        const auto ape0 = ape_vec0[vector_idx];
        const auto kv1 = kv_vec1[vector_idx];
        const auto score1 = score_vec1[vector_idx];
        const auto ape1 = ape_vec1[vector_idx];
        state_kv_vec0[vector_idx] = kv0;
        state_score_vec0[vector_idx] = make_float4(
            score0.x + ape0.x,
            score0.y + ape0.y,
            score0.z + ape0.z,
            score0.w + ape0.w);
        state_kv_vec1[vector_idx] = kv1;
        state_score_vec1[vector_idx] = make_float4(
            score1.x + ape1.x,
            score1.y + ape1.y,
            score1.z + ape1.z,
            score1.w + ape1.w);
    }
}

template <
    uint32_t kHeadSize,
    uint32_t kStateWidth,
    uint32_t kCompressRatio,
    typename Element>
__global__ void __launch_bounds__(kStateSaveNumThreads, 1)
state_save_kernel(
        const Element* kv,
        const uint64_t kv_stride,
        const Element* score,
        const uint64_t score_stride,
        const Element* ape,
        const uint64_t ape_stride,
        const int64_t* positions,
        Element* state_cache,
        const uint64_t state_cache_stride0,
        const uint64_t state_cache_stride1,
        const int64_t* slot_mapping,
        const uint32_t num_tokens,
        const uint32_t block_size) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    static_assert(kHeadSize > 0);
    static_assert(kStateWidth >= kHeadSize);
    static_assert(kCompressRatio > 0);
    static_assert(kHeadSize % (std::is_same_v<Element, float> ? 4 : 8) == 0);

    __shared__ StateSaveTask tasks[2][kStateSaveNumTasksPerIteration];

    const auto warp_idx = static_cast<uint32_t>(threadIdx.x) / 32;
    const auto lane_idx = static_cast<uint32_t>(threadIdx.x) % 32;
    const auto is_producer_warp = warp_idx == 0;
    const auto is_consumer_warp = warp_idx >= 4;

    // Only warp 0 and the seven complete consumer warpgroups participate.
    if (not is_producer_warp and not is_consumer_warp)
        return;

    const auto task_stride =
        static_cast<uint32_t>(gridDim.x) * kStateSaveNumTasksPerIteration;
    auto publish_task = [&](const uint32_t task_base, const uint32_t slot) {
        if (is_producer_warp and
            lane_idx < kStateSaveNumConsumerWarps) {
#pragma unroll
            for (uint32_t task_group = 0;
                 task_group < kStateSaveTasksPerConsumerWarp;
                 ++task_group) {
                const auto task_idx =
                    lane_idx + task_group * kStateSaveNumConsumerWarps;
                const auto token_idx = task_base + task_idx;
                StateSaveTask task;
            task.state_offset = 0;
            task.token_idx = token_idx;
            task.ape_row_idx = 0;
            task.valid = 0;
            task.padding = 0;

            if (token_idx < num_tokens) {
                const auto slot_idx = slot_mapping[token_idx];
                if (slot_idx >= 0) {
                    const auto block_idx =
                        static_cast<uint64_t>(slot_idx) / block_size;
                    const auto pos_in_block_idx =
                        static_cast<uint64_t>(slot_idx) % block_size;
                    const auto position = positions[token_idx];
                    const auto signed_ape_row_idx =
                        position % static_cast<int64_t>(kCompressRatio);
                    task.state_offset =
                        block_idx * state_cache_stride0 +
                        pos_in_block_idx * state_cache_stride1;
                    task.ape_row_idx = static_cast<uint32_t>(
                        signed_ape_row_idx < 0
                            ? signed_ape_row_idx + kCompressRatio
                            : signed_ape_row_idx);
                    task.valid = 1;
                }
            }
                tasks[slot][task_idx] = task;
            }
        }
    };

    uint32_t task_base =
        static_cast<uint32_t>(blockIdx.x) *
        kStateSaveNumTasksPerIteration;
    uint32_t slot = 0;
    publish_task(task_base, slot);

    // Publish the first batch.  Subsequent handoffs use one barrier per batch:
    // producer warp 0 fills the alternate descriptor buffer while the seven
    // consumer warpgroups copy the current 56 tokens.
    named_barrier_sync(
        kStateSaveNumBarrierThreads,
        kStateSaveTaskBarrierIdx);

    for (; task_base < num_tokens; task_base += task_stride) {
        const auto next_task_base = task_base + task_stride;
        const auto next_slot = slot ^ 1;
        publish_task(next_task_base, next_slot);

        if (is_consumer_warp) {
            const auto consumer_warp_idx = warp_idx - 4;
            if constexpr (std::is_same_v<Element, float>) {
                const auto task0 = tasks[slot][consumer_warp_idx];
                const auto task1 = tasks[slot][
                    consumer_warp_idx + kStateSaveNumConsumerWarps];
                if (task0.valid != 0 and task1.valid != 0) {
                    state_save_copy_and_add_pair_fp32(
                        kv + static_cast<uint64_t>(task0.token_idx) * kv_stride,
                        score + static_cast<uint64_t>(task0.token_idx) * score_stride,
                        ape + static_cast<uint64_t>(task0.ape_row_idx) * ape_stride,
                        state_cache + task0.state_offset,
                        kv + static_cast<uint64_t>(task1.token_idx) * kv_stride,
                        score + static_cast<uint64_t>(task1.token_idx) * score_stride,
                        ape + static_cast<uint64_t>(task1.ape_row_idx) * ape_stride,
                        state_cache + task1.state_offset,
                        lane_idx, kHeadSize, kStateWidth);
                } else {
#pragma unroll
                    for (uint32_t task_group = 0;
                         task_group < kStateSaveTasksPerConsumerWarp;
                         ++task_group) {
                        const auto task_idx =
                            consumer_warp_idx +
                            task_group * kStateSaveNumConsumerWarps;
                        const auto task = tasks[slot][task_idx];
                        if (task.valid != 0) {
                            state_save_copy_and_add(
                                kv + static_cast<uint64_t>(task.token_idx) * kv_stride,
                                score + static_cast<uint64_t>(task.token_idx) * score_stride,
                                ape + static_cast<uint64_t>(task.ape_row_idx) * ape_stride,
                                state_cache + task.state_offset,
                                lane_idx, kHeadSize, kStateWidth);
                        }
                    }
                }
            } else {
#pragma unroll
                for (uint32_t task_group = 0;
                     task_group < kStateSaveTasksPerConsumerWarp;
                     ++task_group) {
                    const auto task_idx =
                        consumer_warp_idx +
                        task_group * kStateSaveNumConsumerWarps;
                    const auto task = tasks[slot][task_idx];
                    if (task.valid != 0) {
                        state_save_copy_and_add(
                            kv + static_cast<uint64_t>(task.token_idx) * kv_stride,
                            score + static_cast<uint64_t>(task.token_idx) * score_stride,
                            ape + static_cast<uint64_t>(task.ape_row_idx) * ape_stride,
                            state_cache + task.state_offset,
                            lane_idx, kHeadSize, kStateWidth);
                    }
                }
            }
        }

        // Complete the current copy and publish the alternate descriptor set.
        named_barrier_sync(
            kStateSaveNumBarrierThreads,
            kStateSaveTaskBarrierIdx);
        slot = next_slot;
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        STATE_SAVE_DEVICE_ASSERT(false and "This kernel only supports SM100");
#endif
}

}  // namespace flash_mla_csa

namespace {
constexpr uint32_t kHeadSize = 1024;
constexpr uint32_t kStateWidth = 1024;
constexpr uint32_t kCompressRatio = 4;

template <typename Element>
cudaError_t launch_state_save_typed(
        const Element* kv, uint64_t kv_stride,
        const Element* score, uint64_t score_stride,
        const Element* ape, uint64_t ape_stride,
        const int64_t* positions, Element* state_cache,
        uint64_t state_stride0, uint64_t state_stride1,
        const int64_t* slot_mapping, uint32_t num_tokens,
        uint32_t block_size, uint32_t num_sms, cudaStream_t stream) {
    if (num_tokens == 0 || num_sms == 0)
        return cudaSuccess;
    flash_mla_csa::state_save_kernel<
        kHeadSize, kStateWidth, kCompressRatio, Element>
        <<<num_sms, flash_mla_csa::kStateSaveNumThreads, 0, stream>>>(
            kv, kv_stride, score, score_stride, ape, ape_stride,
            positions, state_cache, state_stride0, state_stride1,
            slot_mapping, num_tokens, block_size);
    return cudaGetLastError();
}
}  // namespace

cudaError_t launch_state_save_fp32(
        const float* kv, uint64_t kv_stride,
        const float* score, uint64_t score_stride,
        const float* ape, uint64_t ape_stride,
        const int64_t* positions, float* state_cache,
        uint64_t state_stride0, uint64_t state_stride1,
        const int64_t* slot_mapping, uint32_t num_tokens,
        uint32_t block_size, uint32_t num_sms, cudaStream_t stream) {
    return launch_state_save_typed(
        kv, kv_stride, score, score_stride, ape, ape_stride,
        positions, state_cache, state_stride0, state_stride1,
        slot_mapping, num_tokens, block_size, num_sms, stream);
}

cudaError_t launch_state_save_bf16(
        const void* kv, uint64_t kv_stride,
        const void* score, uint64_t score_stride,
        const void* ape, uint64_t ape_stride,
        const int64_t* positions, void* state_cache,
        uint64_t state_stride0, uint64_t state_stride1,
        const int64_t* slot_mapping, uint32_t num_tokens,
        uint32_t block_size, uint32_t num_sms, cudaStream_t stream) {
    return launch_state_save_typed(
        static_cast<const __nv_bfloat16*>(kv), kv_stride,
        static_cast<const __nv_bfloat16*>(score), score_stride,
        static_cast<const __nv_bfloat16*>(ape), ape_stride,
        positions, static_cast<__nv_bfloat16*>(state_cache),
        state_stride0, state_stride1, slot_mapping, num_tokens,
        block_size, num_sms, stream);
}

#undef STATE_SAVE_DEVICE_ASSERT
