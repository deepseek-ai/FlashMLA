// SPDX-License-Identifier: Apache-2.0
// Derived from the manifest-selected accepted K12 implementation.
//
// Fixed-shape warp-specialized quantizer.
//
// One producer warp moves a contiguous 112-pack tile through a two-stage
// cp.async.bulk G->S pipeline.  The seven remaining warpgroups are consumers:
// each of their 28 warps owns four complete 512-value packs, processed as two
// strict-order pairs.  Eight simultaneous four-lane subgroups perform each
// pair's eight independent 128-value reductions, raw FP8 stores, and packed
// UE8M0 scale-word stores.  Consequently there is no per-pack cross-warp
// barrier or shared scale assembly.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>

#include "output_quant.h"

namespace {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpsPerWarpgroup = 4;
constexpr uint32_t kThreads = 1024;
constexpr uint32_t kWarps = kThreads / kWarpSize;
constexpr uint32_t kProducerWarp = 0;
constexpr uint32_t kProducerWarps = kWarpsPerWarpgroup;
constexpr uint32_t kConsumerWarpBase = kWarpsPerWarpgroup;
constexpr uint32_t kConsumerWarps = kWarps - kConsumerWarpBase;
constexpr uint32_t kPipelineStages = 2;

constexpr uint32_t kRows = 8192;
constexpr uint32_t kHidden = 16 * 1024;
constexpr uint32_t kPackWidth = 512;
constexpr uint32_t kPacksPerRow = kHidden / kPackWidth;
constexpr uint32_t kGroupWidth = 128;
constexpr uint32_t kGroupsPerPack = kPackWidth / kGroupWidth;
constexpr uint32_t kSubgroupWidth = 4;
constexpr uint32_t kPacksPerConsumer = 4;
constexpr uint32_t kTotalPacks = kRows * kPacksPerRow;

constexpr uint32_t kTilePacks = kConsumerWarps * kPacksPerConsumer;
constexpr uint32_t kBytesPerPack = kPackWidth * sizeof(__nv_bfloat16);
constexpr uint32_t kTileBytes = kTilePacks * kBytesPerPack;
constexpr uint32_t kMaxBulkBytes = 16 * 1024;
constexpr float kFp8Max = 448.0F;
constexpr float kEpsilon = 1.0e-10F;

static_assert(kWarps == 32);
static_assert(kConsumerWarps == 28);
static_assert(kGroupsPerPack * kSubgroupWidth * 2 == kWarpSize);
static_assert(kTotalPacks % kPacksPerConsumer == 0);
static_assert(kTilePacks == 112);
static_assert(kTileBytes == 112 * 1024);
static_assert(kPacksPerRow == 32);
static_assert(kTileBytes % 16 == 0);

struct alignas(128) PipelineStorage {
    alignas(128) uint8_t input[kPipelineStages][kTileBytes];
    alignas(8) uint64_t ready[kPipelineStages];
    alignas(8) uint64_t empty[kPipelineStages];
};

static_assert(alignof(PipelineStorage) == 128);
static_assert(sizeof(PipelineStorage) == 229504);

union alignas(16) PackedBfloat16x8 {
    uint4 vector;
    uint32_t word[4];
    __nv_bfloat162 pair[4];
};

union alignas(16) PackedFp8x16 {
    uint4 vector;
    uint16_t pair[8];
};

static_assert(sizeof(PackedBfloat16x8) == sizeof(uint4));
static_assert(sizeof(PackedFp8x16) == sizeof(uint4));

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrier_init(
    uint64_t* barrier,
    uint32_t arrival_count) {
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :
        : "r"(address), "r"(arrival_count)
        : "memory");
}

__device__ __forceinline__ void mbarrier_init_fence() {
    asm volatile(
        "fence.mbarrier_init.release.cluster;"
        :
        :
        : "memory");
}

__device__ __forceinline__ void producer_warpgroup_sync() {
    asm volatile("bar.sync 1, 128;" : : : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* barrier,
    uint32_t bytes) {
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :
        : "r"(address), "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* barrier) {
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        :
        : "r"(address)
        : "memory");
}

__device__ __forceinline__ void mbarrier_wait(
    const uint64_t* barrier,
    uint32_t phase) {
    const uint32_t address = shared_address(barrier);
    const uint32_t ticks = 0x989680U;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 "
        "P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(address), "r"(phase), "r"(ticks)
        : "memory");
}

__device__ __forceinline__ void bulk_copy_global_to_shared(
    void* shared_destination,
    const void* global_source,
    uint32_t bytes,
    uint64_t* ready_barrier) {
    const uint32_t destination = shared_address(shared_destination);
    const uint32_t barrier = shared_address(ready_barrier);
    asm volatile(
        "cp.async.bulk.shared::cluster.global."
        "mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(destination),
          "l"(global_source),
          "r"(bytes),
          "r"(barrier)
        : "memory");
}

// Triton's tl.max/tl.maximum default to maxNum semantics. Mapping every BF16
// NaN to zero before the unsigned magnitude maximum therefore preserves both
// cases: numeric values (including infinity) win over NaN, while an all-NaN
// group falls through to epsilon in exact_power_of_two_scale().
__device__ __forceinline__ uint32_t bf16_abs_magnitude(uint32_t bits) {
    const uint32_t magnitude = bits & 0x7FFFU;
    return magnitude > 0x7F80U ? 0U : magnitude;
}

__device__ __forceinline__ uint32_t subgroup_max_bits(
    uint32_t value,
    uint32_t subgroup_mask) {
#pragma unroll
    for (uint32_t offset = kSubgroupWidth / 2;
         offset != 0;
         offset >>= 1) {
        const uint32_t other = __shfl_xor_sync(
            subgroup_mask, value, static_cast<int>(offset), kSubgroupWidth);
        value = value < other ? other : value;
    }
    return value;
}

__device__ __forceinline__ float multiply_rn(float lhs, float rhs) {
    float result;
    asm(
        "mul.rn.f32 %0, %1, %2;"
        : "=f"(result)
        : "f"(lhs), "f"(rhs));
    return result;
}

// The input magnitude is BF16.  Therefore max(abs(x), eps) / 448 is a
// positive normal FP32 value and ceil(log2(v)) equals the biased exponent plus
// one iff the mantissa is nonzero.  This is the exact scale identity already
// exhaustively checked by earlier implementation over all finite BF16 magnitudes.
__device__ __forceinline__ void exact_power_of_two_scale(
    float absmax,
    float& inverse_scale,
    uint32_t& scale_byte) {
    const float bounded = fmaxf(absmax, kEpsilon);
    const float raw_scale = multiply_rn(bounded, 1.0F / kFp8Max);
    const uint32_t raw_bits = __float_as_uint(raw_scale);
    scale_byte =
        (raw_bits >> 23) + ((raw_bits & 0x007FFFFFU) != 0 ? 1U : 0U);
    inverse_scale = scale_byte < 0xFFU
        ? __uint_as_float((254U - scale_byte) << 23)
        : 0.0F;
}

__device__ __forceinline__ float normalize_e4m3(
    float value,
    float inverse_scale) {
    const float normalized = multiply_rn(value, inverse_scale);
    // Triton's clamp lowers to the maxNum-equivalent upper bound first,
    // then the lower bound.  The order is observable only for NaN:
    // minNum(+448, NaN) -> +448, whereas the reverse ordering produces -448.
    return fmaxf(-kFp8Max, fminf(kFp8Max, normalized));
}

__device__ __forceinline__ uint16_t quantize_two(
    float2 values,
    float inverse_scale) {
    const float2 normalized = {
        normalize_e4m3(values.x, inverse_scale),
        normalize_e4m3(values.y, inverse_scale),
    };
    return static_cast<uint16_t>(
        __nv_cvt_float2_to_fp8x2(
            normalized,
            __NV_SATFINITE,
            __NV_E4M3));
}

__device__ __forceinline__ void quantize_pack_pair(
    const __nv_bfloat16* shared_pack_pair,
    uint8_t* output_pack_pair,
    int32_t* scales,
    uint64_t task_idx,
    uint32_t lane_idx) {
    const uint32_t group_slot = lane_idx / kSubgroupWidth;
    const uint32_t pack_within = group_slot / kGroupsPerPack;
    const uint32_t group_idx = group_slot % kGroupsPerPack;
    const uint32_t lane_in_group = lane_idx % kSubgroupWidth;
    const uint32_t subgroup_mask = 0xFU << (lane_idx & 28U);
    const uint32_t half_warp_mask =
        lane_idx < 16 ? 0x0000FFFFU : 0xFFFF0000U;
    const auto* input_vectors = reinterpret_cast<const uint4*>(
        shared_pack_pair +
        pack_within * kPackWidth +
        group_idx * kGroupWidth);
    PackedBfloat16x8 fragments[4];
#pragma unroll
    for (uint32_t fragment_idx = 0; fragment_idx < 4; ++fragment_idx) {
        fragments[fragment_idx].vector =
            input_vectors[lane_in_group * 4 + fragment_idx];
    }

    uint32_t local_max_bits = 0;
#pragma unroll
    for (uint32_t fragment_idx = 0; fragment_idx < 4; ++fragment_idx) {
#pragma unroll
        for (uint32_t word_idx = 0; word_idx < 4; ++word_idx) {
            const uint32_t word = fragments[fragment_idx].word[word_idx];
            const uint32_t low = bf16_abs_magnitude(word);
            const uint32_t high = bf16_abs_magnitude(word >> 16);
            local_max_bits =
                local_max_bits < low ? low : local_max_bits;
            local_max_bits =
                local_max_bits < high ? high : local_max_bits;
        }
    }
    const uint32_t max_bits = subgroup_max_bits(
        local_max_bits,
        subgroup_mask);
    const float absmax = __uint_as_float(max_bits << 16);

    float inverse_scale;
    uint32_t scale_byte;
    exact_power_of_two_scale(absmax, inverse_scale, scale_byte);

    auto* output_vectors = reinterpret_cast<uint4*>(
        output_pack_pair +
        pack_within * kPackWidth +
        group_idx * kGroupWidth);
#pragma unroll
    for (uint32_t output_idx = 0; output_idx < 2; ++output_idx) {
        PackedFp8x16 packed_output;
#pragma unroll
        for (uint32_t fragment_idx = 0; fragment_idx < 2;
             ++fragment_idx) {
#pragma unroll
            for (uint32_t pair_idx = 0; pair_idx < 4; ++pair_idx) {
                const uint32_t input_idx =
                    output_idx * 2 + fragment_idx;
                const float2 values = __bfloat1622float2(
                    fragments[input_idx].pair[pair_idx]);
                packed_output.pair[fragment_idx * 4 + pair_idx] =
                    quantize_two(values, inverse_scale);
            }
        }
        output_vectors[lane_in_group * 2 + output_idx] =
            packed_output.vector;
    }

    uint32_t packed_scale = lane_in_group == 0
        ? static_cast<uint32_t>(scale_byte) << (group_idx * 8)
        : 0U;
    packed_scale = __reduce_or_sync(half_warp_mask, packed_scale);

    if (lane_idx == 0 || lane_idx == 16) {
        const uint64_t scale_task_idx = task_idx + pack_within;
        const uint32_t row_idx = static_cast<uint32_t>(scale_task_idx >> 5);
        const uint32_t pack_idx =
            static_cast<uint32_t>(
                scale_task_idx & (kPacksPerRow - 1));
        const uint64_t word_offset =
            static_cast<uint64_t>(pack_idx) * kRows + row_idx;
        reinterpret_cast<uint32_t*>(scales)[word_offset] =
            packed_scale;
    }
}

__global__ __launch_bounds__(kThreads, 1)
void output_quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    int32_t* __restrict__ scales) {
    extern __shared__ __align__(128) uint8_t dynamic_shared[];
    auto& storage =
        *reinterpret_cast<PipelineStorage*>(dynamic_shared);

    const uint32_t thread_idx = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp_idx = thread_idx / kWarpSize;
    const uint32_t lane_idx = thread_idx % kWarpSize;

    if (thread_idx == 0) {
#pragma unroll
        for (uint32_t stage = 0; stage < kPipelineStages; ++stage) {
            mbarrier_init(&storage.ready[stage], 1);
            mbarrier_init(&storage.empty[stage], kConsumerWarps);
        }
        mbarrier_init_fence();
    }
    __syncthreads();

    const uint64_t total_tiles =
        (static_cast<uint64_t>(kTotalPacks) + kTilePacks - 1) /
        kTilePacks;

    if (warp_idx < kProducerWarps) {
        uint32_t iteration = 0;
        for (uint64_t tile_idx = blockIdx.x;
             tile_idx < total_tiles;
             tile_idx += gridDim.x, ++iteration) {
            const uint32_t stage = iteration & (kPipelineStages - 1);
            if (iteration >= kPipelineStages) {
                const uint32_t empty_phase =
                    ((iteration / kPipelineStages) - 1U) & 1U;
                if (warp_idx == kProducerWarp && lane_idx == 0) {
                    mbarrier_wait(
                        &storage.empty[stage],
                        empty_phase);
                }
            }
            const uint64_t first_task = tile_idx * kTilePacks;
            const uint64_t remaining_packs =
                static_cast<uint64_t>(kTotalPacks) - first_task;
            const uint32_t valid_packs = static_cast<uint32_t>(
                remaining_packs < kTilePacks
                    ? remaining_packs
                    : kTilePacks);
            const uint32_t total_bytes = valid_packs * kBytesPerPack;
            if (warp_idx == kProducerWarp && lane_idx == 0) {
                mbarrier_arrive_expect_tx(
                    &storage.ready[stage],
                    total_bytes);
            }
            producer_warpgroup_sync();
            if (lane_idx == 0) {
#pragma unroll
                for (uint32_t transaction = warp_idx;
                     transaction < 7;
                     transaction += kProducerWarps) {
                    const uint32_t byte_offset =
                        transaction * kMaxBulkBytes;
                    if (byte_offset >= total_bytes) {
                        continue;
                    }
                    const uint32_t bytes_left =
                        total_bytes - byte_offset;
                    const uint32_t transaction_bytes =
                        bytes_left < kMaxBulkBytes
                            ? bytes_left
                            : kMaxBulkBytes;
                    bulk_copy_global_to_shared(
                        storage.input[stage] + byte_offset,
                        input + first_task * kPackWidth +
                            byte_offset / sizeof(__nv_bfloat16),
                        transaction_bytes,
                        &storage.ready[stage]);
                }
            }
        }
        return;
    }

    if (warp_idx < kConsumerWarpBase) {
        return;
    }

    const uint32_t consumer_idx = warp_idx - kConsumerWarpBase;
    uint32_t iteration = 0;
    for (uint64_t tile_idx = blockIdx.x;
         tile_idx < total_tiles;
         tile_idx += gridDim.x, ++iteration) {
        const uint32_t stage = iteration & (kPipelineStages - 1);
        const uint32_t ready_phase =
            (iteration / kPipelineStages) & 1U;
        if (lane_idx == 0) {
            mbarrier_wait(
                &storage.ready[stage],
                ready_phase);
        }
        __syncwarp();

        const uint64_t task_idx =
            tile_idx * kTilePacks +
            consumer_idx * kPacksPerConsumer;
#pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < 2; ++pair_idx) {
            const uint64_t pair_task_idx =
                task_idx + pair_idx * 2;
            if (pair_task_idx < kTotalPacks) {
                const auto* shared_pack_pair =
                    reinterpret_cast<const __nv_bfloat16*>(
                        storage.input[stage] +
                        (consumer_idx * kPacksPerConsumer +
                         pair_idx * 2) *
                        kBytesPerPack);
                quantize_pack_pair(
                    shared_pack_pair,
                    output + pair_task_idx * kPackWidth,
                    scales,
                    pair_task_idx,
                    lane_idx);
            }
        }
        __syncwarp();
        if (lane_idx == 0) {
            mbarrier_arrive(&storage.empty[stage]);
        }
    }
}

}  // namespace

cudaError_t launch_output_quant(
    const void* input,
    uint8_t* output,
    int32_t* scales,
    uint32_t num_sms,
    cudaStream_t stream) {
    static std::once_flag attribute_once;
    static cudaError_t attribute_status = cudaSuccess;
    std::call_once(attribute_once, [] {
        attribute_status = cudaFuncSetAttribute(
            output_quant_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sizeof(PipelineStorage));
        if (attribute_status == cudaSuccess) {
            attribute_status = cudaFuncSetAttribute(
                output_quant_kernel,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxShared);
        }
    });
    if (attribute_status != cudaSuccess) {
        return attribute_status;
    }

    output_quant_kernel<<<
        num_sms,
        kThreads,
        sizeof(PipelineStorage),
        stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        output,
        scales);
    return cudaGetLastError();
}
