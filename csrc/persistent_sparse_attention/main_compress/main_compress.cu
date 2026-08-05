#include "main_compress.h"
#include <cuda_bf16.h>
#include <cstdint>

#define MAIN_COMPRESS_DEVICE_ASSERT(CONDITION) \
    do { if (!(CONDITION)) asm volatile("trap;"); } while (false)

namespace flash_mla_csa {

// CuTeDSL-shaped direct-LDG resident-grid kernel.
//
// A 1024-thread CTA is sixteen independent 64-thread teams. Each team owns one
// C4 boundary ordinal at a time and advances by gridDim.x * 16 ordinals. The
// two warps in a team jointly cover all 512 head elements, exactly like the
// two-warp CuTeDSL CTA. There is no CTA scheduler, producer warpgroup, TMA
// ring, transaction barrier, or CTA-wide batch barrier.
//
// State values and scores move directly from global memory into registers
// through 128-bit float4 loads. Shared memory contains only per-team paged
// block numbers and the two-warp RMS reduction scalars. The online-softmax,
// RMSNorm, RoPE, BF16-round-before-FP8, and store order match the byte-exact
// production CuTeDSL path.
namespace main_compress_detail {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kNumWarpsPerTask = 2;
constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kStateSliceWidth = 256;
constexpr uint32_t kStateWidth = 4 * kStateSliceWidth;
constexpr uint32_t kCompressRatio = 4;
constexpr uint32_t kWindow = 2 * kCompressRatio;
constexpr uint32_t kRopeHeadDim = 64;
constexpr uint32_t kNopeHeadDim = kHeadDim - kRopeHeadDim;
constexpr uint32_t kQuantBlock = 64;
constexpr uint32_t kTokenStride = 576;
constexpr uint32_t kScaleDim = 8;
constexpr uint32_t kKVBlockSize = 64;
constexpr uint32_t kElemsPerLane = 8;
constexpr float kFP8Max = 448.0f;
constexpr float kMinScale = 1.0e-4f;

static_assert(kStateWidth == 2 * kHeadDim);
static_assert(kNopeHeadDim % kQuantBlock == 0);

__device__ __forceinline__ float exp2_approx(float value) {
    float result;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float softmax_exp(float value) {
    return exp2_approx(value * __int_as_float(0x3fb8aa3b));
}

__device__ __forceinline__ float warp_bfly_sum(float value) {
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ uint32_t cvt_fp32x2_to_bf16x2(
        float low, float high) {
    uint32_t packed;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed) :
        "f"(low), "f"(high));
    return packed;
}

__device__ __forceinline__ uint16_t cvt_fp32x2_to_fp8e4m3x2(
        float low, float high) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(packed) :
        "f"(low), "f"(high));
    return packed;
}

template <bool kRMSWeightBF16>
__device__ __forceinline__ float load_rms_weight(
        const void* rms_weight, uint32_t element_idx) {
    if constexpr (kRMSWeightBF16) {
        return __bfloat162float(
            reinterpret_cast<const nv_bfloat16*>(rms_weight)[element_idx]);
    }
    return reinterpret_cast<const float*>(rms_weight)[element_idx];
}

constexpr uint32_t kNumThreads = 1024;
constexpr uint32_t kNumTeams = 16;
constexpr uint32_t kNumThreadsPerTeam = 64;
constexpr uint32_t kNumWarpsPerTeam = 2;
constexpr uint32_t kTargetRegistersPerThread = 64;

static_assert(kNumTeams * kNumThreadsPerTeam == kNumThreads);
static_assert(kNumThreadsPerTeam == kNumWarpsPerTeam * kWarpSize);
static_assert(kNumWarpsPerTeam == kNumWarpsPerTask);
static_assert(kNumThreads * kTargetRegistersPerThread == 65536);

struct SharedStorage {
    alignas(16) int32_t block_numbers[kNumTeams][kWindow];
    alignas(16) float
        partial_sums[kNumTeams][kNumWarpsPerTeam];
    alignas(16) float rrms[kNumTeams];
};

constexpr uint32_t kSharedStorageBytes =
    sizeof(SharedStorage);

static_assert(kSharedStorageBytes == 704);

__device__ __forceinline__ void team_sync(const uint32_t team_idx) {
    const auto barrier_idx = team_idx;
    asm volatile(
        "bar.sync %0, %1;"
        :
        : "r"(barrier_idx), "r"(kNumThreadsPerTeam)
        : "memory");
}

#define MAIN_COMPRESS_ONLINE_UPDATE(                                      \
        SCORE, VALUE, MAX_VALUE, SUM_VALUE, PRODUCT_VALUE)             \
    do {                                                               \
        const auto update_score = (SCORE);                             \
        const auto update_value = (VALUE);                             \
        const auto update_new_max =                                    \
            fmaxf((MAX_VALUE), update_score);                          \
        const auto update_old_scale =                                  \
            softmax_exp((MAX_VALUE) - update_new_max);                 \
        const auto update_new_scale =                                  \
            softmax_exp(update_score - update_new_max);                \
        (SUM_VALUE) = fmaf(                                            \
            (SUM_VALUE), update_old_scale, update_new_scale);          \
        (PRODUCT_VALUE) = fmaf(                                        \
            (PRODUCT_VALUE), update_old_scale,                         \
            update_value * update_new_scale);                          \
        (MAX_VALUE) = update_new_max;                                  \
    } while (false)

#define MAIN_COMPRESS_ROPE_PAIR(OFFSET, REAL_PRODUCT, IMAG_PRODUCT)       \
    do {                                                               \
        const auto element_idx = element_base_idx + (OFFSET);          \
        const auto real_input =                                        \
            (REAL_PRODUCT) * rrms *                                    \
            load_rms_weight<kRMSWeightBF16>(                           \
                rms_weight, element_idx);                              \
        const auto imag_input =                                        \
            (IMAG_PRODUCT) * rrms *                                    \
            load_rms_weight<kRMSWeightBF16>(                           \
                rms_weight, element_idx + 1);                          \
        const auto pair_idx =                                          \
            (element_base_idx - kNopeHeadDim) / 2 + (OFFSET) / 2;     \
        const auto cos_value = cos_sin[pair_idx];                      \
        const auto sin_value =                                         \
            cos_sin[kRopeHeadDim / 2 + pair_idx];                      \
        const auto real =                                              \
            real_input * cos_value - imag_input * sin_value;           \
        const auto imag =                                              \
            real_input * sin_value + imag_input * cos_value;           \
        const auto packed = cvt_fp32x2_to_bf16x2(real, imag);          \
        const auto value_byte_offset =                                   \
            kNopeHeadDim +                                             \
            (element_base_idx - kNopeHeadDim + (OFFSET)) *            \
                sizeof(nv_bfloat16);                                   \
        *reinterpret_cast<uint32_t*>(                                  \
            value_ptr + value_byte_offset) = packed;                     \
    } while (false)

#define MAIN_COMPRESS_ROUND_PAIR(OFFSET, LOW_PRODUCT, HIGH_PRODUCT)       \
    do {                                                               \
        const auto low =                                               \
            (LOW_PRODUCT) * rrms *                                     \
            load_rms_weight<kRMSWeightBF16>(                           \
                rms_weight, element_base_idx + (OFFSET));              \
        const auto high =                                              \
            (HIGH_PRODUCT) * rrms *                                    \
            load_rms_weight<kRMSWeightBF16>(                           \
                rms_weight, element_base_idx + (OFFSET) + 1);          \
        const auto packed = cvt_fp32x2_to_bf16x2(low, high);           \
        const auto rounded_low =                                       \
            __uint_as_float(packed << 16);                             \
        const auto rounded_high =                                      \
            __uint_as_float(packed & 0xffff0000u);                     \
        (LOW_PRODUCT) = rounded_low;                                   \
        (HIGH_PRODUCT) = rounded_high;                                 \
        local_absmax = fmaxf(                                          \
            local_absmax,                                              \
            fmaxf(                                                     \
                fmaxf(rounded_low, -rounded_low),                      \
                fmaxf(rounded_high, -rounded_high)));                  \
    } while (false)

#define MAIN_COMPRESS_FP8_PAIR(OFFSET, LOW_PRODUCT, HIGH_PRODUCT)         \
    do {                                                               \
        const auto low = fminf(                                        \
            fmaxf((LOW_PRODUCT) * inv_scale, -kFP8Max), kFP8Max);      \
        const auto high = fminf(                                       \
            fmaxf((HIGH_PRODUCT) * inv_scale, -kFP8Max), kFP8Max);     \
        const auto packed_fp8 =                                        \
            cvt_fp32x2_to_fp8e4m3x2(low, high);                       \
        *reinterpret_cast<uint16_t*>(                                  \
            value_ptr + element_base_idx + (OFFSET)) = packed_fp8;     \
    } while (false)

template <bool kRMSWeightBF16>
__device__ __forceinline__ void process_task(
        const uint32_t team_idx,
        const uint32_t task_warp_idx,
        const uint32_t lane_idx,
        const int64_t position,
        const int32_t req_idx,
        const int64_t kv_slot_idx,
        const float* state_cache,
        const uint32_t state_stride0,
        const uint32_t state_stride1,
        const uint32_t state_block_size,
        const int32_t* state_block_table,
        const uint32_t state_block_table_stride,
        const void* rms_weight,
        const float rms_norm_eps,
        const float* cos_sin_cache,
        const uint32_t cos_sin_stride,
        uint8_t* kv_cache,
        const uint32_t kv_cache_stride0,
        SharedStorage& shared) {
    if (task_warp_idx == 0 and lane_idx < kWindow) {
        const auto row_idx = lane_idx;
        const auto logical_position =
            position - static_cast<int64_t>(kWindow) + 1 + row_idx;
        auto block_number = int32_t(0);
        if (logical_position >= 0) {
            const auto logical_block_idx =
                logical_position /
                static_cast<int64_t>(state_block_size);
            block_number = state_block_table[
                static_cast<int64_t>(req_idx) *
                    state_block_table_stride +
                logical_block_idx];
        }
        shared.block_numbers[team_idx][row_idx] = block_number;
    }
    team_sync(team_idx);

    const auto group_idx =
        task_warp_idx * 4 + lane_idx / 8;
    const auto group_lane_idx = lane_idx % 8;
    const auto element_base_idx =
        group_idx * kQuantBlock +
        group_lane_idx * kElemsPerLane;

    float max0 = -__int_as_float(0x7f800000);
    float max1 = -__int_as_float(0x7f800000);
    float max2 = -__int_as_float(0x7f800000);
    float max3 = -__int_as_float(0x7f800000);
    float max4 = -__int_as_float(0x7f800000);
    float max5 = -__int_as_float(0x7f800000);
    float max6 = -__int_as_float(0x7f800000);
    float max7 = -__int_as_float(0x7f800000);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    float product0 = 0.0f;
    float product1 = 0.0f;
    float product2 = 0.0f;
    float product3 = 0.0f;
    float product4 = 0.0f;
    float product5 = 0.0f;
    float product6 = 0.0f;
    float product7 = 0.0f;

#pragma unroll
    for (uint32_t row_idx = 0; row_idx < kWindow; ++row_idx) {
        const auto logical_position =
            position - static_cast<int64_t>(kWindow) + 1 + row_idx;
        if (logical_position < 0)
            continue;
        const auto logical_block_idx =
            logical_position /
                static_cast<int64_t>(state_block_size);
        const auto block_offset_idx =
            logical_position -
            logical_block_idx *
                static_cast<int64_t>(state_block_size);
        const auto head_offset =
            (row_idx / kCompressRatio) * kHeadDim;
        const auto row_ptr =
            state_cache +
            static_cast<int64_t>(
                shared.block_numbers[team_idx][row_idx]) *
                state_stride0 +
            block_offset_idx * state_stride1;
        const auto value_ptr =
            row_ptr + head_offset + element_base_idx;
        const auto score_ptr =
            row_ptr + kStateWidth + head_offset + element_base_idx;

        // Split the two 128-bit chunks into short live ranges. Each element's
        // row order and FMA order remain identical to the CuTeDSL oracle.
        {
            const auto values03 =
                *reinterpret_cast<const float4*>(value_ptr);
            const auto scores03 =
                *reinterpret_cast<const float4*>(score_ptr);
            MAIN_COMPRESS_ONLINE_UPDATE(
                scores03.x, values03.x, max0, sum0, product0);
            MAIN_COMPRESS_ONLINE_UPDATE(
                scores03.y, values03.y, max1, sum1, product1);
            MAIN_COMPRESS_ONLINE_UPDATE(
                scores03.z, values03.z, max2, sum2, product2);
            MAIN_COMPRESS_ONLINE_UPDATE(
                scores03.w, values03.w, max3, sum3, product3);
        }
        {
            const auto values47 =
                *reinterpret_cast<const float4*>(value_ptr + 4);
            const auto scores47 =
                *reinterpret_cast<const float4*>(score_ptr + 4);
            MAIN_COMPRESS_ONLINE_UPDATE(
                scores47.x, values47.x, max4, sum4, product4);
            MAIN_COMPRESS_ONLINE_UPDATE(
                scores47.y, values47.y, max5, sum5, product5);
            MAIN_COMPRESS_ONLINE_UPDATE(
                scores47.z, values47.z, max6, sum6, product6);
            MAIN_COMPRESS_ONLINE_UPDATE(
                scores47.w, values47.w, max7, sum7, product7);
        }
    }

    auto local_sq_sum = 0.0f;
    product0 /= sum0;
    local_sq_sum = fmaf(product0, product0, local_sq_sum);
    product1 /= sum1;
    local_sq_sum = fmaf(product1, product1, local_sq_sum);
    product2 /= sum2;
    local_sq_sum = fmaf(product2, product2, local_sq_sum);
    product3 /= sum3;
    local_sq_sum = fmaf(product3, product3, local_sq_sum);
    product4 /= sum4;
    local_sq_sum = fmaf(product4, product4, local_sq_sum);
    product5 /= sum5;
    local_sq_sum = fmaf(product5, product5, local_sq_sum);
    product6 /= sum6;
    local_sq_sum = fmaf(product6, product6, local_sq_sum);
    product7 /= sum7;
    local_sq_sum = fmaf(product7, product7, local_sq_sum);

    local_sq_sum = warp_bfly_sum(local_sq_sum);
    if (lane_idx == 0) {
        shared.partial_sums[team_idx][task_warp_idx] =
            local_sq_sum;
    }
    team_sync(team_idx);
    if (task_warp_idx == 0 and lane_idx == 0) {
        auto total = 0.0f;
        total += shared.partial_sums[team_idx][0];
        total += shared.partial_sums[team_idx][1];
        shared.rrms[team_idx] = rsqrtf(
            total / static_cast<float>(kHeadDim) +
            rms_norm_eps);
    }
    team_sync(team_idx);
    const auto rrms = shared.rrms[team_idx];

    const auto page_idx =
        static_cast<uint64_t>(kv_slot_idx) / kKVBlockSize;
    const auto page_offset_idx =
        static_cast<uint32_t>(kv_slot_idx) % kKVBlockSize;
    auto value_ptr =
        kv_cache + page_idx * kv_cache_stride0 +
        page_offset_idx * kTokenStride;

    if (group_idx == kNopeHeadDim / kQuantBlock) {
        const auto compressed_position =
            (position / kCompressRatio) * kCompressRatio;
        const auto cos_sin =
            cos_sin_cache +
            compressed_position * cos_sin_stride;
        MAIN_COMPRESS_ROPE_PAIR(0, product0, product1);
        MAIN_COMPRESS_ROPE_PAIR(2, product2, product3);
        MAIN_COMPRESS_ROPE_PAIR(4, product4, product5);
        MAIN_COMPRESS_ROPE_PAIR(6, product6, product7);
        return;
    }

    auto local_absmax = 0.0f;
    MAIN_COMPRESS_ROUND_PAIR(0, product0, product1);
    MAIN_COMPRESS_ROUND_PAIR(2, product2, product3);
    MAIN_COMPRESS_ROUND_PAIR(4, product4, product5);
    MAIN_COMPRESS_ROUND_PAIR(6, product6, product7);

    const auto quant_mask = __activemask();
    local_absmax = fmaxf(
        local_absmax,
        __shfl_xor_sync(quant_mask, local_absmax, 4));
    local_absmax = fmaxf(
        local_absmax,
        __shfl_xor_sync(quant_mask, local_absmax, 2));
    local_absmax = fmaxf(
        local_absmax,
        __shfl_xor_sync(quant_mask, local_absmax, 1));
    const auto scale_raw =
        fmaxf(kMinScale, local_absmax / kFP8Max);
    const auto scale_bits = __float_as_uint(scale_raw);
    const auto ue8m0 =
        ((scale_bits + 0x007fffffu) >> 23) & 0xffu;
    const auto inv_scale =
        __uint_as_float((254u - ue8m0) << 23);

    MAIN_COMPRESS_FP8_PAIR(0, product0, product1);
    MAIN_COMPRESS_FP8_PAIR(2, product2, product3);
    MAIN_COMPRESS_FP8_PAIR(4, product4, product5);
    MAIN_COMPRESS_FP8_PAIR(6, product6, product7);

    if (group_lane_idx == 0) {
        auto scale_ptr =
            kv_cache + page_idx * kv_cache_stride0 +
            kKVBlockSize * kTokenStride +
            page_offset_idx * kScaleDim;
        scale_ptr[group_idx] =
            static_cast<uint8_t>(ue8m0);
        if (group_idx == 0)
            scale_ptr[kNopeHeadDim / kQuantBlock] = 0;
    }
}

#undef MAIN_COMPRESS_ONLINE_UPDATE
#undef MAIN_COMPRESS_ROPE_PAIR
#undef MAIN_COMPRESS_ROUND_PAIR
#undef MAIN_COMPRESS_FP8_PAIR

}  // namespace main_compress_detail

template <bool kRMSWeightBF16>
__global__ void __launch_bounds__(1024, 1)
main_compress_kernel(
        uint32_t num_actual,
        const float* state_cache,
        uint32_t state_stride0,
        uint32_t state_stride1,
        uint32_t state_block_size,
        const int32_t* token_to_req,
        const int64_t* positions,
        const int64_t* state_slot_mapping,
        const int32_t* state_block_table,
        uint32_t state_block_table_stride,
        const void* rms_weight,
        float rms_norm_eps,
        const float* cos_sin_cache,
        uint32_t cos_sin_stride,
        uint8_t* kv_cache,
        const int64_t* kv_slot_mapping,
        uint32_t kv_cache_stride0) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or \
        defined(__CLION_IDE__)

    using namespace main_compress_detail;

    const auto team_thread_idx =
        static_cast<uint32_t>(threadIdx.x) %
        kNumThreadsPerTeam;
    const auto team_idx =
        static_cast<uint32_t>(threadIdx.x) /
        kNumThreadsPerTeam;
    const auto task_warp_idx = team_thread_idx / kWarpSize;
    const auto lane_idx = team_thread_idx % kWarpSize;

    __shared__ SharedStorage shared;

    const auto first_position_mod =
        static_cast<uint32_t>(positions[0]) &
        (kCompressRatio - 1);
    const auto first_boundary_token =
        (kCompressRatio - 1 - first_position_mod) &
        (kCompressRatio - 1);
    auto candidate_ordinal =
        static_cast<uint64_t>(blockIdx.x) * kNumTeams +
        team_idx;
    const auto candidate_stride =
        static_cast<uint64_t>(gridDim.x) * kNumTeams;

    while (true) {
        const auto candidate_token =
            static_cast<uint64_t>(first_boundary_token) +
            candidate_ordinal * kCompressRatio;
        if (candidate_token >= num_actual)
            break;
        const auto token_idx =
            static_cast<uint32_t>(candidate_token);
        const auto position = positions[token_idx];
        const auto req_idx = token_to_req[token_idx];
        const auto kv_slot_idx = kv_slot_mapping[token_idx];
        const auto active =
            state_slot_mapping[token_idx] >= 0 and
            req_idx >= 0 and
            kv_slot_idx >= 0 and
            (position + 1) % kCompressRatio == 0;
        if (active) {
            process_task<kRMSWeightBF16>(
                team_idx, task_warp_idx, lane_idx, position,
                req_idx, kv_slot_idx, state_cache,
                state_stride0, state_stride1, state_block_size,
                state_block_table, state_block_table_stride,
                rms_weight, rms_norm_eps, cos_sin_cache,
                cos_sin_stride, kv_cache, kv_cache_stride0,
                shared);
        }
        candidate_ordinal += candidate_stride;
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        MAIN_COMPRESS_DEVICE_ASSERT(false and "This kernel only supports SM100");
#endif
}

}  // namespace flash_mla_csa

cudaError_t launch_main_compress(
        uint32_t num_actual, const float* state_cache,
        uint32_t state_stride0, uint32_t state_stride1,
        uint32_t state_block_size, const int32_t* token_to_req,
        const int64_t* positions, const int64_t* state_slot_mapping,
        const int32_t* state_block_table, uint32_t state_block_table_stride,
        const void* rms_weight, float rms_norm_eps,
        const float* cos_sin_cache, uint32_t cos_sin_stride,
        uint8_t* kv_cache, const int64_t* kv_slot_mapping,
        uint32_t kv_cache_stride0, uint32_t num_sms,
        bool rms_weight_bf16, cudaStream_t stream) {
    if (num_actual == 0 || num_sms == 0)
        return cudaSuccess;
    if (rms_weight_bf16) {
        flash_mla_csa::main_compress_kernel<true><<<num_sms, 1024, 0, stream>>>(
            num_actual, state_cache, state_stride0, state_stride1,
            state_block_size, token_to_req, positions, state_slot_mapping,
            state_block_table, state_block_table_stride, rms_weight,
            rms_norm_eps, cos_sin_cache, cos_sin_stride, kv_cache,
            kv_slot_mapping, kv_cache_stride0);
    } else {
        flash_mla_csa::main_compress_kernel<false><<<num_sms, 1024, 0, stream>>>(
            num_actual, state_cache, state_stride0, state_stride1,
            state_block_size, token_to_req, positions, state_slot_mapping,
            state_block_table, state_block_table_stride, rms_weight,
            rms_norm_eps, cos_sin_cache, cos_sin_stride, kv_cache,
            kv_slot_mapping, kv_cache_stride0);
    }
    return cudaGetLastError();
}

#undef MAIN_COMPRESS_DEVICE_ASSERT
