#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "bf16_pair_gemm.cuh"
#include "fp8_fp4_gemm_adapter.cuh"

namespace deep_gemm {

constexpr uint32_t kCSAPTokens = 8192;
constexpr uint32_t kCSAPHidden = 7168;
constexpr uint32_t kCSAPQRank = 1536;
constexpr uint32_t kCSAPKVRank = 512;
constexpr uint32_t kCSAPQKVWidth = kCSAPQRank + kCSAPKVRank;
constexpr uint32_t kCSAPQuantPack = 512;
constexpr uint32_t kCSAPQuantGroup = 128;
constexpr uint32_t kCSAPThreadsPerQuantTask = 32;
constexpr uint32_t kCSAPQuantTasksPerCTA = 16;

CUTLASS_DEVICE uint32_t csa_p_warp_max(uint32_t value) {
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2)
        value = cute::max(value, __shfl_down_sync(0xffffffffu, value, offset));
    return __shfl_sync(0xffffffffu, value, 0);
}

CUTLASS_DEVICE float csa_p_warp_sum(float value) {
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// Triton computes exp2(ceil(log2(max(amax, 1e-10) / 448))).  For a
// nonzero BF16 amax the exponent decision is exact in the BF16 bit pattern:
// 448 = 1.75 * 2^8 and 1.75 has BF16 mantissa 96.
CUTLASS_DEVICE uint8_t csa_p_e4m3_scale_byte(uint32_t amax_bf16) {
    if (amax_bf16 == 0)
        return 85u;
    constexpr uint32_t kExponentShift = 7;
    constexpr uint32_t kMantissaMask = 0x7fu;
    constexpr uint32_t kFinfoExponent = 8;
    constexpr uint32_t kFinfoMantissa = 96;
    const auto exponent = amax_bf16 >> kExponentShift;
    const auto mantissa = amax_bf16 & kMantissaMask;
    auto proposed = static_cast<int32_t>(exponent) -
                    static_cast<int32_t>(kFinfoExponent) +
                    static_cast<int32_t>(mantissa > kFinfoMantissa);
    proposed = cute::max(0, cute::min(255, proposed));
    return static_cast<uint8_t>(proposed);
}

CUTLASS_DEVICE float csa_p_inverse_scale(uint8_t scale_byte) {
    return __uint_as_float(
        (254u - static_cast<uint32_t>(scale_byte)) << 23);
}

CUTLASS_DEVICE __nv_bfloat162 csa_p_inverse_scale_bf16x2(
        uint8_t scale_byte) {
    const auto scale_bf16 =
        (254u - static_cast<uint32_t>(scale_byte)) << 7;
    const auto packed = scale_bf16 | (scale_bf16 << 16);
    return *reinterpret_cast<const __nv_bfloat162*>(&packed);
}

template <uint32_t kWidth, uint32_t kNumCTAs, uint32_t kILPPacks>
CUTLASS_DEVICE void csa_p_quantize_packed(
        const __nv_bfloat16* input,
        uint8_t* output,
        uint32_t* scales,
        uint32_t scale_stride_k,
        uint32_t cta_idx) {
    static_assert(kWidth % kCSAPQuantPack == 0);
    constexpr uint32_t kPacksPerRow = kWidth / kCSAPQuantPack;
    constexpr uint32_t kTotalTasks = kCSAPTokens * kPacksPerRow;
    constexpr uint32_t kTaskStride = kNumCTAs * kCSAPQuantTasksPerCTA;

    const auto task_slot = static_cast<uint32_t>(threadIdx.x) / 32;
    const auto lane_idx = static_cast<uint32_t>(threadIdx.x) % 32;

    static_assert(kILPPacks >= 1 and kILPPacks <= 4);
    for (uint32_t task_base = cta_idx * kCSAPQuantTasksPerCTA;
         task_base < kTotalTasks;
         task_base += kTaskStride * kILPPacks) {
        bool valid[kILPPacks];
        uint32_t row_idx[kILPPacks];
        uint32_t pack_idx[kILPPacks];
        uint64_t pack_base[kILPPacks];
        uint2 raw_bf16x4[kILPPacks][4];

#pragma unroll
        for (uint32_t ilp_idx = 0; ilp_idx < kILPPacks; ++ilp_idx) {
            const auto task_idx =
                task_base + task_slot + ilp_idx * kTaskStride;
            valid[ilp_idx] = task_idx < kTotalTasks;
            row_idx[ilp_idx] =
                valid[ilp_idx] ? task_idx / kPacksPerRow : 0u;
            pack_idx[ilp_idx] =
                valid[ilp_idx] ? task_idx % kPacksPerRow : 0u;
            pack_base[ilp_idx] =
                static_cast<uint64_t>(row_idx[ilp_idx]) * kWidth +
                pack_idx[ilp_idx] * kCSAPQuantPack;
#pragma unroll
            for (uint32_t group_idx = 0; group_idx < 4; ++group_idx) {
                const auto element_idx =
                    group_idx * kCSAPQuantGroup + lane_idx * 4;
                raw_bf16x4[ilp_idx][group_idx] = {0u, 0u};
                if (valid[ilp_idx])
                    raw_bf16x4[ilp_idx][group_idx] =
                        *reinterpret_cast<const uint2*>(
                            input + pack_base[ilp_idx] + element_idx);
            }
        }

#pragma unroll
        for (uint32_t ilp_idx = 0; ilp_idx < kILPPacks; ++ilp_idx) {
            uint32_t packed_scale = 0;
#pragma unroll
            for (uint32_t group_idx = 0; group_idx < 4; ++group_idx) {
                const auto element_idx =
                    group_idx * kCSAPQuantGroup + lane_idx * 4;
                constexpr uint32_t kPackedBF16AbsMask = 0x7fff7fffu;
                const auto packed_amax = __vmaxu2(
                    raw_bf16x4[ilp_idx][group_idx].x &
                        kPackedBF16AbsMask,
                    raw_bf16x4[ilp_idx][group_idx].y &
                        kPackedBF16AbsMask);
                const auto local_amax = cute::max(
                    packed_amax & 0xffffu, packed_amax >> 16);
                const auto scale_byte = csa_p_e4m3_scale_byte(
                    csa_p_warp_max(local_amax));
                packed_scale |= static_cast<uint32_t>(scale_byte)
                                << (group_idx * 8);

                if (valid[ilp_idx]) {
                    const auto* values_bf16x2 =
                        reinterpret_cast<const __nv_bfloat162*>(
                            &raw_bf16x4[ilp_idx][group_idx]);
                    const auto inv_scale =
                        csa_p_inverse_scale_bf16x2(scale_byte);
                    const auto fp8_values = __nv_fp8x4_e4m3(
                        __hmul2(values_bf16x2[0], inv_scale),
                        __hmul2(values_bf16x2[1], inv_scale));
                    *reinterpret_cast<uint32_t*>(
                        output + pack_base[ilp_idx] + element_idx) =
                        *reinterpret_cast<const uint32_t*>(&fp8_values);
                }
            }
            if (valid[ilp_idx] and lane_idx == 0)
                scales[row_idx[ilp_idx] +
                       pack_idx[ilp_idx] * scale_stride_k] = packed_scale;
        }
    }
}

template <uint32_t kNumCTAs>
CUTLASS_DEVICE void csa_p_fused_rmsnorm(
        const __nv_bfloat16* qkv,
        const __nv_bfloat16* q_weight,
        const __nv_bfloat16* kv_weight,
        __nv_bfloat16* q_output,
        __nv_bfloat16* kv_output,
        uint8_t* q_fp8,
        uint32_t* q_scales,
        uint32_t q_scale_stride_k,
        bool quantize_q,
        uint32_t cta_idx) {
    constexpr uint32_t kRowsPerCTA = 16;
    constexpr uint32_t kRowStride = kNumCTAs * kRowsPerCTA;
    const auto row_slot = static_cast<uint32_t>(threadIdx.x) / 32;
    const auto lane_idx = static_cast<uint32_t>(threadIdx.x) % 32;

    for (uint32_t row_base = cta_idx * kRowsPerCTA;
         row_base < kCSAPTokens; row_base += kRowStride) {
        const auto row_idx = row_base + row_slot;
        const auto valid = row_idx < kCSAPTokens;
        const auto* qkv_row = qkv + static_cast<uint64_t>(row_idx) *
                                      kCSAPQKVWidth;

        // One physical warp emulates Triton's four virtual warps.  Each
        // virtual thread owns eight contiguous values in [0,1024), followed
        // by eight in [1024,2048); columns >=1536 contribute zero.
        float q_partial[4];
        float kv_partial[4];
#pragma unroll
        for (uint32_t virtual_warp = 0; virtual_warp < 4; ++virtual_warp) {
            const auto virtual_tid = virtual_warp * 32 + lane_idx;
            uint4 first_q = {0u, 0u, 0u, 0u};
            uint4 second_q = {0u, 0u, 0u, 0u};
            uint2 kv = {0u, 0u};
            if (valid) {
                first_q = *reinterpret_cast<const uint4*>(
                    qkv_row + virtual_tid * 8);
                if (virtual_tid < 64)
                    second_q = *reinterpret_cast<const uint4*>(
                        qkv_row + 1024 + virtual_tid * 8);
                kv = *reinterpret_cast<const uint2*>(
                    qkv_row + kCSAPQRank + virtual_tid * 4);
            }

            const auto* first_values =
                reinterpret_cast<const __nv_bfloat16*>(&first_q);
            const auto* second_values =
                reinterpret_cast<const __nv_bfloat16*>(&second_q);
            const auto* kv_values =
                reinterpret_cast<const __nv_bfloat16*>(&kv);
            auto q_sum =
                __bfloat162float(first_values[0]) *
                    __bfloat162float(first_values[0]) +
                __bfloat162float(first_values[1]) *
                    __bfloat162float(first_values[1]);
#pragma unroll
            for (uint32_t i = 2; i < 8; ++i) {
                const auto value = __bfloat162float(first_values[i]);
                q_sum = value * value + q_sum;
            }
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                const auto value = __bfloat162float(second_values[i]);
                q_sum = value * value + q_sum;
            }
            auto kv_sum =
                __bfloat162float(kv_values[0]) *
                    __bfloat162float(kv_values[0]) +
                __bfloat162float(kv_values[1]) *
                    __bfloat162float(kv_values[1]);
            const auto kv_value2 = __bfloat162float(kv_values[2]);
            kv_sum = kv_value2 * kv_value2 + kv_sum;
            const auto kv_value3 = __bfloat162float(kv_values[3]);
            kv_sum = kv_value3 * kv_value3 + kv_sum;
            q_partial[virtual_warp] = csa_p_warp_sum(q_sum);
            kv_partial[virtual_warp] = csa_p_warp_sum(kv_sum);
        }

        const auto q_total =
            (q_partial[0] + q_partial[2]) +
            (q_partial[1] + q_partial[3]);
        const auto kv_total =
            (kv_partial[0] + kv_partial[2]) +
            (kv_partial[1] + kv_partial[3]);
        const auto q_inv =
            rsqrtf(q_total / static_cast<float>(kCSAPQRank) + 1.0e-6f);
        const auto kv_inv =
            rsqrtf(kv_total / static_cast<float>(kCSAPKVRank) + 1.0e-6f);

        if (valid) {
            auto* q_row = q_output + static_cast<uint64_t>(row_idx) *
                                      kCSAPQRank;
            auto* kv_row = kv_output + static_cast<uint64_t>(row_idx) *
                                        kCSAPKVRank;
            auto* q_fp8_row = q_fp8 + static_cast<uint64_t>(row_idx) *
                                      kCSAPQRank;
#pragma unroll
            for (uint32_t pack_idx = 0; pack_idx < 3; ++pack_idx) {
                uint32_t packed_scale = 0;
#pragma unroll
                for (uint32_t group_idx = 0; group_idx < 4; ++group_idx) {
                    const auto col = pack_idx * kCSAPQuantPack +
                                     group_idx * kCSAPQuantGroup +
                                     lane_idx * 4;
                    const auto q_values = *reinterpret_cast<const uint2*>(
                        qkv_row + col);
                    const auto q_weights = *reinterpret_cast<const uint2*>(
                        q_weight + col);
                    const auto* q_values_bf16 =
                        reinterpret_cast<const __nv_bfloat16*>(&q_values);
                    const auto* q_weights_bf16 =
                        reinterpret_cast<const __nv_bfloat16*>(&q_weights);
                    uint2 q_results;
                    auto* q_results_bf16 =
                        reinterpret_cast<__nv_bfloat16*>(&q_results);
#pragma unroll
                    for (uint32_t i = 0; i < 4; ++i) {
                        const auto value =
                            __bfloat162float(q_values_bf16[i]) * q_inv *
                            __bfloat162float(q_weights_bf16[i]);
                        q_results_bf16[i] = __float2bfloat16_rn(value);
                    }
                    *reinterpret_cast<uint2*>(q_row + col) = q_results;

                    if (quantize_q) {
                        constexpr uint32_t kPackedBF16AbsMask = 0x7fff7fffu;
                        const auto packed_amax = __vmaxu2(
                            q_results.x & kPackedBF16AbsMask,
                            q_results.y & kPackedBF16AbsMask);
                        const auto local_amax = cute::max(
                            packed_amax & 0xffffu, packed_amax >> 16);
                        const auto scale_byte = csa_p_e4m3_scale_byte(
                            csa_p_warp_max(local_amax));
                        packed_scale |= static_cast<uint32_t>(scale_byte)
                                        << (group_idx * 8);
                        const auto* q_results_bf16x2 =
                            reinterpret_cast<const __nv_bfloat162*>(&q_results);
                        const auto inv_scale =
                            csa_p_inverse_scale_bf16x2(scale_byte);
                        const auto fp8_values = __nv_fp8x4_e4m3(
                            __hmul2(q_results_bf16x2[0], inv_scale),
                            __hmul2(q_results_bf16x2[1], inv_scale));
                        *reinterpret_cast<uint32_t*>(q_fp8_row + col) =
                            *reinterpret_cast<const uint32_t*>(&fp8_values);
                    }
                }
                if (quantize_q and lane_idx == 0)
                    q_scales[row_idx + pack_idx * q_scale_stride_k] =
                        packed_scale;
            }
#pragma unroll
            for (uint32_t group_idx = 0; group_idx < 4; ++group_idx) {
                const auto col =
                    group_idx * kCSAPQuantGroup + lane_idx * 4;
                const auto kv_values = *reinterpret_cast<const uint2*>(
                    qkv_row + kCSAPQRank + col);
                const auto kv_weights = *reinterpret_cast<const uint2*>(
                    kv_weight + col);
                const auto* kv_values_bf16 =
                    reinterpret_cast<const __nv_bfloat16*>(&kv_values);
                const auto* kv_weights_bf16 =
                    reinterpret_cast<const __nv_bfloat16*>(&kv_weights);
                uint2 kv_results;
                auto* kv_results_bf16 =
                    reinterpret_cast<__nv_bfloat16*>(&kv_results);
#pragma unroll
                for (uint32_t i = 0; i < 4; ++i) {
                    const auto value =
                        __bfloat162float(kv_values_bf16[i]) * kv_inv *
                        __bfloat162float(kv_weights_bf16[i]);
                    kv_results_bf16[i] = __float2bfloat16_rn(value);
                }
                *reinterpret_cast<uint2*>(kv_row + col) = kv_results;
            }
        }
    }
}

template <
    uint32_t kNumSMs, uint32_t kPQuantILP,
    uint32_t kPBlockM, uint32_t kPBlockN, uint32_t kPBlockK,
    uint32_t kPSwizzleA, uint32_t kPSwizzleB, uint32_t kPSwizzleCD,
    uint32_t kPNumStages,
    uint32_t kPNumNonEpilogueThreads, uint32_t kPNumEpilogueThreads,
    uint32_t kPNumMulticast, bool kPIsMulticastOnA,
    bool kPSwapAB>
CUTLASS_DEVICE void projection_fusion_body(
        const cute::TmaDescriptor& p_tensor_map_a,
        const cute::TmaDescriptor& p_tensor_map_b,
        const cute::TmaDescriptor& p_tensor_map_sfa,
        const cute::TmaDescriptor& p_tensor_map_sfb,
        const cute::TmaDescriptor& p_tensor_map_d,
        const __nv_bfloat16* hidden,
        uint8_t* hidden_fp8,
        uint32_t* hidden_scale,
        uint32_t hidden_scale_stride_k,
        const __nv_bfloat16* q_norm_weight,
        const __nv_bfloat16* kv_norm_weight,
        const __nv_bfloat16* qkv,
        __nv_bfloat16* q_norm,
        __nv_bfloat16* kv_norm,
        uint8_t* q_norm_fp8,
        uint32_t* q_norm_scale,
        uint32_t q_norm_scale_stride_k,
        uint32_t* stage_barrier,
        uint32_t profile_mode,
        uint32_t cta_idx) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    static_assert(kNumSMs % 2 == 0);
    static_assert(kPNumMulticast == 2);
    DG_DEVICE_ASSERT(cta_idx < kNumSMs);

    csa_p_quantize_packed<kCSAPHidden, kNumSMs, kPQuantILP>(
        hidden, hidden_fp8, hidden_scale, hidden_scale_stride_k, cta_idx);
    if (profile_mode < 2)
        return;
    csa_root_lane_barrier<kNumSMs>(stage_barrier, cta_idx);

    sm100_fp8_fp4_gemm_1d1d_body<
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        128, 128, 128,
        0, kCSAPQKVWidth, kCSAPHidden,
        kPBlockM, kPBlockN, kPBlockK, 1,
        kPSwizzleA, kPSwizzleB, kPSwizzleCD,
        kPNumStages, kPNumNonEpilogueThreads, kPNumEpilogueThreads,
        kPNumMulticast, kPIsMulticastOnA, kNumSMs,
        kPSwapAB, true, GemmType::Normal, false,
        cutlass::float_e4m3_t, cutlass::float_e4m3_t,
        cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity,
        true>(
            nullptr, kCSAPTokens, kCSAPQKVWidth, kCSAPHidden,
            p_tensor_map_a, p_tensor_map_b,
            p_tensor_map_sfa, p_tensor_map_sfb, p_tensor_map_d,
            cta_idx);
    if (profile_mode < 3)
        return;
    csa_root_lane_barrier<kNumSMs>(stage_barrier, cta_idx);

    csa_p_fused_rmsnorm<kNumSMs>(
        qkv, q_norm_weight, kv_norm_weight,
        q_norm, kv_norm, q_norm_fp8, q_norm_scale,
        q_norm_scale_stride_k, profile_mode >= 4, cta_idx);
#else
    if (cta_idx == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

template <
    uint32_t kNumSMs, uint32_t kPQuantILP,
    uint32_t kPBlockM, uint32_t kPBlockN, uint32_t kPBlockK,
    uint32_t kPSwizzleA, uint32_t kPSwizzleB, uint32_t kPSwizzleCD,
    uint32_t kPNumStages,
    uint32_t kPNumNonEpilogueThreads, uint32_t kPNumEpilogueThreads,
    uint32_t kPNumMulticast, bool kPIsMulticastOnA,
    bool kPSwapAB>
CUTLASS_GLOBAL void __launch_bounds__(512, 1)
projection_fusion_kernel(
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_b,
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_sfa,
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_sfb,
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_d,
        const __nv_bfloat16* hidden,
        uint8_t* hidden_fp8,
        uint32_t* hidden_scale,
        uint32_t hidden_scale_stride_k,
        const __nv_bfloat16* q_norm_weight,
        const __nv_bfloat16* kv_norm_weight,
        const __nv_bfloat16* qkv,
        __nv_bfloat16* q_norm,
        __nv_bfloat16* kv_norm,
        uint8_t* q_norm_fp8,
        uint32_t* q_norm_scale,
        uint32_t q_norm_scale_stride_k,
        uint32_t* stage_barrier,
        uint32_t profile_mode) {
    projection_fusion_body<
        kNumSMs, kPQuantILP,
        kPBlockM, kPBlockN, kPBlockK,
        kPSwizzleA, kPSwizzleB, kPSwizzleCD, kPNumStages,
        kPNumNonEpilogueThreads, kPNumEpilogueThreads,
        kPNumMulticast, kPIsMulticastOnA, kPSwapAB>(
            p_tensor_map_a, p_tensor_map_b,
            p_tensor_map_sfa, p_tensor_map_sfb, p_tensor_map_d,
            hidden, hidden_fp8, hidden_scale, hidden_scale_stride_k,
            q_norm_weight, kv_norm_weight, qkv, q_norm, kv_norm,
            q_norm_fp8, q_norm_scale, q_norm_scale_stride_k,
            stage_barrier, profile_mode,
            static_cast<uint32_t>(blockIdx.x));
}

}  // namespace deep_gemm
