#pragma once

#include <cuda_bf16.h>

#include "bf16_pair_gemm.cuh"

namespace deep_gemm {

CUTLASS_DEVICE uint8_t csa_cvt_fp32x2_to_e2m1x2(
        const float low, const float high) {
    uint32_t packed;
    asm volatile(
        "{\n\t"
        ".reg .b8 tmp;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 tmp, %2, %1;\n\t"
        "cvt.u32.u8 %0, tmp;\n\t"
        "}"
        : "=r"(packed)
        : "f"(low), "f"(high));
    return static_cast<uint8_t>(packed);
}

CUTLASS_DEVICE float csa_bf16_roundtrip(const float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

CUTLASS_DEVICE float csa_triton_exp(const float value) {
    const auto scaled = value * __int_as_float(0x3fb8aa3b);
    float result;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(scaled));
    return result;
}

CUTLASS_DEVICE uint64_t csa_global_timer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

CUTLASS_DEVICE float csa_warp_sum(float value) {
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return __shfl_sync(0xffffffffu, value, 0);
}

template <bool kDenseCurrentC4>
CUTLASS_DEVICE float4 csa_compress_c4_head(
        const float* current_kv_score,
        const float* state_cache,
        const uint32_t state_stride0,
        const uint32_t state_stride1,
        const int32_t req_idx,
        const int32_t* state_block_table,
        const uint32_t state_block_table_stride,
        const int64_t start,
        const int64_t first_position,
        const float* ape,
        const uint32_t lane_idx) {
    constexpr uint32_t kStateWidth = 256;
    constexpr uint32_t kHeadSize = 128;
    constexpr uint32_t kWindow = 8;
    constexpr uint32_t kStateBlockSize = 4;
    constexpr uint32_t kTokens = 8192;
    constexpr uint32_t kDenseStride = 2 * kStateWidth;

    float scores[4][kWindow];
    float values[4][kWindow];
    float max_scores[4];
#pragma unroll
    for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter)
        max_scores[dim_iter] = -__int_as_float(0x7f800000);

#pragma unroll
    for (uint32_t row_idx = 0; row_idx < kWindow; ++row_idx) {
        const auto logical_position = start + row_idx;
        const float* state_row = nullptr;
        const float* dense_row = nullptr;
        const auto head_offset =
            row_idx >= kStateBlockSize ? kHeadSize : 0;
        if constexpr (kDenseCurrentC4) {
            if (logical_position >= first_position and
                logical_position < first_position + kTokens) {
                const auto token_idx = logical_position - first_position;
                dense_row = current_kv_score +
                    token_idx * kDenseStride + head_offset;
            }
        }
        if (dense_row == nullptr and logical_position >= 0) {
            const auto logical_block =
                logical_position / kStateBlockSize;
            const auto block_number = state_block_table[
                static_cast<int64_t>(req_idx) * state_block_table_stride +
                logical_block];
            if (block_number >= 0) {
                const auto block_offset =
                    logical_position % kStateBlockSize;
                state_row =
                    state_cache +
                    static_cast<int64_t>(block_number) * state_stride0 +
                    block_offset * state_stride1 + head_offset;
            }
        }

#pragma unroll
        for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter) {
            const auto dim = lane_idx + dim_iter * 32;
            auto score = -__int_as_float(0x7f800000);
            auto value = 0.0f;
            if (dense_row != nullptr) {
                const auto ape_row = logical_position % kStateBlockSize;
                score = dense_row[kStateWidth + dim] +
                    ape[ape_row * kStateWidth + head_offset + dim];
                value = dense_row[dim];
            } else if (state_row != nullptr) {
                score = state_row[kStateWidth + dim];
                value = state_row[dim];
            }
            scores[dim_iter][row_idx] = score;
            values[dim_iter][row_idx] = value;
            max_scores[dim_iter] =
                fmaxf(max_scores[dim_iter], score);
        }
    }

    float weights[4][kWindow];
    float exp_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (uint32_t row_idx = 0; row_idx < kWindow; ++row_idx) {
#pragma unroll
        for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter) {
            weights[dim_iter][row_idx] = csa_triton_exp(
                scores[dim_iter][row_idx] - max_scores[dim_iter]);
            exp_sums[dim_iter] += weights[dim_iter][row_idx];
        }
    }

    float compressed[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (uint32_t row_idx = 0; row_idx < kWindow; ++row_idx) {
#pragma unroll
        for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter)
            compressed[dim_iter] = fmaf(
                weights[dim_iter][row_idx] / exp_sums[dim_iter],
                values[dim_iter][row_idx], compressed[dim_iter]);
    }
    return make_float4(
        compressed[0], compressed[1], compressed[2], compressed[3]);
}

template <uint32_t kRightSMs, uint32_t kNumWarps, bool kDenseCurrentC4>
CUTLASS_DEVICE void csa_index_c4_body(
        const uint32_t cta_idx,
        const uint32_t consumer_thread_idx,
        const float* current_kv_score,
        const float* state_cache,
        const uint32_t state_stride0,
        const uint32_t state_stride1,
        const int32_t* token_to_req,
        const int64_t* positions,
        const int64_t* state_slot_mapping,
        const int32_t* state_block_table,
        const uint32_t state_block_table_stride,
        const float* rms_weight,
        const float* cos_sin_cache,
        const uint32_t cos_sin_stride,
        uint8_t* kv_cache,
        const int64_t* kv_slot_mapping,
        const uint32_t kv_cache_stride0,
        const float* ape) {
    constexpr uint32_t kTokens = 8192;
    constexpr uint32_t kStateWidth = 256;
    constexpr uint32_t kHeadSize = 128;
    constexpr uint32_t kCompressRatio = 4;
    // vLLM passes overlap=True, i.e. one preceding compressed group.  The
    // C4 window is therefore (1 + 1) * 4 = 8 rows, not 36 rows.
    constexpr uint32_t kOverlap = 1;
    constexpr uint32_t kWindow = (1 + kOverlap) * kCompressRatio;
    constexpr uint32_t kStateBlockSize = 4;
    constexpr uint32_t kKVBlockSize = 64;
    constexpr uint32_t kTokenStride = 64;
    constexpr uint32_t kScaleDim = 4;
    constexpr uint32_t kRopeHeadDim = 64;
    constexpr uint32_t kWarpSize = 32;
    constexpr uint32_t kTasksPerWave = kRightSMs * kNumWarps;
    DG_STATIC_ASSERT(
        kNumWarps > 0 and kNumWarps <= 16,
        "Invalid number of C4 compute warps");

    const auto warp_idx = consumer_thread_idx / kWarpSize;
    const auto lane_idx = consumer_thread_idx % kWarpSize;

    // Shape-specific S=1 route: schedule only the 1/4 C4 boundary tokens.
    // Each compute warp drains every task assigned to it with direct LDG;
    // there are no TMA stages or per-row acquire barriers in this body.
    const auto first_position = positions[0];
    const auto first_boundary_offset = static_cast<uint32_t>(
        (3 - (first_position & 3) + 4) & 3);
    const auto num_boundary_tasks = kTokens > first_boundary_offset ?
        (kTokens - first_boundary_offset + 3) / 4 : 0;
    for (uint32_t boundary_task_idx =
             cta_idx * kNumWarps + warp_idx;
         boundary_task_idx < num_boundary_tasks;
         boundary_task_idx += kTasksPerWave) {
        const auto token_idx = first_boundary_offset +
            boundary_task_idx * kCompressRatio;
        const auto slot_id = state_slot_mapping[token_idx];
        if (slot_id < 0)
            continue;
        const auto position = positions[token_idx];
        const auto req_idx = token_to_req[token_idx];
        if (req_idx < 0)
            continue;
        const auto start = position - static_cast<int64_t>(kWindow) + 1;
        const auto compressed_head =
            csa_compress_c4_head<kDenseCurrentC4>(
                current_kv_score, state_cache,
                state_stride0, state_stride1, req_idx,
                state_block_table, state_block_table_stride,
                start, first_position, ape, lane_idx);
        const float compressed[4] = {
            compressed_head.x, compressed_head.y,
            compressed_head.z, compressed_head.w};

        float local_sq_sum = 0.0f;
#pragma unroll
        for (uint32_t dim_iter = 0; dim_iter < 4; ++ dim_iter)
            local_sq_sum =
                fmaf(compressed[dim_iter], compressed[dim_iter], local_sq_sum);

        const auto variance_sum = csa_warp_sum(local_sq_sum);
        const auto rrms = rsqrtf(
            variance_sum / static_cast<float>(kHeadSize) + 1.0e-6f);
        float normed[4];
#pragma unroll
        for (uint32_t dim_iter = 0; dim_iter < 4; ++ dim_iter) {
            const auto dim = lane_idx + dim_iter * kWarpSize;
            normed[dim_iter] =
                compressed[dim_iter] * rrms * rms_weight[dim];
        }

        const auto kv_slot_idx = kv_slot_mapping[token_idx];
        if (kv_slot_idx < 0)
            continue;
        auto* cache_block =
            kv_cache + (kv_slot_idx / kKVBlockSize) * kv_cache_stride0;
        auto* value_ptr =
            cache_block + (kv_slot_idx % kKVBlockSize) * kTokenStride;
        auto* scale_ptr = cache_block + kKVBlockSize * kTokenStride +
                          (kv_slot_idx % kKVBlockSize) * kScaleDim;
        const auto compressed_position =
            (position / kCompressRatio) * kCompressRatio;

        // There are 64 adjacent pairs.  A lane handles two pairs; the second
        // half is the RoPE half.  Round through BF16 before MXFP4 to match the
        // production Triton contract.
        float pair_low[2];
        float pair_high[2];
#pragma unroll
        for (uint32_t pair_iter = 0; pair_iter < 2; ++ pair_iter) {
            const auto pair_idx = lane_idx + pair_iter * kWarpSize;
            const auto source_lane_idx = (lane_idx * 2) % kWarpSize;
            const auto source_low_iter = pair_iter * 2;
            const auto even_low = __shfl_sync(
                0xffffffffu, normed[source_low_iter],
                source_lane_idx);
            const auto even_high = __shfl_sync(
                0xffffffffu, normed[source_low_iter + 1],
                source_lane_idx);
            const auto odd_low = __shfl_sync(
                0xffffffffu, normed[source_low_iter],
                source_lane_idx + 1);
            const auto odd_high = __shfl_sync(
                0xffffffffu, normed[source_low_iter + 1],
                source_lane_idx + 1);
            const auto even =
                lane_idx < 16 ? even_low : even_high;
            const auto odd =
                lane_idx < 16 ? odd_low : odd_high;
            float rotated_even = even;
            float rotated_odd = odd;
            if (pair_idx >= (kHeadSize - kRopeHeadDim) / 2) {
                const auto rope_pair =
                    pair_idx - (kHeadSize - kRopeHeadDim) / 2;
                const auto* cos_sin =
                    cos_sin_cache + compressed_position * cos_sin_stride;
                const auto cos_value = cos_sin[rope_pair];
                const auto sin_value = cos_sin[kRopeHeadDim / 2 + rope_pair];
                rotated_even = even * cos_value - odd * sin_value;
                rotated_odd = odd * cos_value + even * sin_value;
            }
            pair_low[pair_iter] = csa_bf16_roundtrip(rotated_even);
            pair_high[pair_iter] = csa_bf16_roundtrip(rotated_odd);
        }

        // Each 32-element quant block is sixteen adjacent pairs.  The lower
        // half-warp packs one pair for each of the four blocks.
#pragma unroll
        for (uint32_t block = 0; block < 4; ++ block) {
            const auto pair_idx = block * 16 + (lane_idx % 16);
            const auto source_lane = pair_idx % kWarpSize;
            const auto source_iter = pair_idx / kWarpSize;
            const auto low = __shfl_sync(
                0xffffffffu, pair_low[source_iter], source_lane);
            const auto high = __shfl_sync(
                0xffffffffu, pair_high[source_iter], source_lane);
            if (lane_idx < 16) {
                float absmax = fmaxf(fabsf(low), fabsf(high));
#pragma unroll
                for (uint32_t offset = 8; offset > 0; offset /= 2)
                    absmax = fmaxf(
                        absmax,
                        __shfl_xor_sync(0x0000ffffu, absmax, offset));
                absmax = fmaxf(absmax, 6.0f * 0x1p-126f);
                auto exponent = ceilf(log2f(absmax * (1.0f / 6.0f)));
                exponent = fminf(fmaxf(exponent, -127.0f), 127.0f);
                const auto inv_scale = exp2f(-exponent);
                value_ptr[block * 16 + lane_idx] =
                    csa_cvt_fp32x2_to_e2m1x2(
                        low * inv_scale, high * inv_scale);
                if (lane_idx == 0)
                    scale_ptr[block] = static_cast<uint8_t>(exponent + 127.0f);
            }
        }
    }
}

CUTLASS_DEVICE void csa_store_live_tail_state(
        const float* current_kv_score,
        float* state_cache,
        const uint32_t state_stride0,
        const uint32_t state_stride1,
        const uint32_t num_state_slots,
        const int64_t* positions,
        const int64_t* state_slot_mapping,
        const float* ape,
        const uint32_t lane_idx) {
    constexpr uint32_t kTokens = 8192;
    constexpr uint32_t kStateWidth = 256;
    constexpr uint32_t kStateRowWidth = 2 * kStateWidth;
    constexpr uint32_t kStateBlockSize = 4;
    constexpr uint32_t kLiveTailRows = 4;

#pragma unroll
    for (uint32_t tail_idx = 0; tail_idx < kLiveTailRows; ++tail_idx) {
        const auto token_idx = kTokens - kLiveTailRows + tail_idx;
        const auto slot_idx = state_slot_mapping[token_idx];
        if (slot_idx < 0 or
            static_cast<uint64_t>(slot_idx) >= num_state_slots)
            continue;
        auto* state_row = state_cache +
            (static_cast<uint64_t>(slot_idx) / kStateBlockSize) *
                state_stride0 +
            (static_cast<uint32_t>(slot_idx) % kStateBlockSize) *
                state_stride1;
        const auto* dense_row =
            current_kv_score + token_idx * kStateRowWidth;
        const auto ape_row =
            static_cast<uint32_t>(positions[token_idx]) % kStateBlockSize;
#pragma unroll
        for (uint32_t element_idx = lane_idx;
             element_idx < kStateRowWidth;
             element_idx += 32) {
            auto value = dense_row[element_idx];
            if (element_idx >= kStateWidth)
                value += ape[
                    ape_row * kStateWidth + element_idx - kStateWidth];
            state_row[element_idx] = value;
        }
    }
}

template <
    uint32_t kLeftSMs, uint32_t kRightSMs, uint32_t kC4WarpConfig,
    bool kDenseCurrentC4, bool kStoreDenseC0,
    uint32_t kC0BlockM, uint32_t kC0BlockN, uint32_t kC0BlockK,
    uint32_t kC0SwizzleA, uint32_t kC0SwizzleB, uint32_t kC0SwizzleCD,
    uint32_t kC0NumStages,
    uint32_t kC0NumNonEpilogueThreads, uint32_t kC0NumEpilogueThreads,
    uint32_t kC0NumMulticast, bool kC0IsMulticastOnA,
    uint32_t kC0KAlignment, bool kC0SwapAB,
    uint32_t kIW0BlockM, uint32_t kIW0BlockN, uint32_t kIW0BlockK,
    uint32_t kIW0SwizzleA, uint32_t kIW0SwizzleB, uint32_t kIW0SwizzleCD,
    uint32_t kIW0NumStages,
    uint32_t kIW0NumNonEpilogueThreads, uint32_t kIW0NumEpilogueThreads,
    uint32_t kIW0NumMulticast, bool kIW0IsMulticastOnA,
    uint32_t kIW0KAlignment, bool kIW0SwapAB>
CUTLASS_DEVICE void index_compress_fusion_body(
        const cute::TmaDescriptor& c0_tensor_map_a,
        const cute::TmaDescriptor& c0_tensor_map_b,
        const cute::TmaDescriptor& c0_tensor_map_d,
        const cute::TmaDescriptor& iw0_tensor_map_a,
        const cute::TmaDescriptor& iw0_tensor_map_b,
        const cute::TmaDescriptor& iw0_tensor_map_d,
        uint32_t* lane_barrier,
        uint64_t* row_ready,
        uint32_t num_state_slots,
        float* c4_workspace,
        uint32_t profile_mode,
        float* current_kv_score,
        float* state_cache,
        uint32_t state_stride0,
        uint32_t state_stride1,
        const int32_t* token_to_req,
        const int64_t* positions,
        const int64_t* state_slot_mapping,
        const int32_t* state_block_table,
        uint32_t state_block_table_stride,
        const float* rms_weight,
        const float* cos_sin_cache,
        uint32_t cos_sin_stride,
        uint8_t* kv_cache,
        const int64_t* kv_slot_mapping,
        uint32_t kv_cache_stride0,
        const float* ape,
        uint32_t cta_idx) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    constexpr uint32_t kClusterSize = 2;
    static_assert(kLeftSMs % kClusterSize == 0);
    static_assert(kRightSMs % kClusterSize == 0);
    static_assert(kC4WarpConfig <= 3);
    static_assert(
        not kDenseCurrentC4 or kStoreDenseC0,
        "Dense-current C4 requires the dense C0 output");
    DG_DEVICE_ASSERT(cta_idx < kRightSMs);

    epilogue::CSAPrefillC0C4Params csa_params = {
        .token_to_req = token_to_req,
        .positions = positions,
        .state_slot_mapping = state_slot_mapping,
        .state_block_table = state_block_table,
        .state_block_table_stride = state_block_table_stride,
        .state_cache = state_cache,
        .state_stride0 = state_stride0,
        .state_stride1 = state_stride1,
        .rms_weight = rms_weight,
        .cos_sin_cache = cos_sin_cache,
        .cos_sin_stride = cos_sin_stride,
        .kv_cache = kv_cache,
        .kv_slot_mapping = kv_slot_mapping,
        .kv_cache_stride0 = kv_cache_stride0,
        .ape = ape,
        .row_ready = nullptr,
        .num_state_slots = num_state_slots,
        .warp_workspace = nullptr,
        .profile_mode = kDenseCurrentC4 ? 1u : profile_mode
    };

    sm100_bf16_gemm_body<
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        8192, 512, 7168,
        kC0BlockM, kC0BlockN, kC0BlockK, 1,
        kC0SwizzleA, kC0SwizzleB, kC0SwizzleCD, kC0NumStages,
        kC0NumNonEpilogueThreads, kC0NumEpilogueThreads,
        kC0NumMulticast, kC0IsMulticastOnA, kRightSMs,
        kC0KAlignment, kC0SwapAB, true,
        GemmType::Normal, false, float, 100, true, false, kStoreDenseC0>(
            nullptr, 8192, 512, 7168,
            c0_tensor_map_a, c0_tensor_map_b, c0_tensor_map_d, cta_idx,
            &csa_params);
    if constexpr (kStoreDenseC0)
        cute::tma_store_wait<0>();
    if (profile_mode < 3)
        return;

    // The release/acquire barrier publishes the complete paged state. The
    // compatibility route additionally publishes dense C0, while the
    // state-only route deliberately skips that unconsumed output.
    csa_root_lane_barrier<kRightSMs>(lane_barrier, cta_idx);
    const auto warp_idx = static_cast<uint32_t>(threadIdx.x) / 32;
    const auto lane_idx = static_cast<uint32_t>(threadIdx.x) % 32;
    if constexpr (kC4WarpConfig == 0) {
        // Retained reference: WG0 remains isolated at 40 registers while
        // WG1-WG3 increase from 128 to 152 registers for twelve C4 warps.
        constexpr uint32_t kC4ComputeRegisters = 152;
        constexpr uint32_t kNumC4Warps = 12;
        if (warp_idx >= 4)
            cutlass::arch::warpgroup_reg_alloc<kC4ComputeRegisters>();
        __syncthreads();
        if constexpr (kDenseCurrentC4) {
            if (cta_idx == 0 and warp_idx == 0)
                csa_store_live_tail_state(
                    current_kv_score, state_cache, state_stride0,
                    state_stride1, num_state_slots, positions,
                    state_slot_mapping, ape, lane_idx);
        }
        if (warp_idx >= 4) {
            const auto consumer_thread_idx = (warp_idx - 4) * 32 + lane_idx;
            csa_index_c4_body<
                kRightSMs, kNumC4Warps, kDenseCurrentC4>(
                cta_idx, consumer_thread_idx,
                current_kv_score,
                state_cache, state_stride0, state_stride1,
                token_to_req, positions, state_slot_mapping,
                state_block_table, state_block_table_stride,
                rms_weight, cos_sin_cache, cos_sin_stride,
                kv_cache, kv_slot_mapping, kv_cache_stride0, ape);
        }
        __syncthreads();
        if (warp_idx >= 4)
            cutlass::arch::warpgroup_reg_dealloc<128>();
        __syncthreads();
    } else if constexpr (kC4WarpConfig == 1) {
        // Standalone-like candidate: all four warpgroups use the legal
        // setmaxnreg value nearest the standalone kernel's 108-register
        // compile footprint. Fourteen warps compute; warps 14-15 stay idle.
        constexpr uint32_t kC4UniformRegisters = 112;
        constexpr uint32_t kNumC4Warps = 14;

        // C0 leaves the CTA at {40, 128, 128, 128}. Release WG1-WG3 first,
        // synchronize the complete CTA, and only then grow WG0.
        if (warp_idx >= 4)
            cutlass::arch::warpgroup_reg_dealloc<kC4UniformRegisters>();
        __syncthreads();
        if (warp_idx < 4)
            cutlass::arch::warpgroup_reg_alloc<kC4UniformRegisters>();
        __syncthreads();

        if constexpr (kDenseCurrentC4) {
            if (cta_idx == 0 and warp_idx == 14)
                csa_store_live_tail_state(
                    current_kv_score, state_cache, state_stride0,
                    state_stride1, num_state_slots, positions,
                    state_slot_mapping, ape, lane_idx);
        }
        if (warp_idx < kNumC4Warps) {
            const auto consumer_thread_idx = warp_idx * 32 + lane_idx;
            csa_index_c4_body<
                kRightSMs, kNumC4Warps, kDenseCurrentC4>(
                cta_idx, consumer_thread_idx,
                current_kv_score,
                state_cache, state_stride0, state_stride1,
                token_to_req, positions, state_slot_mapping,
                state_block_table, state_block_table_stride,
                rms_weight, cos_sin_cache, cos_sin_stride,
                kv_cache, kv_slot_mapping, kv_cache_stride0, ape);
        }
        __syncthreads();

        // Restore the C0/IW0 role layout in the opposite order.
        if (warp_idx < 4)
            cutlass::arch::warpgroup_reg_dealloc<40>();
        __syncthreads();
        if (warp_idx >= 4)
            cutlass::arch::warpgroup_reg_alloc<128>();
        __syncthreads();
    } else {
        // C0 already leaves the CTA at {40, 128, 128, 128}. Reuse that
        // layout directly for config 2. Config 3 explicitly releases and
        // reacquires the WG1-WG3 register budget at the C0 -> C4 boundary so
        // ptxas can end the C0 allocation region before compiling C4.
        constexpr uint32_t kNumC4Warps = 12;
        if constexpr (kDenseCurrentC4) {
            if (cta_idx == 0 and warp_idx == 0)
                csa_store_live_tail_state(
                    current_kv_score, state_cache, state_stride0,
                    state_stride1, num_state_slots, positions,
                    state_slot_mapping, ape, lane_idx);
        }
        if (warp_idx >= 4) {
            if constexpr (kC4WarpConfig == 3) {
                cutlass::arch::warpgroup_reg_dealloc<112>();
                cutlass::arch::warpgroup_reg_alloc<128>();
            }
            const auto consumer_thread_idx =
                (warp_idx - 4) * 32 + lane_idx;
            csa_index_c4_body<
                kRightSMs, kNumC4Warps, kDenseCurrentC4>(
                cta_idx, consumer_thread_idx,
                current_kv_score,
                state_cache, state_stride0, state_stride1,
                token_to_req, positions, state_slot_mapping,
                state_block_table, state_block_table_stride,
                rms_weight, cos_sin_cache, cos_sin_stride,
                kv_cache, kv_slot_mapping, kv_cache_stride0, ape);
        }
        __syncthreads();
    }
    if (profile_mode < 4)
        return;

    // IW0 is independent. Clusters that finish their C0/C4 task range can
    // hand off immediately; there is deliberately no device-wide barrier.
    sm100_bf16_gemm_body<
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        8192, 64, 7168,
        kIW0BlockM, kIW0BlockN, kIW0BlockK, 1,
        kIW0SwizzleA, kIW0SwizzleB, kIW0SwizzleCD, kIW0NumStages,
        kIW0NumNonEpilogueThreads, kIW0NumEpilogueThreads,
        kIW0NumMulticast, kIW0IsMulticastOnA, kRightSMs,
        kIW0KAlignment, kIW0SwapAB, true,
        GemmType::Normal, false, cutlass::bfloat16_t, 100>(
            nullptr, 8192, 64, 7168,
            iw0_tensor_map_a, iw0_tensor_map_b, iw0_tensor_map_d, cta_idx);
#else
    if (cta_idx == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports SM100");
#endif
}

template <
    uint32_t kLeftSMs, uint32_t kRightSMs, uint32_t kC4WarpConfig,
    bool kDenseCurrentC4, bool kStoreDenseC0,
    uint32_t kC0BlockM, uint32_t kC0BlockN, uint32_t kC0BlockK,
    uint32_t kC0SwizzleA, uint32_t kC0SwizzleB, uint32_t kC0SwizzleCD,
    uint32_t kC0NumStages,
    uint32_t kC0NumNonEpilogueThreads, uint32_t kC0NumEpilogueThreads,
    uint32_t kC0NumMulticast, bool kC0IsMulticastOnA,
    uint32_t kC0KAlignment, bool kC0SwapAB,
    uint32_t kIW0BlockM, uint32_t kIW0BlockN, uint32_t kIW0BlockK,
    uint32_t kIW0SwizzleA, uint32_t kIW0SwizzleB, uint32_t kIW0SwizzleCD,
    uint32_t kIW0NumStages,
    uint32_t kIW0NumNonEpilogueThreads, uint32_t kIW0NumEpilogueThreads,
    uint32_t kIW0NumMulticast, bool kIW0IsMulticastOnA,
    uint32_t kIW0KAlignment, bool kIW0SwapAB>
CUTLASS_GLOBAL void __launch_bounds__(512, 1)
index_compress_fusion_kernel(
        const __grid_constant__ cute::TmaDescriptor c0_tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor c0_tensor_map_b,
        const __grid_constant__ cute::TmaDescriptor c0_tensor_map_d,
        const __grid_constant__ cute::TmaDescriptor iw0_tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor iw0_tensor_map_b,
        const __grid_constant__ cute::TmaDescriptor iw0_tensor_map_d,
        uint32_t* lane_barrier,
        uint64_t* row_ready,
        uint32_t num_state_slots,
        float* c4_workspace,
        uint32_t profile_mode,
        float* current_kv_score,
        float* state_cache,
        uint32_t state_stride0,
        uint32_t state_stride1,
        const int32_t* token_to_req,
        const int64_t* positions,
        const int64_t* state_slot_mapping,
        const int32_t* state_block_table,
        uint32_t state_block_table_stride,
        const float* rms_weight,
        const float* cos_sin_cache,
        uint32_t cos_sin_stride,
        uint8_t* kv_cache,
        const int64_t* kv_slot_mapping,
        uint32_t kv_cache_stride0,
        const float* ape) {
    constexpr uint32_t kClusterSize = 2;
    const auto cluster_idx =
        static_cast<uint32_t>(blockIdx.x) / kClusterSize;
    if (cluster_idx < kLeftSMs / kClusterSize)
        return;
    const auto cta_idx = static_cast<uint32_t>(blockIdx.x) - kLeftSMs;
    index_compress_fusion_body<
        kLeftSMs, kRightSMs, kC4WarpConfig,
        kDenseCurrentC4, kStoreDenseC0,
        kC0BlockM, kC0BlockN, kC0BlockK,
        kC0SwizzleA, kC0SwizzleB, kC0SwizzleCD, kC0NumStages,
        kC0NumNonEpilogueThreads, kC0NumEpilogueThreads,
        kC0NumMulticast, kC0IsMulticastOnA,
        kC0KAlignment, kC0SwapAB,
        kIW0BlockM, kIW0BlockN, kIW0BlockK,
        kIW0SwizzleA, kIW0SwizzleB, kIW0SwizzleCD, kIW0NumStages,
        kIW0NumNonEpilogueThreads, kIW0NumEpilogueThreads,
        kIW0NumMulticast, kIW0IsMulticastOnA,
        kIW0KAlignment, kIW0SwapAB>(
            c0_tensor_map_a, c0_tensor_map_b, c0_tensor_map_d,
            iw0_tensor_map_a, iw0_tensor_map_b, iw0_tensor_map_d,
            lane_barrier, row_ready, num_state_slots, c4_workspace,
            profile_mode, current_kv_score, state_cache,
            state_stride0, state_stride1, token_to_req, positions,
            state_slot_mapping, state_block_table,
            state_block_table_stride, rms_weight,
            cos_sin_cache, cos_sin_stride, kv_cache,
            kv_slot_mapping, kv_cache_stride0, ape, cta_idx);
}

}  // namespace deep_gemm
