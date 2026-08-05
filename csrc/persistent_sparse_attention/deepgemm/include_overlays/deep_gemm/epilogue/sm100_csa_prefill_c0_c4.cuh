#pragma once

#include <cuda_bf16.h>
#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

// Runtime arguments used only by the CSA C0 -> C4 streaming specialization.
// The generic BF16 GEMM keeps its original entrypoint and passes nullptr.
struct CSAPrefillC0C4Params {
    const int32_t* token_to_req;
    const int64_t* positions;
    const int64_t* state_slot_mapping;
    const int32_t* state_block_table;
    uint32_t state_block_table_stride;
    float* state_cache;
    uint32_t state_stride0;
    uint32_t state_stride1;
    const float* rms_weight;
    const float* cos_sin_cache;
    uint32_t cos_sin_stride;
    uint8_t* kv_cache;
    const int64_t* kv_slot_mapping;
    uint32_t kv_cache_stride0;
    const float* ape;
    uint64_t* row_ready;
    uint32_t num_state_slots;
    float* warp_workspace;
    uint32_t profile_mode;
};

CUTLASS_DEVICE void csa_stream_st_global_256(
        uint32_t* ptr, const uint32_t* values) {
    asm volatile(
        "st.global.cg.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :
        : "l"(ptr),
          "r"(values[0]), "r"(values[1]), "r"(values[2]),
          "r"(values[3]), "r"(values[4]), "r"(values[5]),
          "r"(values[6]), "r"(values[7])
        : "memory");
}

// Mirror Ray's C1 epilogue protocol: copy each completed TMEM N tile directly
// into the paged state cache, then publish one release bit per physical row.
// The fused root kernel normally stops here because C4 consumes paged state;
// an optional compatibility route also runs the ordinary dense epilogue.
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kNumUMMAStoreThreads,
          typename pattern_cd_t>
CUTLASS_DEVICE void sm100_store_csa_state_tile(
        const utils::PatternVisitor<pattern_cd_t>&,
        const uint32_t& tmem_base_addr,
        const uint32_t& base_m_idx,
        const uint32_t& base_n_idx,
        const uint32_t& shape_m,
        const uint32_t& epilogue_warp_idx,
        const uint32_t& lane_idx,
        const CSAPrefillC0C4Params& params) {
    constexpr uint32_t kHeadDim = 256;
    constexpr uint32_t kStateBlockSize = 4;
    constexpr uint32_t kNumElemsPerTMEMLoad = 16;
    constexpr uint32_t kNumElemsPerStore = 8;
    constexpr uint32_t kNumStoresPerTMEMLoad =
        kNumElemsPerTMEMLoad / kNumElemsPerStore;
    constexpr uint32_t kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    constexpr uint32_t kNumTMEMLoads = BLOCK_N / kNumElemsPerTMEMLoad;

    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0,
                     "Invalid CSA store block M");
    DG_STATIC_ASSERT(BLOCK_N == 64 or BLOCK_N == 128,
                     "CSA tile-ready protocol requires N tile 64/128");
    DG_STATIC_ASSERT(BLOCK_N % kNumElemsPerTMEMLoad == 0,
                     "CSA state tiles must be 16-channel aligned");
    DG_STATIC_ASSERT(kNumUMMAStoreThreads == STORE_BLOCK_M,
                     "CSA state epilogue requires one row owner per thread");
    DG_STATIC_ASSERT(STORE_BLOCK_N % 4 == 0,
                     "Invalid CSA state store atom");

    const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
    const auto is_score_tile = base_n_idx >= kHeadDim;

#pragma unroll
    for (uint32_t m_wave_idx = 0; m_wave_idx < kNumMWaves; ++m_wave_idx) {
        const auto global_m_idx = base_m_idx +
            m_wave_idx * STORE_BLOCK_M + epilogue_thread_idx;
        const auto valid_m = global_m_idx < shape_m;
        const auto slot_idx = valid_m ?
            params.state_slot_mapping[global_m_idx] : -1;
        const auto valid_slot = slot_idx >= 0 and
            static_cast<uint64_t>(slot_idx) < params.num_state_slots;

        float* state_row = nullptr;
        uint32_t ape_row_idx = 0;
        if (valid_slot) {
            state_row = params.state_cache +
                (static_cast<uint64_t>(slot_idx) / kStateBlockSize) *
                    params.state_stride0 +
                (static_cast<uint32_t>(slot_idx) % kStateBlockSize) *
                    params.state_stride1;
            if (is_score_tile)
                ape_row_idx = static_cast<uint32_t>(
                    params.positions[global_m_idx]) % kStateBlockSize;
        }

#pragma unroll
        for (uint32_t load_idx = 0; load_idx < kNumTMEMLoads; ++load_idx) {
            const auto tmem_addr = tmem_base_addr +
                m_wave_idx * BLOCK_N +
                load_idx * kNumElemsPerTMEMLoad;
            uint32_t raw_values[kNumElemsPerTMEMLoad];
            cute::SM100_TMEM_LOAD_32dp32b16x::copy(
                tmem_addr,
                raw_values[0], raw_values[1], raw_values[2],
                raw_values[3], raw_values[4], raw_values[5],
                raw_values[6], raw_values[7], raw_values[8],
                raw_values[9], raw_values[10], raw_values[11],
                raw_values[12], raw_values[13], raw_values[14],
                raw_values[15]);
            cutlass::arch::fence_view_async_tmem_load();

            if (valid_slot) {
#pragma unroll
                for (uint32_t store_idx = 0;
                     store_idx < kNumStoresPerTMEMLoad; ++store_idx) {
                    const auto global_n_idx = base_n_idx +
                        load_idx * kNumElemsPerTMEMLoad +
                        store_idx * kNumElemsPerStore;
                    if (is_score_tile) {
                        const auto ape_col_idx = global_n_idx - kHeadDim;
#pragma unroll
                        for (uint32_t float4_idx = 0; float4_idx < 2;
                             ++float4_idx) {
                            const auto vector_idx = store_idx * 2 + float4_idx;
                            auto values =
                                reinterpret_cast<float4*>(raw_values)[vector_idx];
                            const auto ape_values =
                                reinterpret_cast<const float4*>(
                                    params.ape +
                                    ape_row_idx * kHeadDim + ape_col_idx +
                                    float4_idx * 4)[0];
                            values.x += ape_values.x;
                            values.y += ape_values.y;
                            values.z += ape_values.z;
                            values.w += ape_values.w;
                            reinterpret_cast<float4*>(
                                raw_values)[vector_idx] = values;
                        }
                    }
                    csa_stream_st_global_256(
                        reinterpret_cast<uint32_t*>(
                            state_row + global_n_idx),
                        raw_values + store_idx * kNumElemsPerStore);
                }
            }
        }

        if (valid_slot and params.row_ready != nullptr) {
            const auto n_tile_idx = base_n_idx / BLOCK_N;
            DG_DEVICE_ASSERT(n_tile_idx < 16);
            ptx::red_or_rel_gpu(
                params.row_ready + slot_idx, 1ull << n_tile_idx);
        }
    }
}

CUTLASS_DEVICE uint8_t csa_stream_cvt_fp32x2_to_e2m1x2(
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

CUTLASS_DEVICE float csa_stream_bf16_roundtrip(const float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

CUTLASS_DEVICE float csa_stream_triton_exp(const float value) {
    const auto scaled = value * __int_as_float(0x3fb8aa3b);
    float result;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(scaled));
    return result;
}

CUTLASS_DEVICE float csa_stream_warp_sum(float value) {
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return __shfl_sync(0xffffffffu, value, 0);
}

// Eight consumer warps occupy the otherwise idle half of a 512-thread CTA.
// A warp processes one token at a time. It waits on the exact state N tiles
// needed by each of the eight rows, so C4 can follow the C0 epilogue tile by
// tile instead of waiting for the full [8192, 512] GEMM to drain.
template <uint32_t kNumConsumerThreads,
          uint32_t kNumC1Tiles,
          uint32_t kNumTaskCTAs>
CUTLASS_DEVICE void sm100_csa_c4_consumer(
        const uint32_t& shape_m,
        const uint32_t& task_cta_idx,
        const uint32_t& consumer_thread_idx,
        const CSAPrefillC0C4Params& params) {
    constexpr uint32_t kStateWidth = 256;
    constexpr uint32_t kHeadSize = 128;
    constexpr uint32_t kCompressRatio = 4;
    constexpr uint32_t kWindow = 8;
    constexpr uint32_t kStateBlockSize = 4;
    constexpr uint32_t kKVBlockSize = 64;
    constexpr uint32_t kTokenStride = 64;
    constexpr uint32_t kScaleDim = 4;
    constexpr uint32_t kRopeHeadDim = 64;
    constexpr uint32_t kWarpSize = 32;
    constexpr uint32_t kNumWarps = kNumConsumerThreads / kWarpSize;
    constexpr uint32_t kTasksPerWave = kNumTaskCTAs * kNumWarps;
    constexpr uint32_t kC1TileWidth = (2 * kStateWidth) / kNumC1Tiles;

    DG_STATIC_ASSERT(kNumConsumerThreads == 256,
                     "CSA C4 requires two consumer warpgroups");
    DG_STATIC_ASSERT(kNumC1Tiles == 4 or kNumC1Tiles == 8,
                     "CSA C4 supports 64/128-channel C1 tiles");
    DG_STATIC_ASSERT(kC1TileWidth == 64 or kC1TileWidth == 128,
                     "Invalid CSA C1 tile width");

    const auto warp_idx = consumer_thread_idx / kWarpSize;
    const auto lane_idx = consumer_thread_idx % kWarpSize;
    auto* normed_workspace =
        params.warp_workspace + warp_idx * kHeadSize;

    for (uint32_t token_idx = task_cta_idx * kNumWarps + warp_idx;
         token_idx < shape_m; token_idx += kTasksPerWave) {
        const auto slot_id = params.state_slot_mapping[token_idx];
        if (slot_id < 0)
            continue;
        const auto position = params.positions[token_idx];
        if ((position + 1) % kCompressRatio != 0)
            continue;
        const auto req_idx = params.token_to_req[token_idx];
        if (req_idx < 0)
            continue;
        const auto start = position - static_cast<int64_t>(kWindow) + 1;

        float scores[4][kWindow];
        float values[4][kWindow];
        float max_scores[4];
#pragma unroll
        for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter)
            max_scores[dim_iter] = -__int_as_float(0x7f800000);

        // Load rows in the same order as the production Triton program. The
        // acquire is per physical row and only covers the half-head consumed
        // by that row, preserving exact arithmetic while enabling streaming.
#pragma unroll
        for (uint32_t row = 0; row < kWindow; ++row) {
            const auto logical_position = start + row;
            const auto head_offset = row >= kCompressRatio ? kHeadSize : 0;
            const auto kv_first_tile = head_offset / kC1TileWidth;
            const auto kv_last_tile =
                (head_offset + kHeadSize - 1) / kC1TileWidth;
            const auto score_first_tile =
                (kStateWidth + head_offset) / kC1TileWidth;
            const auto score_last_tile =
                (kStateWidth + head_offset + kHeadSize - 1) /
                    kC1TileWidth;
            uint64_t required_tiles = 0;
#pragma unroll
            for (uint32_t tile = kv_first_tile; tile <= kv_last_tile; ++tile)
                required_tiles |= 1ull << tile;
#pragma unroll
            for (uint32_t tile = score_first_tile;
                 tile <= score_last_tile; ++tile)
                required_tiles |= 1ull << tile;

            const float* state_row = nullptr;
            bool row_valid = false;
            int64_t ready_slot = -1;
            if (logical_position >= 0) {
                const auto logical_block =
                    logical_position / kStateBlockSize;
                const auto block_number = params.state_block_table[
                    static_cast<uint64_t>(req_idx) *
                        params.state_block_table_stride + logical_block];
                if (block_number >= 0) {
                    const auto block_offset =
                        logical_position % kStateBlockSize;
                    ready_slot =
                        static_cast<int64_t>(block_number) *
                            kStateBlockSize + block_offset;
                    if (ready_slot < params.num_state_slots) {
                        state_row = params.state_cache +
                            static_cast<uint64_t>(block_number) *
                                params.state_stride0 +
                            block_offset * params.state_stride1 + head_offset;
                        row_valid = true;
                    }
                }
            }

            if (row_valid) {
                if (lane_idx == 0) {
                    const auto start_clock = clock64();
                    while ((ptx::ld_acq_gpu(
                                params.row_ready + ready_slot) &
                            required_tiles) != required_tiles) {
                        if (clock64() - start_clock >=
                            comm::kNumTimeoutCycles) {
                            printf("CSA C0-tile timeout: token=%u row=%u "
                                   "slot=%lld ready=0x%llx expected=0x%llx\n",
                                   token_idx, row,
                                   static_cast<long long>(ready_slot),
                                   static_cast<unsigned long long>(
                                       ptx::ld_acq_gpu(
                                           params.row_ready + ready_slot)),
                                   static_cast<unsigned long long>(
                                       required_tiles));
                            DG_DEVICE_ASSERT(
                                false and "CSA C0 tiles were not published");
                        }
                    }
                }
                __syncwarp();
                (void)ptx::ld_acq_gpu(params.row_ready + ready_slot);
            }

#pragma unroll
            for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter) {
                const auto dim = lane_idx + dim_iter * kWarpSize;
                auto score = -__int_as_float(0x7f800000);
                auto kv = 0.0f;
                if (row_valid) {
                    score = state_row[kStateWidth + dim];
                    kv = state_row[dim];
                }
                scores[dim_iter][row] = score;
                values[dim_iter][row] = kv;
                max_scores[dim_iter] =
                    fmaxf(max_scores[dim_iter], score);
            }
        }

        float weights[4][kWindow];
        float exp_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (uint32_t row = 0; row < kWindow; ++row) {
#pragma unroll
            for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter) {
                weights[dim_iter][row] = csa_stream_triton_exp(
                    scores[dim_iter][row] - max_scores[dim_iter]);
                exp_sums[dim_iter] += weights[dim_iter][row];
            }
        }

        float compressed[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (uint32_t row = 0; row < kWindow; ++row) {
#pragma unroll
            for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter)
                compressed[dim_iter] = fmaf(
                    weights[dim_iter][row] / exp_sums[dim_iter],
                    values[dim_iter][row], compressed[dim_iter]);
        }

        float local_sq_sum = 0.0f;
#pragma unroll
        for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter) {
            const auto dim = lane_idx + dim_iter * kWarpSize;
            normed_workspace[dim] = compressed[dim_iter];
            local_sq_sum = fmaf(
                compressed[dim_iter], compressed[dim_iter], local_sq_sum);
        }
        const auto variance_sum = csa_stream_warp_sum(local_sq_sum);
        const auto rrms = rsqrtf(
            variance_sum / static_cast<float>(kHeadSize) + 1.0e-6f);
#pragma unroll
        for (uint32_t dim_iter = 0; dim_iter < 4; ++dim_iter) {
            const auto dim = lane_idx + dim_iter * kWarpSize;
            normed_workspace[dim] *= rrms * params.rms_weight[dim];
        }
        __syncwarp();

        const auto kv_slot_idx = params.kv_slot_mapping[token_idx];
        if (kv_slot_idx < 0)
            continue;
        auto* cache_block = params.kv_cache +
            (kv_slot_idx / kKVBlockSize) * params.kv_cache_stride0;
        auto* value_ptr = cache_block +
            (kv_slot_idx % kKVBlockSize) * kTokenStride;
        auto* scale_ptr = cache_block + kKVBlockSize * kTokenStride +
            (kv_slot_idx % kKVBlockSize) * kScaleDim;
        const auto compressed_position =
            (position / kCompressRatio) * kCompressRatio;

        float pair_low[2];
        float pair_high[2];
#pragma unroll
        for (uint32_t pair_iter = 0; pair_iter < 2; ++pair_iter) {
            const auto pair_idx = lane_idx + pair_iter * kWarpSize;
            const auto even = normed_workspace[pair_idx * 2];
            const auto odd = normed_workspace[pair_idx * 2 + 1];
            float rotated_even = even;
            float rotated_odd = odd;
            if (pair_idx >= (kHeadSize - kRopeHeadDim) / 2) {
                const auto rope_pair =
                    pair_idx - (kHeadSize - kRopeHeadDim) / 2;
                const auto* cos_sin = params.cos_sin_cache +
                    compressed_position * params.cos_sin_stride;
                const auto cos_value = cos_sin[rope_pair];
                const auto sin_value =
                    cos_sin[kRopeHeadDim / 2 + rope_pair];
                rotated_even = even * cos_value - odd * sin_value;
                rotated_odd = odd * cos_value + even * sin_value;
            }
            pair_low[pair_iter] =
                csa_stream_bf16_roundtrip(rotated_even);
            pair_high[pair_iter] =
                csa_stream_bf16_roundtrip(rotated_odd);
        }

#pragma unroll
        for (uint32_t block = 0; block < 4; ++block) {
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
                    csa_stream_cvt_fp32x2_to_e2m1x2(
                        low * inv_scale, high * inv_scale);
                if (lane_idx == 0)
                    scale_ptr[block] =
                        static_cast<uint8_t>(exponent + 127.0f);
            }
        }
    }
}

}  // namespace deep_gemm::epilogue
