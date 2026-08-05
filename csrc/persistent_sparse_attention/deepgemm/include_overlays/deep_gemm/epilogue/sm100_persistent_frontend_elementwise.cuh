#pragma once

#include <cuda_bf16.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm::epilogue::hca_fe {

// The two elementwise phases of the merged persistent attention front-end kernel, written to
// match `fused_norm_quant_rope.py` / `per_token_group_quant_packed_into`
// statement for statement -- including the BF16 round trip before quantizing,
// which is what the serial chain actually feeds downstream.
//
// Both are warp-level: a warp owns a whole reduction group, so every reduction
// is a shuffle and none of them needs shared memory or a block barrier. That is
// what lets these run inside the GEMM's epilogue warps without disturbing the
// store pipeline's own NamedBarriers.
//
// Layout convention shared by both: within a 512-element pack, lane `l` holds
// the 16 contiguous elements `[l * 16, l * 16 + 16)`. A 128-element
// quantization group is therefore exactly 8 lanes, and the four group
// exponents of a pack end up in lanes 0, 8, 16 and 24.

constexpr float kFP8Max = 448.0f;
constexpr float kAmaxFloor = 1.0e-10f;

CUTLASS_DEVICE uint16_t fp32x2_to_e4m3x2(const float& lo, const float& hi) {
    // `cvt.rn.satfinite.e4m3x2.f32 d, a, b` puts `a` in the high byte and `b`
    // in the low byte, and saturates at +-448 -- the same clamp the Triton
    // kernel spells out before its cast.
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(out) : "f"(hi), "f"(lo));
    return out;
}

// amax -> UE8M0 scale and its exact reciprocal. The scale is a power of two, so
// multiplying by `sf_inv` is bit-identical to Triton's division by `sf`.
CUTLASS_DEVICE void ue8m0_scale(const float& amax, float& sf_inv, uint32_t& exp_bits) {
    const auto e = math::fast_log2_ceil(fmaxf(amax, kAmaxFloor) * (1.0f / kFP8Max));
    sf_inv = math::fast_pow2(-e);
    const auto sf = math::fast_pow2(e);
    exp_bits = (*reinterpret_cast<const uint32_t*>(&sf) >> 23) & 0xFFu;
}

// Assemble one packed scale word from the four group exponents of a pack.
CUTLASS_DEVICE uint32_t pack_four_exponents(const uint32_t& exp_bits) {
    return exp_bits |
           (__shfl_sync(0xffffffffu, exp_bits,  8) <<  8) |
           (__shfl_sync(0xffffffffu, exp_bits, 16) << 16) |
           (__shfl_sync(0xffffffffu, exp_bits, 24) << 24);
}

// Quantize the 16 values this lane holds of one pack, emit them as a single
// 16-byte store, and return the packed-scale word (valid in lane 0).
template <uint32_t kElemsPerLane>
CUTLASS_DEVICE uint32_t quant_pack(const float (&values)[kElemsPerLane],
                                   uint8_t* dst) {
    DG_STATIC_ASSERT(kElemsPerLane == 16, "One 16-byte store per lane per pack");

    float amax = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < kElemsPerLane; ++ i)
        amax = fmaxf(amax, fabsf(values[i]));
    // A 128-element group spans 8 lanes.
    #pragma unroll
    for (uint32_t mask = 1; mask < 8; mask <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));

    float sf_inv;
    uint32_t exp_bits;
    ue8m0_scale(amax, sf_inv, exp_bits);

    uint4 packed;
    auto* halves = reinterpret_cast<uint16_t*>(&packed);
    #pragma unroll
    for (uint32_t i = 0; i < kElemsPerLane / 2; ++ i)
        halves[i] = fp32x2_to_e4m3x2(values[2 * i] * sf_inv, values[2 * i + 1] * sf_inv);
    *reinterpret_cast<uint4*>(dst) = packed;

    return pack_four_exponents(exp_bits);
}

template <uint32_t kElemsPerLane>
CUTLASS_DEVICE void load_bf16_lane(const nv_bfloat16* src, float (&out)[kElemsPerLane]) {
    DG_STATIC_ASSERT(kElemsPerLane % 8 == 0, "BF16 loads are 16 bytes wide");
    #pragma unroll
    for (uint32_t i = 0; i < kElemsPerLane / 8; ++ i) {
        const uint4 raw = *reinterpret_cast<const uint4*>(src + i * 8);
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            const float2 v = __bfloat1622float2(pairs[j]);
            out[i * 8 + j * 2] = v.x;
            out[i * 8 + j * 2 + 1] = v.y;
        }
    }
}

// ---------------------------------------------------------------------------
// P0: BF16 hidden -> FP8 + packed UE8M0, per 128-element group.
// One warp per (row, 512-element pack).
// ---------------------------------------------------------------------------
template <uint32_t SHAPE_K, uint32_t PACK = 512, uint32_t kElemsPerLane = PACK / 32>
CUTLASS_DEVICE void p0_quant(const nv_bfloat16* hidden, const uint32_t& hidden_stride,
                             uint8_t* out_fp8, const uint32_t& fp8_stride,
                             uint32_t* out_sf, const uint32_t& sf_stride_k,
                             const uint32_t& row_begin, const uint32_t& num_rows,
                             const uint32_t& warp_idx, const uint32_t& num_warps,
                             const uint32_t& lane_idx) {
    constexpr uint32_t kNumPacks = SHAPE_K / PACK;
    DG_STATIC_ASSERT(SHAPE_K % PACK == 0, "Hidden size must be a whole number of packs");

    const auto total = num_rows * kNumPacks;

    auto one = [&](const uint32_t& item, float (&values)[kElemsPerLane],
                   const bool& do_load) {
        const auto row = row_begin + item / kNumPacks;
        const auto pack = item % kNumPacks;
        const auto offset = pack * PACK + lane_idx * kElemsPerLane;
        if (do_load) {
            load_bf16_lane(hidden + static_cast<int64_t>(row) * hidden_stride + offset, values);
            return;
        }
        const auto packed_sf = quant_pack(values,
            out_fp8 + static_cast<int64_t>(row) * fp8_stride + offset);
        if (lane_idx == 0)
            out_sf[row + pack * sf_stride_k] = packed_sf;
    };

    // Two items in flight, loads issued before either is reduced. The
    // persistent kernel's 211 KB of shared memory pins it to one CTA per SM, so this
    // phase runs at 16 warps per SM out of 64 -- a quarter of the occupancy a
    // standalone quantize kernel would get, and therefore a quarter of the
    // latency hiding. There is no second source of parallelism to fall back on
    // either: 976 x 14 = 13664 items over 2432 warps is only ~5.6 iterations.
    // `quant_pack`'s amax is a shfl chain, so with one item per iteration the
    // next load cannot issue until the current one has fully retired.
    uint32_t item = warp_idx;
    for (; item + 3 * num_warps < total; item += 4 * num_warps) {
        float v0[kElemsPerLane], v1[kElemsPerLane];
        float v2[kElemsPerLane], v3[kElemsPerLane];
        one(item, v0, true);
        one(item + num_warps, v1, true);
        one(item + 2 * num_warps, v2, true);
        one(item + 3 * num_warps, v3, true);
        one(item, v0, false);
        one(item + num_warps, v1, false);
        one(item + 2 * num_warps, v2, false);
        one(item + 3 * num_warps, v3, false);
    }
    for (; item + num_warps < total; item += 2 * num_warps) {
        float v0[kElemsPerLane], v1[kElemsPerLane];
        one(item, v0, true);
        one(item + num_warps, v1, true);
        one(item, v0, false);
        one(item + num_warps, v1, false);
    }
    for (; item < total; item += num_warps) {
        float v[kElemsPerLane];
        one(item, v, true);
        one(item, v, false);
    }
}

// ---------------------------------------------------------------------------
// F1': RMSNorm(q) + q quantization, RMSNorm(kv) + RoPE. One warp per row.
//
// The q reduction spans all 1536 columns and the kv reduction all 512, so the
// warp holds its whole row in registers (48 + 16 floats) rather than making a
// second pass over global memory.
// ---------------------------------------------------------------------------
template <uint32_t Q_RANK, uint32_t KV_DIM, uint32_t NOPE_DIM, uint32_t ROPE_DIM,
          uint32_t PACK = 512, uint32_t kElemsPerLane = PACK / 32,
          uint32_t kNumQPacks = Q_RANK / PACK>
CUTLASS_DEVICE void f1_norm_quant_rope(const nv_bfloat16* qkv, const uint32_t& qkv_stride,
                                       const nv_bfloat16* q_weight, const nv_bfloat16* kv_weight,
                                       uint8_t* q_fp8, const uint32_t& q_fp8_stride,
                                       uint32_t* q_sf, const uint32_t& q_sf_stride_k,
                                       nv_bfloat16* kv_out, const uint32_t& kv_out_stride,
                                       const int64_t* positions,
                                       const float* cos_sin_cache, const uint32_t& cos_sin_stride,
                                       const float& eps,
                                       const uint32_t& row_begin, const uint32_t& num_rows,
                                       const uint32_t& warp_idx, const uint32_t& num_warps,
                                       const uint32_t& lane_idx) {
    DG_STATIC_ASSERT(Q_RANK % PACK == 0 and KV_DIM == PACK, "Unsupported F1 shape");
    DG_STATIC_ASSERT(NOPE_DIM + ROPE_DIM == KV_DIM, "Unsupported KV split");

    const auto lane_offset = lane_idx * kElemsPerLane;

    for (uint32_t row_local = warp_idx; row_local < num_rows; row_local += num_warps) {
        const auto row = row_begin + row_local;
        const auto* row_ptr = qkv + static_cast<int64_t>(row) * qkv_stride;

        // ---- q half: RMSNorm over all Q_RANK columns, then quantize --------
        float q[kNumQPacks][kElemsPerLane];
        float ss = 0.0f;
        #pragma unroll
        for (uint32_t p = 0; p < kNumQPacks; ++ p) {
            load_bf16_lane(row_ptr + p * PACK + lane_offset, q[p]);
            #pragma unroll
            for (uint32_t i = 0; i < kElemsPerLane; ++ i)
                ss += q[p][i] * q[p][i];
        }
        #pragma unroll
        for (uint32_t mask = 1; mask < 32; mask <<= 1)
            ss += __shfl_xor_sync(0xffffffffu, ss, mask);
        const auto q_inv = rsqrtf(ss / static_cast<float>(Q_RANK) + eps);

        #pragma unroll
        for (uint32_t p = 0; p < kNumQPacks; ++ p) {
            float w[kElemsPerLane];
            load_bf16_lane(q_weight + p * PACK + lane_offset, w);
            #pragma unroll
            for (uint32_t i = 0; i < kElemsPerLane; ++ i) {
                // Round to BF16 before quantizing: this is the value the
                // separate P3 kernel would have read back.
                q[p][i] = __bfloat162float(__float2bfloat16(q[p][i] * q_inv * w[i]));
            }
            const auto packed_sf = quant_pack(q[p],
                q_fp8 + static_cast<int64_t>(row) * q_fp8_stride + p * PACK + lane_offset);
            if (lane_idx == 0)
                q_sf[row + p * q_sf_stride_k] = packed_sf;
        }

        // ---- kv half: RMSNorm over all KV_DIM columns, then RoPE -----------
        float kv[kElemsPerLane], kw[kElemsPerLane];
        load_bf16_lane(row_ptr + Q_RANK + lane_offset, kv);
        load_bf16_lane(kv_weight + lane_offset, kw);
        float kv_ss = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < kElemsPerLane; ++ i)
            kv_ss += kv[i] * kv[i];
        #pragma unroll
        for (uint32_t mask = 1; mask < 32; mask <<= 1)
            kv_ss += __shfl_xor_sync(0xffffffffu, kv_ss, mask);
        const auto kv_inv = rsqrtf(kv_ss / static_cast<float>(KV_DIM) + eps);

        #pragma unroll
        for (uint32_t i = 0; i < kElemsPerLane; ++ i)
            kv[i] = __bfloat162float(__float2bfloat16(kv[i] * kv_inv * kw[i]));

        // The RoPE half is the tail ROPE_DIM columns, so only the last few
        // lanes touch it, and both elements of a rotated pair live in the same
        // lane -- no shuffle, no gather.
        if (lane_offset + kElemsPerLane > NOPE_DIM) {
            const auto position = positions[row];
            const auto* cos_ptr = cos_sin_cache + position * cos_sin_stride;
            const auto* sin_ptr = cos_ptr + ROPE_DIM / 2;
            #pragma unroll
            for (uint32_t i = 0; i < kElemsPerLane; i += 2) {
                const auto col = lane_offset + i;
                if (col >= NOPE_DIM) {
                    const auto pair_idx = (col - NOPE_DIM) / 2;
                    const auto cos_v = cos_ptr[pair_idx];
                    const auto sin_v = sin_ptr[pair_idx];
                    const auto even = kv[i], odd = kv[i + 1];
                    kv[i]     = even * cos_v - odd * sin_v;
                    kv[i + 1] = odd * cos_v + even * sin_v;
                }
            }
        }

        // 16 BF16 is 32 bytes: two aligned 16-byte stores, never 16 two-byte
        // ones (see the FlashMLA fused_o regression for what that costs).
        DG_STATIC_ASSERT(kElemsPerLane == 16, "Two 16-byte stores per lane");
        __nv_bfloat162 pairs[kElemsPerLane / 2];
        #pragma unroll
        for (uint32_t i = 0; i < kElemsPerLane / 2; ++ i)
            pairs[i] = __float22bfloat162_rn({kv[2 * i], kv[2 * i + 1]});
        auto* dst = reinterpret_cast<uint4*>(
            kv_out + static_cast<int64_t>(row) * kv_out_stride + lane_offset);
        dst[0] = reinterpret_cast<const uint4*>(pairs)[0];
        dst[1] = reinterpret_cast<const uint4*>(pairs)[1];
    }
}

}  // namespace deep_gemm::epilogue::hca_fe
