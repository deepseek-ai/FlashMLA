#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>


namespace faithful_k4 {

constexpr uint32_t kNumHeads = 64;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kRopeDim = 64;
constexpr uint32_t kHalfRopeDim = kRopeDim / 2;
constexpr uint32_t kNopeDim = kHeadDim - kRopeDim;
constexpr uint32_t kMXFP4Block = 32;
constexpr uint32_t kNumMXFP4Blocks = kHeadDim / kMXFP4Block;
constexpr uint32_t kNumPackedBytesPerHead = kHeadDim / 2;
constexpr uint32_t kNumScaleBytesPerHead =
    kHeadDim / kMXFP4Block;

// A full 1024-thread CTA is split into four independent eight-warp teams.
// Each team cooperatively stages and quantizes one token, so a fixed-grid
// CTA processes four token sequences concurrently without a producer warp.
constexpr uint32_t kNumTeams = 4;
constexpr uint32_t kWarpsPerTeam = 8;
constexpr uint32_t kNumWarps = kNumTeams * kWarpsPerTeam;
constexpr uint32_t kNumThreads = kNumWarps * 32;
constexpr uint32_t kTeamThreads = kWarpsPerTeam * 32;

static_assert(
    kNumHeads % kWarpsPerTeam == 0,
    "Invalid head split");
static_assert(kNopeDim == 64, "IQ1-Q specialization requires NoPE=64");
static_assert(kRopeDim == 64, "IQ1-Q specialization requires RoPE=64");
static_assert(
    kTeamThreads % 32 == 0,
    "Named-barrier participant count must be warp aligned");

struct alignas(128) Stage {
    __nv_bfloat16 q[kNumHeads * kHeadDim];
    float cos_sin[kRopeDim];
};

struct SharedStorage {
    Stage teams[kNumTeams];
};

__device__ __forceinline__ uint8_t cvt_fp32x2_to_e2m1x2(
        const float low, const float high) {
    uint32_t packed;
    // PTX source 1 becomes the high nibble and source 2 the low nibble.
    // The operand reversal therefore preserves vLLM's even=low/odd=high
    // packed-byte contract.
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

__device__ __forceinline__ float bf16_roundtrip(const float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// Bit-exact finite-input equivalent of:
//   ceil(log2(max(amax, 6 * 2^-126) / 6)) + 127
// The addition rounds any non-power-of-two positive normal value up to the
// next exponent before extracting the UE8M0 byte.
__device__ __forceinline__ uint8_t ue8m0_from_amax(float amax) {
    constexpr float kMinimumAmax = 6.0f * 0x1p-126f;
    amax = fmaxf(amax, kMinimumAmax);
    const auto fp4_scale = amax * (1.0f / 6.0f);
    const auto bits = __float_as_uint(fp4_scale);
    return static_cast<uint8_t>(
        ((bits + 0x007fffffu) >> 23) & 0xffu);
}

__device__ __forceinline__ float inverse_ue8m0_scale(const uint8_t scale) {
    // scale stores exponent+127.  All finite BF16-derived IQ1-Q inputs are
    // within the normal reciprocal range.
    return __uint_as_float(
        (254u - static_cast<uint32_t>(scale)) << 23);
}

template <uint32_t kBarrierIdx>
__device__ __forceinline__ void team_barrier() {
    asm volatile("bar.sync %0, %1;" :: "r"(kBarrierIdx), "r"(kTeamThreads) : "memory");
}

__device__ __forceinline__ void team_sync(const uint32_t team_idx) {
    if (team_idx == 0)
        team_barrier<0>();
    else if (team_idx == 1)
        team_barrier<1>();
    else if (team_idx == 2)
        team_barrier<2>();
    else
        team_barrier<3>();
}

__device__ __forceinline__ void producer_copy_token(
        Stage& stage,
        const __nv_bfloat16* index_q,
        const int64_t* positions,
        const float* cos_sin_cache,
        const uint32_t token_idx,
        const uint32_t team_thread_idx) {
    constexpr uint32_t kNumQBytes =
        kNumHeads * kHeadDim * sizeof(__nv_bfloat16);
    constexpr uint32_t kNumQVectors = kNumQBytes / sizeof(uint4);
    constexpr uint32_t kNumCosSinBytes = kRopeDim * sizeof(float);
    constexpr uint32_t kNumCosSinVectors =
        kNumCosSinBytes / sizeof(uint4);

    const auto q_src = reinterpret_cast<const uint4*>(
        index_q + static_cast<uint64_t>(token_idx) *
            kNumHeads * kHeadDim);
    const auto q_dst = reinterpret_cast<uint4*>(stage.q);
#pragma unroll
    for (uint32_t vector_idx = team_thread_idx;
         vector_idx < kNumQVectors;
         vector_idx += kTeamThreads)
        q_dst[vector_idx] = q_src[vector_idx];

    const auto position = positions[token_idx];
    const auto cos_sin_src = reinterpret_cast<const uint4*>(
        cos_sin_cache + position * kRopeDim);
    const auto cos_sin_dst =
        reinterpret_cast<uint4*>(stage.cos_sin);
    if (team_thread_idx < kNumCosSinVectors)
        cos_sin_dst[team_thread_idx] =
            cos_sin_src[team_thread_idx];
}

__device__ __forceinline__ void consumer_quantize_head_pair(
        const Stage& stage,
        uint8_t* packed,
        uint8_t* scales,
        const uint32_t token_idx,
        const uint32_t first_head_idx,
        const uint32_t lane_idx) {
    constexpr uint32_t kSubwarpSize = 16;
    constexpr uint32_t kHeadsPerWarp = 2;
    const auto subwarp_idx = lane_idx / kSubwarpSize;
    const auto pair_idx = lane_idx % kSubwarpSize;
    const auto head_idx = first_head_idx + subwarp_idx;
    const auto q_row = stage.q + head_idx * kHeadDim;
    const auto packed_row = packed +
        (static_cast<uint64_t>(token_idx) * kNumHeads + head_idx) *
            kNumPackedBytesPerHead;
    const auto scale_row = scales +
        (static_cast<uint64_t>(token_idx) * kNumHeads + head_idx) *
            kNumScaleBytesPerHead;
    constexpr uint32_t kFullWarpMask = 0xffffffffu;

    static_assert(
        kHeadsPerWarp * kSubwarpSize == 32,
        "Each consumer warp must stay fully active");

#pragma unroll
    for (uint32_t block_idx = 0;
         block_idx < kNumMXFP4Blocks;
         ++block_idx) {
        const auto dim_idx =
            block_idx * kMXFP4Block + pair_idx * 2;
        auto low = __bfloat162float(q_row[dim_idx]);
        auto high = __bfloat162float(q_row[dim_idx + 1]);
        if (block_idx >= kNopeDim / kMXFP4Block) {
            const auto rope_pair_idx =
                (block_idx - kNopeDim / kMXFP4Block) *
                    kSubwarpSize +
                pair_idx;
            const auto cos_value = stage.cos_sin[rope_pair_idx];
            const auto sin_value =
                stage.cos_sin[rope_pair_idx + kHalfRopeDim];
            // Match Triton's generated FP32 instruction sequence, including
            // which product rounds before the FMA:
            //   even = fma(low, cos, -mul(high, sin))
            //   odd  = fma(low, sin,  mul(high, cos))
            // Plain C++ expressions let nvcc contract the opposite product
            // (and previously left the even path as mul+mul+sub), which can
            // cross a BF16/FP4 tie while leaving the block scale unchanged.
            const auto high_sin = __fmul_rn(high, sin_value);
            const auto rotated_low =
                __fmaf_rn(low, cos_value, -high_sin);
            const auto high_cos = __fmul_rn(high, cos_value);
            const auto rotated_high =
                __fmaf_rn(low, sin_value, high_cos);
            low = bf16_roundtrip(rotated_low);
            high = bf16_roundtrip(rotated_high);
        }

        auto amax = fmaxf(fabsf(low), fabsf(high));
#pragma unroll
        for (uint32_t offset = kSubwarpSize / 2;
             offset > 0;
             offset /= 2)
            amax = fmaxf(
                amax,
                __shfl_down_sync(
                    kFullWarpMask, amax, offset, kSubwarpSize));

        uint32_t scale = 0;
        if (pair_idx == 0)
            scale = ue8m0_from_amax(amax);
        scale = __shfl_sync(
            kFullWarpMask,
            scale,
            subwarp_idx * kSubwarpSize,
            32);
        const auto inv_scale = inverse_ue8m0_scale(
            static_cast<uint8_t>(scale));
        packed_row[block_idx * kSubwarpSize + pair_idx] =
            cvt_fp32x2_to_e2m1x2(
                low * inv_scale, high * inv_scale);
        if (pair_idx == 0)
            scale_row[block_idx] = static_cast<uint8_t>(scale);
    }
}

__global__
__launch_bounds__(kNumThreads, 1)
void sm100_iq1_q_persistent_kernel(
        const int64_t* positions,
        const __nv_bfloat16* index_q,
        const float* cos_sin_cache,
        uint8_t* packed,
        uint8_t* scales,
        const uint32_t num_tokens) {
    extern __shared__ __align__(128) uint8_t smem_buffer[];
    const auto storage =
        reinterpret_cast<SharedStorage*>(smem_buffer);
    const auto warp_idx = static_cast<uint32_t>(threadIdx.x) / 32;
    const auto lane_idx = static_cast<uint32_t>(threadIdx.x) % 32;
    const auto cta_idx = static_cast<uint32_t>(blockIdx.x);
    const auto num_ctas = static_cast<uint32_t>(gridDim.x);

    const auto team_idx = warp_idx / kWarpsPerTeam;
    const auto team_warp_idx = warp_idx % kWarpsPerTeam;
    const auto team_thread_idx = team_warp_idx * 32 + lane_idx;
    for (uint32_t token_idx = cta_idx + team_idx * num_ctas;
         token_idx < num_tokens;
         token_idx += num_ctas * kNumTeams) {
        auto& stage = storage->teams[team_idx];
        producer_copy_token(
            stage,
            index_q,
            positions,
            cos_sin_cache,
            token_idx,
            team_thread_idx);
        team_sync(team_idx);

        // Each team warp owns eight heads as four two-head subwarp waves.
#pragma unroll
        for (uint32_t pair_wave_idx = 0;
             pair_wave_idx < kNumHeads / kWarpsPerTeam / 2;
             ++pair_wave_idx) {
            const auto first_head_idx =
                team_warp_idx * (kNumHeads / kWarpsPerTeam) +
                pair_wave_idx * 2;
            consumer_quantize_head_pair(
                stage,
                packed,
                scales,
                token_idx,
                first_head_idx,
                lane_idx);
        }
        team_sync(team_idx);
    }
}

}  // namespace faithful_k4

extern "C" cudaError_t launch_index_query_quant(
    const int64_t* positions, const void* index_q, const float* cos_sin_cache,
    uint8_t* packed, uint8_t* scales, uint32_t num_tokens,
    uint32_t num_sms, cudaStream_t stream) {
  constexpr uint32_t kSmemBytes = sizeof(faithful_k4::SharedStorage);
  cudaError_t status = cudaFuncSetAttribute(
      faithful_k4::sm100_iq1_q_persistent_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (status != cudaSuccess) return status;
  faithful_k4::sm100_iq1_q_persistent_kernel<<<
      num_sms, faithful_k4::kNumThreads, kSmemBytes, stream>>>(
      positions, reinterpret_cast<const __nv_bfloat16*>(index_q),
      cos_sin_cache, packed, scales, num_tokens);
  return cudaGetLastError();
}
