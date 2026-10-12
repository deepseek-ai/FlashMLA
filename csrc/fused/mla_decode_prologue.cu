// Fused MLA decode prologue for sm_86 (works on any sm_80+): see mla_decode_prologue.h.
//
// grid  = (H + 1, T): blockIdx.x < H -> q head, blockIdx.x == H -> the token's KV row; blockIdx.y = token
// block = 96 threads (3 warps)
//   q head block : warps 0-1 copy q_nope (512 elems, one 16 B vector per thread), warp 2 rotates q_pe (32 pairs)
//   kv block     : warps 0-1 load kv_c, optional RMSNorm (block reduction), optional per-token quantization
//                  (block amax reduction), 16 B / 8 B stores; warp 2 rotates k_pe (32 pairs)
// All math is fp32 and rounds to the cache dtype exactly once, mirroring the unfused PyTorch reference.
#include "mla_decode_prologue.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstring>

namespace {

constexpr int kLat = 512;       // latent (kv_c) dims
constexpr int kRowElems = 576;  // native cache row: 512 latent + 64 rope
constexpr int kQRowBytes = 640; // quantized cache row: 512 B latent + 64 bf16 rope

template <bool BF16>
__device__ __forceinline__ float bits2f(uint16_t b) {
    if constexpr (BF16) return __uint_as_float(static_cast<uint32_t>(b) << 16);
    else return __half2float(__ushort_as_half(b));
}

template <bool BF16>
__device__ __forceinline__ uint16_t f2bits(float f) {
    if constexpr (BF16) return __bfloat16_as_ushort(__float2bfloat16_rn(f));
    else return __half_as_ushort(__float2half_rn(f));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// One rotation pair (i1, i2) of the 64 rope dims. neox: (i, i+32); interleaved: (2i, 2i+1). Same frequency index i.
template <bool BF16, bool INTERLEAVED>
__device__ __forceinline__ void rope_pair(const uint16_t *src, uint16_t *dst, int i, const float *cs) {
    const int i1 = INTERLEAVED ? 2 * i : i;
    const int i2 = INTERLEAVED ? 2 * i + 1 : i + 32;
    const float x1 = bits2f<BF16>(src[i1]);
    const float x2 = bits2f<BF16>(src[i2]);
    const float c = cs[i], s = cs[32 + i];
    dst[i1] = f2bits<BF16>(x1 * c - x2 * s);
    dst[i2] = f2bits<BF16>(x2 * c + x1 * s);
}

template <bool BF16, int KV_DTYPE, bool INTERLEAVED, bool NORM>
__global__ void __launch_bounds__(96) mla_decode_prologue_kernel(const MlaPrologueParams p) {
    const int t = blockIdx.y;
    const int hb = blockIdx.x;
    const int tid = threadIdx.x;
    const int b = t / p.s_q;
    const int j = t - b * p.s_q;
    const int pos = __ldg(p.seqlens + b) - p.s_q + j;
    if (pos < 0 || pos >= p.max_pos) return;  // block-uniform
    const float *cs = p.cos_sin + static_cast<int64_t>(pos) * 64;

    if (hb < p.H) {  // ------------------------------------------------------------------ q head
        uint16_t *qo = static_cast<uint16_t *>(p.q_out) + (static_cast<int64_t>(t) * p.H + hb) * kRowElems;
        if (tid < 64) {
            const uint16_t *src = static_cast<const uint16_t *>(p.q_nope) + t * p.q_nope_stride_t + hb * p.q_nope_stride_h;
            reinterpret_cast<uint4 *>(qo)[tid] = __ldg(reinterpret_cast<const uint4 *>(src) + tid);
        } else {
            const uint16_t *pe = static_cast<const uint16_t *>(p.q_pe) + t * p.q_pe_stride_t + hb * p.q_pe_stride_h;
            rope_pair<BF16, INTERLEAVED>(pe, qo + kLat, tid - 64, cs);
        }
        return;
    }

    // ---------------------------------------------------------------------------------- KV row
    const int blk = __ldg(p.block_table + static_cast<int64_t>(b) * p.block_table_stride + pos / p.page);
    const int64_t slot = static_cast<int64_t>(blk) * p.page + (pos % p.page);
    __shared__ float s_red[2];
    __shared__ float s_amax[2];
    const int warp = tid >> 5, lane = tid & 31;

    float v[8];
    if (tid < 64) {
        const uint16_t *src = static_cast<const uint16_t *>(p.kv_c) + t * p.kv_c_stride_t;
        const uint4 raw = __ldg(reinterpret_cast<const uint4 *>(src) + tid);
        const uint16_t *e = reinterpret_cast<const uint16_t *>(&raw);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = bits2f<BF16>(e[k]);
        if constexpr (NORM) {
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
            ss = warp_sum(ss);
            if (lane == 0) s_red[warp] = ss;
        }
    }
    __syncthreads();

    uint16_t *rope_dst;
    if constexpr (KV_DTYPE == 0) rope_dst = static_cast<uint16_t *>(p.kv_cache) + slot * kRowElems + kLat;
    else rope_dst = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(p.kv_cache) + slot * kQRowBytes + kLat);

    if (tid < 64) {
        if constexpr (NORM) {
            // HF DeepseekV2RMSNorm: weight * (x * rsqrt(mean(x^2) + eps)).to(dtype) -- cast to dtype BEFORE the weight product
            const float r = rsqrtf((s_red[0] + s_red[1]) * (1.f / kLat) + p.eps);
            const uint4 wraw = __ldg(reinterpret_cast<const uint4 *>(p.norm_w) + tid);
            const uint16_t *we = reinterpret_cast<const uint16_t *>(&wraw);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float tn = bits2f<BF16>(f2bits<BF16>(v[k] * r));
                v[k] = bits2f<BF16>(f2bits<BF16>(tn * bits2f<BF16>(we[k])));
            }
        }
        if constexpr (KV_DTYPE == 0) {
            uint4 out;
            uint16_t *o = reinterpret_cast<uint16_t *>(&out);
#pragma unroll
            for (int k = 0; k < 8; ++k) o[k] = f2bits<BF16>(v[k]);
            reinterpret_cast<uint4 *>(static_cast<uint16_t *>(p.kv_cache) + slot * kRowElems)[tid] = out;
        } else {
            float am = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) am = fmaxf(am, fabsf(v[k]));
            am = warp_max(am);
            if (lane == 0) s_amax[warp] = am;
        }
    } else {
        const uint16_t *kpe = static_cast<const uint16_t *>(p.k_pe) + t * p.k_pe_stride_t;
        rope_pair<BF16, INTERLEAVED>(kpe, rope_dst, tid - 64, cs);
    }

    if constexpr (KV_DTYPE != 0) {
        __syncthreads();
        if (tid < 64) {
            const float amax = fmaxf(s_amax[0], s_amax[1]);
            // docs/kv_quant_format.md: scale = amax / 448 (e4m3) or amax / 127 (int8); amax == 0 -> scale 1, q = 0
            const float scale = amax > 0.f ? __fdiv_rn(amax, KV_DTYPE == 1 ? 448.f : 127.f) : 1.f;
            uint8_t q[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float y = __fdiv_rn(v[k], scale);
                if constexpr (KV_DTYPE == 1) {
                    q[k] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(y, __NV_SATFINITE, __NV_E4M3));
                } else {
                    int qi = __float2int_rn(y);
                    qi = max(-127, min(127, qi));
                    q[k] = static_cast<uint8_t>(static_cast<int8_t>(qi));
                }
            }
            uint2 packed;
            memcpy(&packed, q, 8);
            reinterpret_cast<uint2 *>(static_cast<uint8_t *>(p.kv_cache) + slot * kQRowBytes)[tid] = packed;
            if (tid == 0) p.kv_scale[slot] = scale;
        }
    }
}

template <bool BF16, int KV, bool IL, bool NORM>
void launch_one(const MlaPrologueParams &p, cudaStream_t stream) {
    dim3 grid(p.H + 1, p.T);
    mla_decode_prologue_kernel<BF16, KV, IL, NORM><<<grid, 96, 0, stream>>>(p);
}

template <bool BF16, int KV, bool IL>
void launch_norm(const MlaPrologueParams &p, cudaStream_t stream) {
    if (p.norm_w != nullptr) launch_one<BF16, KV, IL, true>(p, stream);
    else launch_one<BF16, KV, IL, false>(p, stream);
}

template <bool BF16, int KV>
void launch_il(const MlaPrologueParams &p, cudaStream_t stream) {
    if (p.interleaved) launch_norm<BF16, KV, true>(p, stream);
    else launch_norm<BF16, KV, false>(p, stream);
}

}  // namespace

void launch_mla_decode_prologue(const MlaPrologueParams &p, cudaStream_t stream) {
    if (p.T == 0) return;
    if (p.is_bf16) {
        switch (p.kv_dtype) {
            case 0: launch_il<true, 0>(p, stream); break;
            case 1: launch_il<true, 1>(p, stream); break;
            case 2: launch_il<true, 2>(p, stream); break;
        }
    } else {
        launch_il<false, 0>(p, stream);  // quantized caches are bf16-only (docs/kv_quant_format.md)
    }
}
