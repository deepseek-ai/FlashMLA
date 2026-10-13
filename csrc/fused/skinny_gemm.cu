// Weight-streaming skinny GEMM (see skinny_gemm.h).
//
// The decode-layer projections are pure weight streams: at T <= 64 tokens every weight byte is used T times, far below
// the ~80 FLOP/B ridge of the 3090, so the kernel is built to keep enough weight bytes in flight on all 82 SMs:
//   grid  = (N / N_TILE, splits, G); one CTA owns N_TILE weight rows x all T tokens x a contiguous K range
//   block = 128 threads; the K range is consumed in 64-element chunks through a cp.async ring of kStages stages
//           (weight rows and token rows are both 128-byte smem rows, 16-byte units XOR-swizzled by row)
//   math  = mma.sync m16n8k16 with the operands swapped: the weight tile is the A operand (16 rows per m-tile) and the
//           tokens are the B operand (8 per n-tile), so T = 1..8 wastes nothing beyond the n8 granule.  Warp w takes
//           k16 step w of every chunk (split-K inside the CTA), so warps never load the same fragment twice.
//   end   = the 4 warp partials are summed in warp order through smem; with splits > 1 each CTA stores its fp32
//           partial, and the last CTA of the tile (atomic counter) adds the partials in split order and writes y.
#include "skinny_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 128;
constexpr int kChunk = 64;  // K elements per pipeline stage
#ifndef SKINNY_RING_BYTES
#define SKINNY_RING_BYTES 49152  // target cp.async ring per CTA (bytes in flight; 32/48/64 KB within 2 %, R-051), 3..8 stages
#endif

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr) : "memory");
}

// byte offset of 16-byte unit u (0..7) of 128-byte smem row `row`
__device__ __forceinline__ uint32_t swz(int row, int u) { return row * 128 + ((u ^ (row & 7)) << 4); }

template <typename E> struct Mma;
template <> struct Mma<__nv_bfloat16> {
    static __device__ __forceinline__ void run(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
        __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<uint32_t *>(&v);
    }
};
template <> struct Mma<__half> {
    static __device__ __forceinline__ void run(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
        __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<uint32_t *>(&v);
    }
};

template <int N_TILE, int T8>
struct Cfg {
    static constexpr int kMT = N_TILE / 16;                  // m16 weight tiles per warp
    static constexpr int kTPad = 8 * T8;                     // token rows in smem (T rounded up to the n8 granule)
    static constexpr int kStageBytes = (N_TILE + kTPad) * 128;
    static constexpr int kStagesFit = SKINNY_RING_BYTES / kStageBytes;
    static constexpr int kStages = kStagesFit < 3 ? 3 : (kStagesFit > 8 ? 8 : kStagesFit);
    static constexpr int kRedStride = N_TILE + 4;            // floats per token row of the cross-warp buffer (conflict free)
    static constexpr int kRedBytes = 4 * kTPad * kRedStride * 4;
    // only the ring slots a CTA can use are allocated: min(kStages, chunks per split) (small K -> more CTAs per SM)
    static int smem_bytes(int max_chunks) {
        const int ring = (max_chunks < kStages ? max_chunks : kStages) * kStageBytes;
        return ring > kRedBytes ? ring : kRedBytes;
    }
};

template <typename E, int N_TILE, int T8>
__global__ void __launch_bounds__(kThreads) skinny_gemm_kernel(const __grid_constant__ SkinnyGemmParams p) {
    using C = Cfg<N_TILE, T8>;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ int s_last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int tile = blockIdx.x, split = blockIdx.y, g = blockIdx.z;
    const int T = p.T, K = p.K, n0 = tile * N_TILE;
    const E *xg = reinterpret_cast<const E *>(p.x) + g * p.x_sg;
    const E *wg = reinterpret_cast<const E *>(p.w) + ((int64_t)g * p.N + n0) * K;
    const int nct = K / kChunk;
    const int c_begin = (int)((int64_t)split * nct / p.splits);
    const int nc = (int)((int64_t)(split + 1) * nct / p.splits) - c_begin;
    const uint32_t sbase = smem_u32(smem);

    // padding token rows (T..kTPad) are never written by cp.async: zero them once in every slot this CTA uses
    if (T < C::kTPad) {
        const int per_stage = (C::kTPad - T) * 8;
        const int slots = nc < C::kStages ? nc : C::kStages;
        for (int i = tid; i < slots * per_stage; i += kThreads) {
            const int st = i / per_stage, r = i - st * per_stage;
            *reinterpret_cast<uint4 *>(smem + st * C::kStageBytes + (N_TILE + T + (r >> 3)) * 128 + (r & 7) * 16) = make_uint4(0, 0, 0, 0);
        }
    }

    auto load = [&](int c, int slot) {
        const int k0 = (c_begin + c) * kChunk;
        const uint32_t sw = sbase + slot * C::kStageBytes;
#pragma unroll
        for (int i = 0; i < N_TILE * 8 / kThreads; ++i) {
            const int idx = tid + i * kThreads, row = idx >> 3, u = idx & 7;
            cp_async_16(sw + swz(row, u), wg + (int64_t)row * K + k0 + u * 8);
        }
        const uint32_t sx = sw + N_TILE * 128;
        for (int idx = tid; idx < T * 8; idx += kThreads) {
            const int row = idx >> 3, u = idx & 7;
            cp_async_16(sx + swz(row, u), xg + (int64_t)row * p.x_st + k0 + u * 8);
        }
    };

    float acc[C::kMT][T8][4];
#pragma unroll
    for (int mt = 0; mt < C::kMT; ++mt)
#pragma unroll
        for (int nt = 0; nt < T8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

#pragma unroll
    for (int s = 0; s < C::kStages - 1; ++s) {
        if (s < nc) load(s, s);
        cp_async_commit();
    }
    int rd = 0, wr = C::kStages - 1;
#pragma unroll 1
    for (int c = 0; c < nc; ++c) {
        cp_async_wait<C::kStages - 2>();
        __syncthreads();  // stage c landed for everyone; everyone is done with stage c-1 (= the slot refilled below)
        if (c + C::kStages - 1 < nc) load(c + C::kStages - 1, wr);
        cp_async_commit();
        wr = (wr + 1 == C::kStages) ? 0 : wr + 1;

        const uint32_t sw = sbase + rd * C::kStageBytes, sx = sw + N_TILE * 128;
        rd = (rd + 1 == C::kStages) ? 0 : rd + 1;
        uint32_t a[C::kMT][4];
#pragma unroll
        for (int mt = 0; mt < C::kMT; ++mt) ldsm_x4(a[mt], sw + swz(mt * 16 + (lane & 15), 2 * warp + (lane >> 4)));
        uint32_t b[T8][2];
        if constexpr (T8 == 1) {
            ldsm_x2(b[0], sx + swz(lane & 7, 2 * warp + ((lane >> 3) & 1)));
        } else {
#pragma unroll
            for (int np = 0; np < T8 / 2; ++np) {
                uint32_t r[4];
                ldsm_x4(r, sx + swz(np * 16 + (lane & 7) + ((lane >> 4) << 3), 2 * warp + ((lane >> 3) & 1)));
                b[2 * np][0] = r[0];
                b[2 * np][1] = r[1];
                b[2 * np + 1][0] = r[2];
                b[2 * np + 1][1] = r[3];
            }
        }
#pragma unroll
        for (int mt = 0; mt < C::kMT; ++mt)
#pragma unroll
            for (int nt = 0; nt < T8; ++nt) Mma<E>::run(acc[mt][nt], a[mt], b[nt][0], b[nt][1]);
    }
    cp_async_wait<0>();
    __syncthreads();  // ring is free: reuse it for the cross-warp reduction

    // C fragment (swapped operands): c0/c1 = weight row r, tokens cq/cq+1; c2/c3 = weight row r+8
    float *red = reinterpret_cast<float *>(smem);
    {
        float *rw = red + warp * C::kTPad * C::kRedStride;
        const int r = lane >> 2, cq = 2 * (lane & 3);
#pragma unroll
        for (int mt = 0; mt < C::kMT; ++mt)
#pragma unroll
            for (int nt = 0; nt < T8; ++nt) {
                const int n = mt * 16 + r, t = nt * 8 + cq;
                rw[t * C::kRedStride + n] = acc[mt][nt][0];
                rw[(t + 1) * C::kRedStride + n] = acc[mt][nt][1];
                rw[t * C::kRedStride + n + 8] = acc[mt][nt][2];
                rw[(t + 1) * C::kRedStride + n + 8] = acc[mt][nt][3];
            }
    }
    __syncthreads();

    constexpr int NH = N_TILE / 2;
    const bool split_k = p.splits > 1;
    const int64_t tile_id = (int64_t)g * gridDim.x + tile;
    float *part = split_k ? p.ws + (tile_id * p.splits + split) * (int64_t)T * N_TILE : nullptr;
    E *yg = reinterpret_cast<E *>(p.y) + g * p.y_sg + n0;
    for (int e = tid; e < T * NH; e += kThreads) {
        const int t = e / NH, n = 2 * (e - t * NH);
        float v0 = 0.f, v1 = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 v = *reinterpret_cast<const float2 *>(red + (w * C::kTPad + t) * C::kRedStride + n);
            v0 += v.x;
            v1 += v.y;
        }
        if (split_k) *reinterpret_cast<float2 *>(part + t * N_TILE + n) = make_float2(v0, v1);
        else *reinterpret_cast<uint32_t *>(yg + (int64_t)t * p.y_st + n) = Mma<E>::pack(v0, v1);
    }
    if (!split_k) return;

    // split-K: the last CTA of this tile reduces all partials in split order (threadFenceReduction pattern)
    __threadfence();
    __syncthreads();
    int *ctr = p.counters + tile_id;
    if (tid == 0) s_last = atomicAdd(ctr, 1) == p.splits - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const float *base = p.ws + tile_id * p.splits * (int64_t)T * N_TILE;
    for (int e = tid; e < T * NH; e += kThreads) {
        const int t = e / NH, n = 2 * (e - t * NH);
        float v0 = 0.f, v1 = 0.f;
        for (int s = 0; s < p.splits; ++s) {
            const float2 v = __ldcg(reinterpret_cast<const float2 *>(base + (int64_t)s * T * N_TILE + t * N_TILE + n));
            v0 += v.x;
            v1 += v.y;
        }
        *reinterpret_cast<uint32_t *>(yg + (int64_t)t * p.y_st + n) = Mma<E>::pack(v0, v1);
    }
    if (tid == 0) *ctr = 0;  // ready for the next launch / graph replay
}

template <typename E, int N_TILE, int T8>
void launch_cfg(const SkinnyGemmParams &p, cudaStream_t stream) {
    using C = Cfg<N_TILE, T8>;
    auto kernel = skinny_gemm_kernel<E, N_TILE, T8>;
    const int nct = p.K / kChunk;
    const int smem = C::smem_bytes((nct + p.splits - 1) / p.splits);
    // the 48 KB default limit counts static smem (s_last) too
    if (smem > 46 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(p.N / N_TILE, p.splits, p.G), kThreads, smem, stream>>>(p);
}

template <typename E, int N_TILE>
void launch_t(const SkinnyGemmParams &p, cudaStream_t stream) {
    if (p.T <= 8) launch_cfg<E, N_TILE, 1>(p, stream);
    else if (p.T <= 16) launch_cfg<E, N_TILE, 2>(p, stream);
    else if (p.T <= 32) launch_cfg<E, N_TILE, 4>(p, stream);
    else launch_cfg<E, N_TILE, 8>(p, stream);
}

template <typename E>
void launch_n(const SkinnyGemmParams &p, cudaStream_t stream) {
    if (p.n_tile == 16) launch_t<E, 16>(p, stream);
    else if (p.n_tile == 32) launch_t<E, 32>(p, stream);
    else launch_t<E, 64>(p, stream);
}

}  // namespace

void launch_skinny_gemm(const SkinnyGemmParams &p, cudaStream_t stream) {
    if (p.is_bf16) launch_n<__nv_bfloat16>(p, stream);
    else launch_n<__half>(p, stream);
}
