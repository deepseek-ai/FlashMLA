#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct P {
    const __nv_bfloat16 *x, *w;
    __nv_bfloat16 *y;
    int T, N, K, G;
    long x_sg, x_st, y_sg, y_st;
    // prefetch of the next GEMM's weights: rows x first pf_lines 128-B lines of each row (row pitch pf_pitch bytes)
    const char *pf_ptr;
    long pf_rows, pf_pitch;
    int pf_lines, pf_at, pf_hint, ef;
};
constexpr int kThreads = 128, kChunk = 64, kRing = 49152;
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa(uint32_t d, const void *s) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(s) : "memory");
}
__device__ __forceinline__ void cpa_ef(uint32_t d, const void *s, uint64_t pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(d), "l"(s), "l"(pol) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void waitg() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
}
__device__ __forceinline__ uint32_t swz(int row, int u) { return row * 128 + ((u ^ (row & 7)) << 4); }
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void do_prefetch(const P &p) {
    if (p.pf_ptr == nullptr) return;
    const long total = p.pf_rows * p.pf_lines;
    const long ncta = (long)gridDim.x * gridDim.z, cta = (long)blockIdx.z * gridDim.x + blockIdx.x;
    const long b = cta * total / ncta, e = (cta + 1) * total / ncta;
    for (long i = b + threadIdx.x; i < e; i += kThreads) {
        const long r = i / p.pf_lines, l = i - r * p.pf_lines;
        const char *a = p.pf_ptr + r * p.pf_pitch + l * 128;
        if (p.pf_hint) asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(a));
        else asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a));
    }
}

template <int N_TILE>
__global__ void __launch_bounds__(kThreads) gemm(const __grid_constant__ P p) {
    constexpr int kMT = N_TILE / 16, kTPad = 8, kStageBytes = (N_TILE + kTPad) * 128;
    constexpr int kFit = kRing / kStageBytes, kS = kFit < 3 ? 3 : (kFit > 8 ? 8 : kFit);
    constexpr int kRedStride = N_TILE + 4;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, tile = blockIdx.x, g = blockIdx.z;
    const int T = p.T, K = p.K, n0 = tile * N_TILE, nc = K / kChunk;
    const __nv_bfloat16 *xg = p.x + g * p.x_sg, *wg = p.w + ((long)g * p.N + n0) * K;
    const uint32_t sb = smem_u32(smem);
    uint64_t pol = 0;
    if (p.ef) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
    if (T < kTPad) {
        const int per = (kTPad - T) * 8, slots = nc < kS ? nc : kS;
        for (int i = tid; i < slots * per; i += kThreads) {
            const int st = i / per, r = i - st * per;
            *reinterpret_cast<uint4 *>(smem + st * kStageBytes + (N_TILE + T + (r >> 3)) * 128 + (r & 7) * 16) = make_uint4(0, 0, 0, 0);
        }
    }
    auto load = [&](int c, int slot) {
        const int k0 = c * kChunk;
        const uint32_t sw = sb + slot * kStageBytes;
#pragma unroll
        for (int i = 0; i < N_TILE * 8 / kThreads; ++i) {
            const int idx = tid + i * kThreads, row = idx >> 3, u = idx & 7;
            if (p.ef) cpa_ef(sw + swz(row, u), wg + (long)row * K + k0 + u * 8, pol);
            else cpa(sw + swz(row, u), wg + (long)row * K + k0 + u * 8);
        }
        const uint32_t sx = sw + N_TILE * 128;
        for (int idx = tid; idx < T * 8; idx += kThreads) {
            const int row = idx >> 3, u = idx & 7;
            cpa(sx + swz(row, u), xg + (long)row * p.x_st + k0 + u * 8);
        }
    };
    float acc[kMT][4];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
#pragma unroll
    for (int s = 0; s < kS - 1; ++s) {
        if (s < nc) load(s, s);
        commit();
    }
    const int pf_c = nc - (kS - 1) > 0 ? nc - (kS - 1) : 0;  // first iteration that issues no more weight loads
    int rd = 0, wr = kS - 1;
#pragma unroll 1
    for (int c = 0; c < nc; ++c) {
        waitg<kS - 2>();
        __syncthreads();
        if (c + kS - 1 < nc) load(c + kS - 1, wr);
        commit();
        if (p.pf_at == 1 && c == pf_c) do_prefetch(p);
        wr = (wr + 1 == kS) ? 0 : wr + 1;
        const uint32_t sw = sb + rd * kStageBytes, sx = sw + N_TILE * 128;
        rd = (rd + 1 == kS) ? 0 : rd + 1;
        uint32_t a[kMT][4];
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) ldsm4(a[mt], sw + swz(mt * 16 + (lane & 15), 2 * warp + (lane >> 4)));
        uint32_t b[2];
        ldsm2(b, sx + swz(lane & 7, 2 * warp + ((lane >> 3) & 1)));
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) mma(acc[mt], a[mt], b[0], b[1]);
    }
    waitg<0>();
    __syncthreads();
    float *red = reinterpret_cast<float *>(smem);
    {
        float *rw = red + warp * kTPad * kRedStride;
        const int r = lane >> 2, cq = 2 * (lane & 3);
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
            const int n = mt * 16 + r, t = cq;
            rw[t * kRedStride + n] = acc[mt][0];
            rw[(t + 1) * kRedStride + n] = acc[mt][1];
            rw[t * kRedStride + n + 8] = acc[mt][2];
            rw[(t + 1) * kRedStride + n + 8] = acc[mt][3];
        }
    }
    __syncthreads();
    constexpr int NH = N_TILE / 2;
    __nv_bfloat16 *yg = p.y + g * p.y_sg + n0;
    for (int e = tid; e < T * NH; e += kThreads) {
        const int t = e / NH, n = 2 * (e - t * NH);
        float v0 = 0.f, v1 = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 v = *reinterpret_cast<const float2 *>(red + (w * kTPad + t) * kRedStride + n);
            v0 += v.x;
            v1 += v.y;
        }
        *reinterpret_cast<__nv_bfloat162 *>(yg + (long)t * p.y_st + n) = __floats2bfloat162_rn(v0, v1);
    }
    if (p.pf_at == 2) do_prefetch(p);
}
