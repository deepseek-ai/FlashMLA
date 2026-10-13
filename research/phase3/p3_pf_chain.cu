// Phase-3 feasibility: L2 prefetch chain between the four decode-layer GEMMs (standalone program).
// A copy of csrc/fused/skinny_gemm.cu (bf16, splits = 1, T <= 8) plus
//   pf_*   : at a chosen point every CTA issues prefetch.global.L2 for its share of the next GEMM's first pipeline stages
//   ef     : weight cp.async with an L2::evict_first policy (streamed weights should not evict the prefetched lines)
// Chain per layer: qkv_a [3648x2048] -> W_UK [16][512x128] -> W_UV [16][128x512] -> o_proj [2048x2048]; 27 layers with
// their own weights; one CUDA graph; time per layer.  Output of the last layer is compared bitwise across variants.
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

__global__ void fill(__nv_bfloat16 *a, long n, uint32_t seed) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        a[i] = __float2bfloat16(((h & 0xffff) / 65535.f - 0.5f) * 0.04f);
    }
}
__global__ void empty_k() {}

// decode-like: every CTA writes its fp32 partial-O share of the workspace (plain stores -> dirty L2 lines)
__global__ void scratch_write(float4 *ws, long n4, float v) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x)
        ws[i] = make_float4(v, v + 1.f, v + 2.f, v + 3.f);
}
// combine-like: one thread reads a whole 128-B line (8 float4), optionally discards it from L2 (no write-back), and the
// CTA optionally prefetches the next GEMM's weights (the combine is latency bound: DRAM is idle meanwhile)
__device__ __forceinline__ void reader_prefetch(const P &pf) {
    const long total = pf.pf_rows * pf.pf_lines;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
        const long r = i / pf.pf_lines, l = i - r * pf.pf_lines;
        asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(pf.pf_ptr + r * pf.pf_pitch + l * 128));
    }
}
// spin_ns: the kernel lasts at least that long (the real combine is latency bound for 4.3 - 6.5 us);
// pf_first: issue the prefetch before the reads (so it overlaps the latency-bound work) instead of after them
__global__ void scratch_read(const float4 *ws, long nlines, float *out, int discard, P pf, int spin_ns, int pf_first) {
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t0));
    if (pf.pf_ptr && pf_first) reader_prefetch(pf);
    float acc = 0.f;
    for (long l = blockIdx.x * (long)blockDim.x + threadIdx.x; l < nlines; l += (long)gridDim.x * blockDim.x) {
        const float4 *a = ws + l * 8;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float4 v = __ldcg(a + k);
            acc += v.x + v.y + v.z + v.w;
        }
        if (discard) asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(a) : "memory");
    }
    if (pf.pf_ptr && !pf_first) reader_prefetch(pf);
    if (acc == 12345.f) out[blockIdx.x] = acc;  // keep the loads alive
    if (spin_ns > 0) {
        unsigned long long t;
        do { asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t)); } while (t - t0 < (unsigned long long)spin_ns);
    }
}

template <int NT> int stages_of() {
    constexpr int sbytes = (NT + 8) * 128;
    const int fit = kRing / sbytes;
    return fit < 3 ? 3 : (fit > 8 ? 8 : fit);
}
template <int NT> int smem_of(int K) {
    constexpr int sbytes = (NT + 8) * 128;
    const int s = stages_of<NT>(), nc = K / 64;
    const int ring = (nc < s ? nc : s) * sbytes, red = 4 * 8 * (NT + 4) * 4;
    return ring > red ? ring : red;
}
static void launch(const P &p, int nt, cudaStream_t s) {
    if (nt == 16) gemm<16><<<dim3(p.N / 16, 1, p.G), kThreads, smem_of<16>(p.K), s>>>(p);
    else gemm<32><<<dim3(p.N / 32, 1, p.G), kThreads, smem_of<32>(p.K), s>>>(p);
}

int main(int argc, char **argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 1;
    const int L = 27;
    int dev;
    CK(cudaGetDevice(&dev));
    int l2 = 0, pl2 = 0;
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    CK(cudaDeviceGetAttribute(&pl2, cudaDevAttrMaxPersistingL2CacheSize, dev));
    printf("L2 %d B, max persisting L2 %d B, T=%d\n", l2, pl2, T);
    CK(cudaFuncSetAttribute(gemm<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CK(cudaFuncSetAttribute(gemm<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    const long nw1 = 3648L * 2048, nw2 = 16L * 512 * 128, nw3 = 16L * 128 * 512, nw4 = 2048L * 2048;
    std::vector<__nv_bfloat16 *> W1(L), W2(L), W3(L), W4(L);
    for (int l = 0; l < L; ++l) {
        CK(cudaMalloc(&W1[l], nw1 * 2));
        CK(cudaMalloc(&W2[l], nw2 * 2));
        CK(cudaMalloc(&W3[l], nw3 * 2));
        CK(cudaMalloc(&W4[l], nw4 * 2));
        fill<<<256, 256>>>(W1[l], nw1, 11 + l);
        fill<<<256, 256>>>(W2[l], nw2, 101 + l);
        fill<<<256, 256>>>(W3[l], nw3, 211 + l);
        fill<<<256, 256>>>(W4[l], nw4, 307 + l);
    }
    __nv_bfloat16 *h0, *y1, *y2, *y3, *hb[2];
    CK(cudaMalloc(&h0, T * 2048 * 2));
    CK(cudaMalloc(&y1, T * 3648 * 2));
    CK(cudaMalloc(&y2, T * 16 * 512 * 2));
    CK(cudaMalloc(&y3, T * 2048 * 2));
    CK(cudaMalloc(&hb[0], T * 2048 * 2));
    CK(cudaMalloc(&hb[1], T * 2048 * 2));
    fill<<<64, 256>>>(h0, T * 2048, 7);
    CK(cudaDeviceSynchronize());
    const int nt_bmm = T <= 16 ? 16 : 32;
    // prefetch descriptor of GEMM i of layer l: all weight rows x the lines of its first (stages - 1) chunks
    auto desc = [&](int l, int i, long &rows, long &pitch, int &lines) -> const char * {
        rows = 0; pitch = 0; lines = 0;
        if (l >= L) return nullptr;
        int K, st;
        const __nv_bfloat16 *w;
        if (i == 0) { w = W1[l]; rows = 3648; K = 2048; st = stages_of<32>(); }
        else if (i == 1) { w = W2[l]; rows = 16 * 512; K = 128; st = nt_bmm == 16 ? stages_of<16>() : stages_of<32>(); }
        else if (i == 2) { w = W3[l]; rows = 16 * 128; K = 512; st = nt_bmm == 16 ? stages_of<16>() : stages_of<32>(); }
        else { w = W4[l]; rows = 2048; K = 2048; st = stages_of<32>(); }
        pitch = (long)K * 2;
        const int nc = K / 64;
        lines = (st - 1) < nc ? st - 1 : nc;  // 64 bf16 = one 128-B line per chunk and row
        return reinterpret_cast<const char *>(w);
    };
    // S: bytes of fp32 scratch written (decode-like) and read back (combine-like) between W_UK and W_UV
    struct V { const char *name; int pf_at, hint, ef; long budget; long S; int discard, rpf; int spin = 0, pf_first = 0; };
    const long MB = 1L << 20, ALL = 1L << 40, S1 = 2 * MB + 700 * 1024;  // b1: 82 parts x 16 rows x 2 KB = 2.6 MB
    const std::string mode = argc > 2 ? argv[2] : "pf";
    std::vector<V> vs;
    if (mode == "pf")
        vs = {{"base", 0, 0, 0, 0, 0, 0, 0},
              {"ef_only", 0, 0, 1, 0, 0, 0, 0},
              {"pf_last_issue", 1, 0, 0, ALL, 0, 0, 0},
              {"pf_last_issue_el", 1, 1, 0, ALL, 0, 0, 0},
              {"pf_last_issue_el_ef", 1, 1, 1, ALL, 0, 0, 0},
              {"pf_epi", 2, 0, 0, ALL, 0, 0, 0},
              {"pf_epi_el_ef", 2, 1, 1, ALL, 0, 0, 0},
              {"pf_last_issue_el_ef_1MB", 1, 1, 1, MB, 0, 0, 0},
              {"pf_last_issue_el_ef_2MB", 1, 1, 1, 2 * MB, 0, 0, 0},
              {"base_again", 0, 0, 0, 0, 0, 0, 0}};
    else if (mode == "comb")  // combine-like reader lasting 4.3 us: does prefetching W_UV from it pay?
        vs = {{"spin4.3_discard", 0, 0, 0, 0, S1, 1, 0, 4300, 0},
              {"spin4.3_discard_rpf_first", 0, 0, 0, 0, S1, 1, 1, 4300, 1},
              {"spin4.3_discard_rpf_last", 0, 0, 0, 0, S1, 1, 1, 4300, 0},
              {"spin4.3_discard_rpf_first_pf1MB", 1, 1, 1, MB, S1, 1, 1, 4300, 1},
              {"spin4.3_discard_pf1MB", 1, 1, 1, MB, S1, 1, 0, 4300, 0},
              {"spin4.3_nodiscard", 0, 0, 0, 0, S1, 0, 0, 4300, 0},
              {"spin4.3_nodiscard_rpf_first", 0, 0, 0, 0, S1, 0, 1, 4300, 1},
              {"spin4.3_discard_again", 0, 0, 0, 0, S1, 1, 0, 4300, 0},
              {"spin4.3_discard_rpf_first_again", 0, 0, 0, 0, S1, 1, 1, 4300, 1}};
    else
        vs = {{"base(no scratch)", 0, 0, 0, 0, 0, 0, 0},
              {"scratch2.7MB", 0, 0, 0, 0, S1, 0, 0},
              {"scratch2.7MB_discard", 0, 0, 0, 0, S1, 1, 0},
              {"scratch2.7MB_discard_rpf", 0, 0, 0, 0, S1, 1, 1},
              {"scratch2.7MB_rpf", 0, 0, 0, 0, S1, 0, 1},
              {"scratch2.7MB_discard_rpf_pf1MB", 1, 1, 1, MB, S1, 1, 1},
              {"scratch5.4MB", 0, 0, 0, 0, 2 * S1, 0, 0},
              {"scratch5.4MB_discard", 0, 0, 0, 0, 2 * S1, 1, 0},
              {"scratch2.7MB_again", 0, 0, 0, 0, S1, 0, 0},
              {"base_again", 0, 0, 0, 0, 0, 0, 0}};
    float4 *scr;
    float *rout;
    CK(cudaMalloc(&scr, 4 * S1 + 4096));
    CK(cudaMalloc(&rout, 4096 * 4));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    std::vector<char> ref(T * 2048 * 2), out(T * 2048 * 2);
    for (size_t vi = 0; vi < vs.size(); ++vi) {
        const V &v = vs[vi];
        auto step = [&]() {
            const __nv_bfloat16 *h = h0;
            for (int l = 0; l < L; ++l) {
                __nv_bfloat16 *ho = hb[l & 1];
                const __nv_bfloat16 *xs[4] = {h, y1, y2, y3};
                __nv_bfloat16 *ys[4] = {y1, y2, y3, ho};
                const __nv_bfloat16 *ws[4] = {W1[l], W2[l], W3[l], W4[l]};
                const int Ns[4] = {3648, 512, 128, 2048}, Ks[4] = {2048, 128, 512, 2048}, Gs[4] = {1, 16, 16, 1};
                const long xsg[4] = {0, 192, 512, 0}, xst[4] = {2048, 3648, 16 * 512, 2048};
                const long ysg[4] = {0, 512, 128, 0}, yst[4] = {3648, 16 * 512, 2048, 2048};
                const int nts[4] = {32, nt_bmm, nt_bmm, 32};
                for (int i = 0; i < 4; ++i) {
                    P p{};
                    p.x = xs[i]; p.w = ws[i]; p.y = ys[i]; p.T = T; p.N = Ns[i]; p.K = Ks[i]; p.G = Gs[i];
                    p.x_sg = xsg[i]; p.x_st = xst[i]; p.y_sg = ysg[i]; p.y_st = yst[i];
                    p.ef = v.ef; p.pf_at = v.pf_at; p.pf_hint = v.hint;
                    if (i == 2 && v.S > 0) {  // decode-like partial write + combine-like read before W_UV
                        scratch_write<<<82, 128, 0, s>>>(scr, v.S / 16, (float)l);
                        P rp{};
                        if (v.rpf) {   // the combine-like kernel prefetches all of W_UV (2 MB)
                            rp.pf_ptr = reinterpret_cast<const char *>(W3[l]);
                            rp.pf_rows = 16 * 128; rp.pf_pitch = 1024; rp.pf_lines = 8;
                        }
                        scratch_read<<<64, 512, 0, s>>>(scr, v.S / 128, rout, v.discard, rp, v.spin, v.pf_first);
                    }
                    if (v.pf_at && !(i == 1 && v.S > 0)) {   // with scratch, W_UK's successor is not W_UV
                        const int nl = i == 3 ? l + 1 : l, ni = (i + 1) & 3;
                        long rows, pitch;
                        int lines;
                        p.pf_ptr = desc(nl, ni, rows, pitch, lines);
                        if (p.pf_ptr) {
                            const long cap = v.budget / (rows * 128);
                            if (cap < lines) lines = (int)(cap < 1 ? 1 : cap);
                        }
                        p.pf_rows = rows; p.pf_pitch = pitch; p.pf_lines = lines;
                    }
                    launch(p, nts[i], s);
                }
                h = ho;
            }
        };
        step();
        CK(cudaStreamSynchronize(s));
        CK(cudaGetLastError());
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        step();
        CK(cudaStreamEndCapture(s, &gr));
        CK(cudaGraphInstantiate(&ge, gr, 0));
        for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int R = 50;
        std::vector<float> ts;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(e0, s);
            for (int i = 0; i < R; ++i) cudaGraphLaunch(ge, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            ts.push_back(ms * 1000.f / R / L);
        }
        std::sort(ts.begin(), ts.end());
        CK(cudaMemcpy(out.data(), hb[(L - 1) & 1], out.size(), cudaMemcpyDeviceToHost));
        if (vi == 0) ref = out;
        printf("%-26s T=%d  us/layer median %.2f (min %.2f max %.2f)  bitwise_equal=%d\n", v.name, T, ts[2], ts[0], ts[4],
               (int)(memcmp(ref.data(), out.data(), ref.size()) == 0));
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(gr);
    }
    {  // empty-kernel chain: cost of one dependent launch inside a graph
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < 108; ++i) empty_k<<<82, 128, 0, s>>>();
        CK(cudaStreamEndCapture(s, &gr));
        CK(cudaGraphInstantiate(&ge, gr, 0));
        for (int i = 0; i < 5; ++i) cudaGraphLaunch(ge, s);
        cudaStreamSynchronize(s);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0, s);
        for (int i = 0; i < 50; ++i) cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        printf("empty kernel (82x128) in graph: %.3f us per kernel\n", ms * 1000.f / 50 / 108);
    }
    return 0;
}
