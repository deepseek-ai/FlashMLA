// FlashMLA decode kernel for a quantized (fp8 e4m3 / int8) KV cache on sm_86 (no 8-bit tensor cores, no cvt.e4m3).
// Format: docs/kv_quant_format.md -- per token 640 B = 512 B latent (8 bit) + 64 bf16/fp16 rope dims, plus an fp32 scale.
//
// Same CTA structure as the bf16 kernel (Q in registers, QK split-K over 4 warps, PV split by 128 V columns) but the
// 8-bit tile stays 8-bit in shared memory and is converted to bf16/fp16 in registers right before the MMAs:
//   * QK B-fragments : plain ldmatrix on the fp8 bytes (8 tokens x 16 dims per 8x8 "b16" matrix).  The 4 bytes of a lane
//     (dims 4t..4t+3) are converted to two bf16x2 registers holding dim pairs (0,2) and (1,3); the Q fragments are
//     loaded with the same dim permutation, so the dot product is unchanged.
//   * PV B-fragments: ldmatrix.trans on the fp8 bytes hands every lane [tok(2t): dims 2g,2g+1 | tok(2t+1): dims 2g,2g+1];
//     the even dims form one MMA n-tile and the odd dims another (a free permutation of the output columns that makes every
//     lane own 4 contiguous output dims in the epilogue).
//   * The per-token scale is applied in fp32 outside the MMAs:  S = scale_t * (Q_nope . q_t) + Q_rope . k_rope_t   and
//     O += (P * scale_t) . q_t   (the softmax row sums use the unscaled P).
//   * KV groups (KG = 2, BM 16): warps 4-7 form a second set of 4 warps that takes the odd tiles of every part (own ring,
//     own S exchange buffer, named barriers 1+kg / 3+kg instead of the CTA barrier), so each SM sub-partition holds two
//     warps that interleave HMMA bursts with dequant work (R-048) without doubling the split count; the two (m, l, O)
//     states are merged through smem in a fixed order at the end of a segment and KV group 0 writes the result.
//     The second group only works when the load per part (metadata int 6) is >= params.kg_min_tiles: the Q load and the
//     merge are per-segment costs that short parts do not amortize (R-055).  The choice is made for the whole launch -- a
//     CTA with one group keeps half the bytes in flight of a CTA with two and gets starved of DRAM bandwidth (R-056).
#pragma once

#include "mla_decode_sm86.cuh"

namespace mla_sm86 {

constexpr int kQRowBytes = 640;   // 512 B latent + 128 B rope
constexpr int kQChunks = 40;      // 16-byte chunks per row

template <typename T_, int KVQ_, int BM_, int NT_, int NS_, int MINB_ = 1, int KG_ = 1>
struct CfgQ {
    using T = T_;
    static constexpr int kKVQ = KVQ_;   // kKVFp8E4m3 or kKVInt8
    static constexpr int kBM = BM_;
    static constexpr int kGroups = BM_ / 16;
    static constexpr int kKG = KG_;                      // KV groups (warp sets splitting the tiles of a part)
    static constexpr int kGrpThreads = kGroups * 128;    // threads of one KV group
    static constexpr int kConsWarps = kGroups * 4 * KG_;
    static constexpr int kConsThreads = kConsWarps * 32;
    static constexpr int kNThreads = kConsThreads;
    static constexpr int kNt = NT_;
    static constexpr int kNT8 = NT_ / 8;
    static constexpr int kStages = NS_;
    static constexpr int kMinB = MINB_;
    static constexpr int kStageBytes = kNt * kQRowBytes + 128;   // Nt rows + per-token scales (4*Nt <= 128 B)
    static constexpr int kRingBytes = NS_ * kStageBytes;                    // per KV group
    static constexpr int kSBytes = KG_ * kGroups * 4 * kNT8 * 32 * 16;
    static constexpr int kMergeBytes = KG_ > 1 ? 4 * 68 * 32 * 4 : 0;       // KV group 1: acc_o (64) + m0 m1 l0 l1 per lane
    static constexpr int kSmem = KG_ * kRingBytes + kSBytes;
    static_assert(BM_ == 16 || BM_ == 32, "BM must be 16 or 32");
    static_assert(KG_ == 1 || (KG_ == 2 && BM_ == 16), "KV groups: KG 2 needs BM 16");
    static_assert(KG_ * kRingBytes >= kMergeBytes, "the merge buffer reuses the ring");
    static_assert(NT_ == 16 || NT_ == 32, "Nt must be 16 or 32");
    static_assert(NS_ >= 2 && NS_ <= 8, "need 2..8 stages");
};

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

__device__ __forceinline__ void cp_async_4(uint32_t dst, const void *src, bool pred) {
    const int sz = pred ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(sz) : "memory");
}

__device__ __forceinline__ uint2 ldg_nc_u64(const void *p) {
    uint2 v;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

// e4m3 bytes in the HIGH byte of each 16-bit half of q -> two bf16/fp16 values (exact).  The sign stays at bit 15, the 7
// magnitude bits are moved next to the exponent field; the result is scaled by 2^(bias difference) with an exact FMA.
template <typename T> struct Dq;
template <> struct Dq<__nv_bfloat16> {
    static __device__ __forceinline__ uint32_t e4m3_hi(uint32_t q) {
        const uint32_t r = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 4);   // bf16 pattern = value * 2^-120
        uint32_t d;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(r), "r"(0x7B807B80u), "r"(0x80008000u));   // * 2^120 + (-0)
        return d;
    }
};
template <> struct Dq<__half> {
    static __device__ __forceinline__ uint32_t e4m3_hi(uint32_t q) {
        const uint32_t r = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 1);   // fp16 pattern = value * 2^-8
        uint32_t d;
        asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(r), "r"(0x5C005C00u), "r"(0x80008000u));   // * 256 + (-0)
        return d;
    }
};

// 4 bytes q = [b3 b2 b1 b0]  ->  hi_pair = (b1, b3),  lo_pair = (b0, b2)  as bf16x2 / half2 (lo element first)
template <typename T, int KVQ>
__device__ __forceinline__ void dq_pairs(uint32_t q, uint32_t &hi_pair, uint32_t &lo_pair) {
    if constexpr (KVQ == kKVFp8E4m3) {
        hi_pair = Dq<T>::e4m3_hi(q);
        lo_pair = Dq<T>::e4m3_hi(q << 8);
    } else {
        // int8: byte ^ 0x80 is the offset-binary value u = v + 128; float(0x4B000000 | u) = 2^23 + u  =>  v = f - (2^23 + 128)
        const uint32_t x = q ^ 0x80808080u;
        const uint32_t m = 0x4B000000u;
        const float c = 8388736.0f;
        const float f0 = __uint_as_float(prmt_b32(x, m, 0x7540u)) - c;
        const float f1 = __uint_as_float(prmt_b32(x, m, 0x7541u)) - c;
        const float f2 = __uint_as_float(prmt_b32(x, m, 0x7542u)) - c;
        const float f3 = __uint_as_float(prmt_b32(x, m, 0x7543u)) - c;
        hi_pair = ElemTraits<T>::pack(f1, f3);
        lo_pair = ElemTraits<T>::pack(f0, f2);
    }
}

// ---- tile copy: 16 rows x 40 chunks (XOR-swizzled) + 16 per-token scales (zero-filled beyond the sequence end)
template <int kNt, int NISS>
__device__ __forceinline__ void issue_tile_copy_q(uint32_t dst, const uint8_t *src, int64_t row_stride, const float *scale_src, int tok0,
                                                  int seqlen_k, int iss_tid) {
#pragma unroll
    for (int it = 0; it < (kNt * kQChunks + NISS - 1) / NISS; ++it) {
        const int idx = iss_tid + it * NISS;
        if ((kNt * kQChunks) % NISS == 0 || idx < kNt * kQChunks) {
            const int tl = idx / kQChunks;
            const int ch = idx - tl * kQChunks;
            const bool valid = (tok0 + tl) < seqlen_k;
            cp_async_16(dst + tl * kQRowBytes + ((ch ^ (tl & 7)) << 4), src + (int64_t)tl * row_stride + ch * 16, valid);
        }
    }
    if (iss_tid < kNt) cp_async_4(dst + kNt * kQRowBytes + 4 * iss_tid, scale_src + iss_tid, (tok0 + iss_tid) < seqlen_k);
}

// ---- QK partial for one Nt-token tile
template <typename CQ, typename ET>
__device__ __forceinline__ void qk_stage_q(const uint32_t sbase, const uint32_t (&kqq_off)[4], const uint32_t rope_off, const uint32_t (&qf)[9][4],
                                           const float (&sc)[CQ::kNT8][2], float (&sacc)[CQ::kNT8][4]) {
    using T = typename CQ::T;
    constexpr int NT8 = CQ::kNT8;
    float lat[NT8][4], rope[NT8][4];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) { lat[nt][e] = 0.f; rope[nt][e] = 0.f; }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
#pragma unroll
        for (int npp = 0; npp < NT8 / 2; ++npp) {
            uint32_t w[4];
            ldsm_x4(w, sbase + kqq_off[u] + npp * 16 * kQRowBytes);   // w[mi]: n-tile 2*npp + (mi&1), k-step 2u + (mi>>1)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                uint32_t b1, b0;
                dq_pairs<T, CQ::kKVQ>(w[e], b1, b0);
                ET::mma(lat[2 * npp + (e & 1)], qf[2 * u + (e >> 1)], b0, b1);
            }
        }
    }
#pragma unroll
    for (int npp = 0; npp < NT8 / 2; ++npp) {
        uint32_t b[4];
        ldsm_x4(b, sbase + rope_off + npp * 16 * kQRowBytes);
        ET::mma(rope[2 * npp], qf[8], b[0], b[1]);
        ET::mma(rope[2 * npp + 1], qf[8], b[2], b[3]);
    }
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) sacc[nt][e] = fmaf(sc[nt][e & 1], lat[nt][e], rope[nt][e]);
}

// ---- PV for one Nt-token tile: O[:, 128 columns] += (P*scale) V ; n-tile 2P = even dims 16P+2g, 2P+1 = odd dims
template <typename CQ, typename ET>
__device__ __forceinline__ void pv_stage_q(const uint32_t (&pa)[CQ::kNt / 16][4], const uint32_t sbase, const uint32_t (&vqq_off)[4], float (&acc_o)[16][4]) {
    using T = typename CQ::T;
#pragma unroll
    for (int kc = 0; kc < CQ::kNt / 16; ++kc) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            uint32_t w[4];
            ldsm_x4_trans(w, sbase + vqq_off[u] + kc * 16 * kQRowBytes);   // w[mi]: token half (mi&1) of this 16-token slab, 16-dim group 2u + (mi>>1)
#pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
                uint32_t xb0, xb1, yb0, yb1;
                dq_pairs<T, CQ::kKVQ>(w[2 * pp], yb0, xb0);
                dq_pairs<T, CQ::kKVQ>(w[2 * pp + 1], yb1, xb1);
                ET::mma(acc_o[2 * (2 * u + pp)], pa[kc], xb0, xb1);
                ET::mma(acc_o[2 * (2 * u + pp) + 1], pa[kc], yb0, yb1);
            }
        }
    }
}

template <typename CQ, typename ET>
__device__ __forceinline__ void compute_tile_q(
    const uint32_t sbase, const int tok0, float4 *sS_mine, const float4 *sS_grp, const int xbar, const uint32_t (&kqq_off)[4], const uint32_t rope_off,
    const uint32_t (&vqq_off)[4], const int t, const float sl2, const uint32_t (&qf)[9][4], float (&acc_o)[16][4], float &m0, float &m1, float &l0,
    float &l1, const int lim0, const int lim1, const int mask_min) {
    constexpr int NT8 = CQ::kNT8;
    // per-token scales of this lane's S columns: tokens 8*nt + 2t + {0,1}
    float sc[NT8][2];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
        const float2 sv = lds_f2(sbase + CQ::kNt * kQRowBytes + 4 * (8 * nt + 2 * t));
        sc[nt][0] = sv.x; sc[nt][1] = sv.y;
    }
    float sacc[NT8][4];
    qk_stage_q<CQ, ET>(sbase, kqq_off, rope_off, qf, sc, sacc);
    exch_store<CQ>(sS_mine, sacc);
    bar_sync(xbar, 128);
    float s[NT8][4];
    exch_sum<CQ>(sS_grp, s);
    mask_stage<CQ>(s, tok0, t, lim0, lim1, mask_min);
    uint32_t pa[CQ::kNt / 16][4];
    softmax_stage<CQ, ET, true>(s, sl2, m0, m1, l0, l1, acc_o, pa, sc);
    pv_stage_q<CQ, ET>(pa, sbase, vqq_off, acc_o);
}

template <typename CQ>
__global__ void __launch_bounds__(CQ::kNThreads, CQ::kMinB)
mla_decode_q_kernel(const __grid_constant__ Flash_fwd_mla_params params) {
    using C = CQ;
    using T = typename C::T;
    using ET = ElemTraits<T>;
    constexpr int NT = C::kNt;
    constexpr int NT8 = C::kNT8;
    constexpr int NS = C::kStages;

    extern __shared__ __align__(128) unsigned char smem_buf[];
    float4 *sS_all = reinterpret_cast<float4 *>(smem_buf + C::kKG * C::kRingBytes);

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int grp = (warp >> 2) % C::kGroups, j = warp & 3;
    const int kg = warp / (4 * C::kGroups);              // KV group: takes tiles kg, kg + KG, kg + 2 KG, ... of a part
    const int gtid = tid - kg * C::kGrpThreads;          // thread index inside the KV group (cp.async issue)
    const int xbar = 1 + kg * C::kGroups + grp;          // named barrier of the S exchange (128 threads)
    const bool writer = kg == 0;                         // KV group 0 holds the merged state and writes O / LSE
    const uint32_t ring_u32 = smem_u32(smem_buf) + kg * C::kRingBytes;
    const int g = lane >> 2, t = lane & 3;
    const int m_block = blockIdx.x, bidh = blockIdx.y, part = blockIdx.z;

    const int *meta = params.tile_scheduler_metadata_ptr + part * TileSchedulerMetaDataSize;
    const int4 md = __ldg(reinterpret_cast<const int4 *>(meta));
    const int4 md2 = __ldg(reinterpret_cast<const int4 *>(meta + 4));
    const int begin_n_split_idx = md2.x;
    const int begin_idx = md.x, begin_seqlen = md.y, end_idx = md.z, end_seqlen = md.w;
    if (begin_idx >= params.b) return;
    // KV groups at work (uniform over the launch): two only when every part carries at least kg_min_tiles units
    const int kgs = (C::kKG == 2 && md2.z >= params.kg_min_tiles) ? 2 : 1;

    const int M = params.seqlen_q;
    const int row0 = m_block * C::kBM + grp * 16 + g;
    const int row1 = row0 + 8;
    const bool rv0 = row0 < M, rv1 = row1 < M;
    const int s_orig = M / params.ngroups;
    const int ps_shift = 31 - __clz(params.page_block_size);
    const int ps_mask = params.page_block_size - 1;
    const float sl2 = params.scale_softmax_log2;

    // ldmatrix lane constants (see the file comment)
    const int mi = lane >> 3, ri = lane & 7;
    uint32_t kqq_off[4], vqq_off[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        kqq_off[u] = (8 * (mi & 1) + ri) * kQRowBytes + ((((4 * (2 * u + (mi >> 1))) + j) ^ ri) << 4);   // fp8 K: row = token, chunk = k-step
        vqq_off[u] = (8 * (mi & 1) + ri) * kQRowBytes + ((((8 * j) + 2 * u + (mi >> 1)) ^ ri) << 4);     // fp8 V (trans): chunk = 16-dim group
    }
    const uint32_t rope_off = (8 * (mi >> 1) + ri) * kQRowBytes + (((32 + 2 * j + (mi & 1)) ^ ri) << 4);

    float4 *sS_mine_base = sS_all + (((kg * C::kGroups + grp) * 4 + j) * NT8) * 32 + lane;
    const float4 *sS_grp_base = sS_all + ((kg * C::kGroups + grp) * 4 * NT8) * 32 + lane;

    const uint8_t *kbase = reinterpret_cast<const uint8_t *>(params.k_ptr) + (int64_t)bidh * params.k_head_stride;
    const int64_t k_row_stride = params.k_row_stride;
    const int64_t k_batch_stride = params.k_batch_stride;
    const float *scale_base = params.kv_scale_ptr;

#pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int seqlen_k = (batch_id == begin_idx) ? md2.y : __ldg(params.cu_seqlens_k + batch_id);
        const int tok_begin = (batch_id == begin_idx) ? begin_seqlen : 0;
        const int tok_end = (batch_id == end_idx) ? min(end_seqlen, seqlen_k) : seqlen_k;
        const int n_split_idx = (batch_id == begin_idx) ? begin_n_split_idx : 0;
        const bool split = !(tok_begin == 0 && tok_end == seqlen_k);
        const int n_tiles = (tok_end - tok_begin + NT - 1) / NT;
        const int n_my = (kg < kgs && n_tiles > kg) ? (n_tiles - kg + kgs - 1) / kgs : 0;   // this KV group's tiles
        auto tile_tok = [&](int i) { return tok_begin + (kg + i * kgs) * NT; };              // first token of local tile i

        const int *bt = params.block_table + (int64_t)batch_id * params.block_table_batch_stride;
        const int bt_last = (int)params.block_table_batch_stride - 1;
        auto load_pg = [&](int i) { return __ldg(bt + min(tile_tok(i) >> ps_shift, bt_last)); };

        if (batch_id > begin_idx) __syncthreads();   // previous segment is done with the smem ring

        // ---- Q -> registers; fp8 k-steps (i < 8) use the dim permutation (0,2 | 1,3) of the converted K fragments ----
        uint32_t qf[9][4];
        if (C::kKG == 1 || n_my > 0) {
            const T *qb = reinterpret_cast<const T *>(params.q_ptr) + (int64_t)batch_id * params.q_batch_stride +
                          (int64_t)bidh * params.q_head_stride;
            const T *q0 = qb + (int64_t)row0 * params.q_row_stride;
            const T *q1 = qb + (int64_t)row1 * params.q_row_stride;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int e0 = 16 * (4 * i + j) + 4 * t;   // 4 consecutive dims of this lane
                const uint2 a = rv0 ? ldg_nc_u64(q0 + e0) : make_uint2(0u, 0u);
                const uint2 b = rv1 ? ldg_nc_u64(q1 + e0) : make_uint2(0u, 0u);
                qf[i][0] = prmt_b32(a.x, a.y, 0x5410u);   // (e0, e2)
                qf[i][1] = prmt_b32(b.x, b.y, 0x5410u);
                qf[i][2] = prmt_b32(a.x, a.y, 0x7632u);   // (e1, e3)
                qf[i][3] = prmt_b32(b.x, b.y, 0x7632u);
            }
            const T *r0 = q0 + 512 + 16 * j + 2 * t;
            const T *r1 = q1 + 512 + 16 * j + 2 * t;
            qf[8][0] = rv0 ? ldg_nc_u32(r0) : 0u;
            qf[8][1] = rv1 ? ldg_nc_u32(r1) : 0u;
            qf[8][2] = rv0 ? ldg_nc_u32(r0 + 8) : 0u;
            qf[8][3] = rv1 ? ldg_nc_u32(r1 + 8) : 0u;
        }

        int pgv[NS];
#pragma unroll
        for (int s = 0; s < NS; ++s) pgv[s] = load_pg(s);
#pragma unroll
        for (int s = 0; s < NS - 1; ++s) {
            if (s < n_my) {
                const int tok0 = tile_tok(s);
                const uint8_t *src = kbase + (int64_t)pgv[s] * k_batch_stride + (int64_t)(tok0 & ps_mask) * k_row_stride;
                const float *ssrc = scale_base + (int64_t)pgv[s] * params.page_block_size + (tok0 & ps_mask);
                issue_tile_copy_q<NT, C::kGrpThreads>(ring_u32 + s * C::kStageBytes, src, k_row_stride, ssrc, tok0, seqlen_k, gtid);
            }
            cp_async_commit();
        }
        int pg_pending = pgv[NS - 1];

        int lim0 = tok_end, lim1 = tok_end;
        int mask_min = tok_end;
        if (params.is_causal) {
            lim0 = min(tok_end, seqlen_k - (s_orig - 1 - row0 / params.ngroups));
            lim1 = min(tok_end, seqlen_k - (s_orig - 1 - row1 / params.ngroups));
            mask_min = min(tok_end, seqlen_k - (s_orig - 1));
        }

        float acc_o[16][4];
#pragma unroll
        for (int i = 0; i < 16; ++i) { acc_o[i][0] = acc_o[i][1] = acc_o[i][2] = acc_o[i][3] = 0.f; }
        float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

        int rd_slot = 0, wr_slot = NS - 1;
#pragma unroll 1
        for (int n = 0; n < n_my; ++n) {
            const int tok0 = tile_tok(n);
            cp_async_wait<NS - 2>();
            if constexpr (C::kKG == 1) __syncthreads();
            else bar_sync(3 + kg, C::kGrpThreads);   // this KV group's ring only: the groups run decoupled
            {
                const int pg_cur = pg_pending;
                pg_pending = load_pg(n + NS);
                if (n + NS - 1 < n_my) {
                    const int tok_p = tile_tok(n + NS - 1);
                    const uint8_t *src = kbase + (int64_t)pg_cur * k_batch_stride + (int64_t)(tok_p & ps_mask) * k_row_stride;
                    const float *ssrc = scale_base + (int64_t)pg_cur * params.page_block_size + (tok_p & ps_mask);
                    issue_tile_copy_q<NT, C::kGrpThreads>(ring_u32 + wr_slot * C::kStageBytes, src, k_row_stride, ssrc, tok_p, seqlen_k, gtid);
                }
            }
            cp_async_commit();
            compute_tile_q<C, ET>(ring_u32 + rd_slot * C::kStageBytes, tok0, sS_mine_base, sS_grp_base, xbar, kqq_off, rope_off, vqq_off, t, sl2, qf,
                                  acc_o, m0, m1, l0, l1, lim0, lim1, mask_min);
            rd_slot = (rd_slot + 1 == NS) ? 0 : rd_slot + 1;
            wr_slot = (wr_slot + 1 == NS) ? 0 : wr_slot + 1;
        }

        if (C::kKG == 2 && kgs == 2) {   // CTA-uniform
            // merge KV group 1 into group 0 (same lane layout): O = O0 * 2^((m0-m) sl2) + O1 * 2^((m1-m) sl2), l likewise
            __syncthreads();   // both groups are done with their rings; the merge buffer reuses ring memory
            float *mb = reinterpret_cast<float *>(smem_buf) + j * (68 * 32) + lane;
            if (kg == 1) {
#pragma unroll
                for (int i = 0; i < 16; ++i)
#pragma unroll
                    for (int e = 0; e < 4; ++e) mb[(4 * i + e) * 32] = acc_o[i][e];
                mb[64 * 32] = m0;
                mb[65 * 32] = m1;
                mb[66 * 32] = l0;
                mb[67 * 32] = l1;
            }
            __syncthreads();
            if (kg == 0) {
                const float mo0 = mb[64 * 32], mo1 = mb[65 * 32];
                const float nm0 = fmaxf(m0, mo0), nm1 = fmaxf(m1, mo1);
                const float ms0 = (nm0 == -INFINITY) ? 0.f : nm0;
                const float ms1 = (nm1 == -INFINITY) ? 0.f : nm1;
                const float fa0 = ex2_approx((m0 - ms0) * sl2), fb0 = ex2_approx((mo0 - ms0) * sl2);
                const float fa1 = ex2_approx((m1 - ms1) * sl2), fb1 = ex2_approx((mo1 - ms1) * sl2);
                l0 = l0 * fa0 + mb[66 * 32] * fb0;
                l1 = l1 * fa1 + mb[67 * 32] * fb1;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    acc_o[i][0] = acc_o[i][0] * fa0 + mb[(4 * i + 0) * 32] * fb0;
                    acc_o[i][1] = acc_o[i][1] * fa0 + mb[(4 * i + 1) * 32] * fb0;
                    acc_o[i][2] = acc_o[i][2] * fa1 + mb[(4 * i + 2) * 32] * fb1;
                    acc_o[i][3] = acc_o[i][3] * fa1 + mb[(4 * i + 3) * 32] * fb1;
                }
                m0 = nm0;
                m1 = nm1;
            }
        }

        // ---- epilogue: lane owns dims 128j + 16P + 4t + {0..3} of rows g / g+8:  (X.c, Y.c, X.c', Y.c') = acc_o[2P], acc_o[2P+1] ----
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        const bool bad0 = (l0 == 0.f) || (l0 != l0), bad1 = (l1 == 0.f) || (l1 != l1);
        const float lse0 = bad0 ? (split ? -INFINITY : INFINITY) : m0 * params.scale_softmax + logf(l0);
        const float lse1 = bad1 ? (split ? -INFINITY : INFINITY) : m1 * params.scale_softmax + logf(l1);
        const float inv0 = bad0 ? 1.f : 1.f / l0;
        const float inv1 = bad1 ? 1.f : 1.f / l1;
        const int hh = params.h;
        if (!split && writer) {
            T *ob = reinterpret_cast<T *>(params.o_ptr) + (int64_t)batch_id * params.o_batch_stride + (int64_t)bidh * params.o_head_stride;
            T *o0 = ob + (int64_t)row0 * params.o_row_stride + 128 * j + 4 * t;
            T *o1 = ob + (int64_t)row1 * params.o_row_stride + 128 * j + 4 * t;
#pragma unroll
            for (int P = 0; P < 8; ++P) {
                if (rv0) {
                    uint2 v;
                    v.x = ET::pack(acc_o[2 * P][0] * inv0, acc_o[2 * P + 1][0] * inv0);
                    v.y = ET::pack(acc_o[2 * P][1] * inv0, acc_o[2 * P + 1][1] * inv0);
                    *reinterpret_cast<uint2 *>(o0 + 16 * P) = v;
                }
                if (rv1) {
                    uint2 v;
                    v.x = ET::pack(acc_o[2 * P][2] * inv1, acc_o[2 * P + 1][2] * inv1);
                    v.y = ET::pack(acc_o[2 * P][3] * inv1, acc_o[2 * P + 1][3] * inv1);
                    *reinterpret_cast<uint2 *>(o1 + 16 * P) = v;
                }
            }
            if (j == 0 && t == 0) {
                float *lp = reinterpret_cast<float *>(params.softmax_lse_ptr) + ((int64_t)batch_id * hh + bidh) * M;
                if (rv0) lp[row0] = lse0;
                if (rv1) lp[row1] = lse1;
            }
        } else if (split) {
            const int split_offset = __ldg(params.num_splits_ptr + batch_id);
            const int64_t slot = split_offset + n_split_idx;
            float *oa = reinterpret_cast<float *>(params.oaccum_ptr) + ((slot * hh + bidh) * M) * kDV;
            float *o0 = oa + (int64_t)row0 * kDV + 128 * j + 4 * t;
            float *o1 = oa + (int64_t)row1 * kDV + 128 * j + 4 * t;
#pragma unroll
            for (int P = 0; P < 8; ++P) {
                if (writer && rv0)
                    *reinterpret_cast<float4 *>(o0 + 16 * P) =
                        make_float4(acc_o[2 * P][0] * inv0, acc_o[2 * P + 1][0] * inv0, acc_o[2 * P][1] * inv0, acc_o[2 * P + 1][1] * inv0);
                if (writer && rv1)
                    *reinterpret_cast<float4 *>(o1 + 16 * P) =
                        make_float4(acc_o[2 * P][2] * inv1, acc_o[2 * P + 1][2] * inv1, acc_o[2 * P][3] * inv1, acc_o[2 * P + 1][3] * inv1);
            }
            if (writer && j == 0 && t == 0) {
                float *la = reinterpret_cast<float *>(params.softmax_lseaccum_ptr) + (slot * hh + bidh) * M;
                if (rv0) la[row0] = lse0;
                if (rv1) la[row1] = lse1;
            }
            const int nsp = __ldg(params.num_splits_ptr + batch_id + 1) - split_offset;
            if (params.combine_counters != nullptr && nsp <= params.fuse_max_splits) {   // CTA-uniform
                __shared__ int s_last;
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    int *cnt = params.combine_counters + ((int64_t)batch_id * hh + bidh) * ((M + 15) / 16) + m_block;
                    const int prev = atomicAdd(cnt, 1);
                    const int last = (prev == nsp - 1);
                    if (last) *cnt = 0;
                    s_last = last;
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    fused_combine<C, ET>(params, batch_id, bidh, m_block, nsp, split_offset);
                }
                __syncthreads();
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------------------
template <typename CQ>
void launch_main_q(Flash_fwd_mla_params &p, cudaStream_t stream) {
    auto kernel = &mla_decode_q_kernel<CQ>;
    static bool attr_done[16] = {false};
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    if (dev < 16 && !attr_done[dev]) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, CQ::kSmem));
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        attr_done[dev] = true;
    }
    const int num_m_blocks = (p.seqlen_q + CQ::kBM - 1) / CQ::kBM;
    kernel<<<dim3(num_m_blocks, p.h, p.num_sm_parts), CQ::kNThreads, CQ::kSmem, stream>>>(p);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template <typename T, int KVQ>
void mla_sm86_run_q_impl(Flash_fwd_mla_params &p, cudaStream_t stream) {
    PlanQ plan = pick_plan_q(p.seqlen_q);
    if (plan.nt > p.page_block_size) { plan.nt = p.page_block_size; if (plan.ns < 3) plan.ns = 3; }   // tile within one page
    bool launched = false;
#define MLA_SM86_QCASE(BM_, NT_, NS_, MB_, KG_)                                                       \
    if (plan.bm == BM_ && plan.nt == NT_ && plan.ns == NS_ && plan.minb == MB_ && plan.kg == KG_) {   \
        launch_main_q<CfgQ<T, KVQ, BM_, NT_, NS_, MB_, KG_>>(p, stream);                              \
        launched = true;                                                                              \
    }
    MLA_SM86_QCASE(16, 16, 3, 1, 1)
    MLA_SM86_QCASE(16, 16, 4, 1, 1)
    MLA_SM86_QCASE(16, 16, 5, 1, 1)
    MLA_SM86_QCASE(16, 16, 3, 2, 1)
    MLA_SM86_QCASE(16, 32, 2, 2, 1)
    MLA_SM86_QCASE(16, 32, 2, 1, 1)
    MLA_SM86_QCASE(16, 32, 3, 1, 1)
    MLA_SM86_QCASE(16, 32, 4, 1, 1)
    MLA_SM86_QCASE(16, 16, 2, 1, 2)   // two KV groups (8 warps), 1 CTA/SM
    MLA_SM86_QCASE(16, 16, 3, 1, 2)
    MLA_SM86_QCASE(16, 16, 4, 1, 2)
    MLA_SM86_QCASE(16, 32, 2, 1, 2)
    MLA_SM86_QCASE(32, 16, 3, 1, 1)
    MLA_SM86_QCASE(32, 16, 4, 1, 1)
    MLA_SM86_QCASE(32, 32, 2, 1, 1)
    MLA_SM86_QCASE(32, 32, 3, 1, 1)
#undef MLA_SM86_QCASE
    FLASH_ASSERT(launched);
}

template <typename T>
void mla_sm86_run_q(Flash_fwd_mla_params &p, cudaStream_t stream) {
    FLASH_ASSERT(p.d == kD && p.d_v == kDV);
    FLASH_ASSERT(p.k_ptr == p.v_ptr);
    FLASH_ASSERT(p.page_block_size == 16 || p.page_block_size == 32 || p.page_block_size == 64);
    FLASH_ASSERT(p.kv_scale_ptr != nullptr);
    FLASH_ASSERT(p.k_row_stride == kQRowBytes && (p.k_batch_stride % 128) == 0);
    if (p.combine_counters == nullptr) p.fuse_max_splits = 0;
    if (p.fuse_max_splits > kFuseCap) p.fuse_max_splits = kFuseCap;
    p.kg_min_tiles = kg_min_tiles_default();
    if (p.kv_dtype == kKVFp8E4m3) mla_sm86_run_q_impl<T, kKVFp8E4m3>(p, stream);
    else if (p.kv_dtype == kKVInt8) mla_sm86_run_q_impl<T, kKVInt8>(p, stream);
    else FLASH_ASSERT(false);
    launch_combine<T>(p, stream);
}

}  // namespace mla_sm86
