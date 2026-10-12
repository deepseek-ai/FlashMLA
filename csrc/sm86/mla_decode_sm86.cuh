// FlashMLA decode kernels for sm_86 (RTX 3090 class).
//
//  * CTA = BM rows (heads x s_q) x a KV token range.  BM=16 -> 4 consumer warps, BM=32 -> 8 (2 M-groups x 4 warps).
//  * Warp j of a group holds its slice of Q (16 rows x 9 k16-steps = 36 regs) in registers for the whole
//    segment; QK^T is split-K across the 4 warps of a group (k-steps 4i+j, i = 0..8).  Partial S goes through
//    a small smem exchange; every warp sums the 4 partials in the same order (bit-identical S), runs the
//    online softmax redundantly, converts P straight into MMA A fragments and computes PV for its own
//    128 V columns.
//  * KV tiles (Nt tokens x 576 bf16/fp16) stream through a multistage smem ring (XOR-swizzled rows, 1152 B/row).
//      v2a (kSpec = false): all warps issue cp.async, one CTA barrier per tile.
//      v2b (kSpec = true) : one producer warp issues cp.async and signals tile arrival with
//                           cp.async.mbarrier.arrive; consumers wait on full/empty mbarriers, so the M-groups
//                           run decoupled and the producer prefetches across batch-segment boundaries.
//  * Persistent scheduler: blockIdx.z indexes tile_scheduler_metadata (same format as the legacy kernels).
//  * Split-KV partials (fp32 O + LSE) are merged by a separate combine kernel.
#pragma once

#include <cstdio>
#include <cstdlib>

#include "sm86_ptx.cuh"
#include "../flash_mla.h"
#include "../static_switch.h"
#include "mla_sm86.h"

namespace mla_sm86 {

constexpr int kD = 576;
constexpr int kDV = 512;
constexpr int kRowBytes = kD * 2;               // 1152 B per token row (native dtype)
constexpr int kChunksPerRow = kRowBytes / 16;   // 72 16-byte chunks
constexpr int kMaxSplits = 256;

template <typename T_, int BM_, int NT_, int NS_, int MINB_ = 1, bool SPEC_ = false>
struct Cfg {
    using T = T_;
    static constexpr int kBM = BM_;
    static constexpr int kGroups = BM_ / 16;
    static constexpr int kConsWarps = kGroups * 4;
    static constexpr int kConsThreads = kConsWarps * 32;
    static constexpr bool kSpec = SPEC_;
    static constexpr int kNThreads = kConsThreads + (SPEC_ ? 32 : 0);
    static constexpr int kNt = NT_;
    static constexpr int kNT8 = NT_ / 8;
    static constexpr int kStages = NS_;
    static constexpr int kMinB = MINB_;
    static constexpr int kStageBytes = NT_ * kRowBytes;
    static constexpr int kRingBytes = NS_ * kStageBytes;
    static constexpr int kSBytes = kGroups * 4 * kNT8 * 32 * 16;   // partial-S exchange buffer (one)
    static constexpr int kSBufs = SPEC_ ? 2 : 1;
    static constexpr int kBarOff = kRingBytes + kSBufs * kSBytes;
    static constexpr int kSmem = kBarOff + (SPEC_ ? 2 * NS_ * 8 : 0);
    static_assert(BM_ == 16 || BM_ == 32, "BM must be 16 or 32");
    static_assert(NT_ == 16 || NT_ == 32, "Nt must be 16 or 32");
    static_assert(NS_ >= 2 && NS_ <= 8, "need 2..8 stages");
};

// ---------------------------------------------------------------------------------------------------------
// KV tile copy: Nt rows x 72 chunks of 16 B into a swizzled smem stage.  `NISS` threads cooperate.
//   smem byte offset of (token tl, chunk ch) = tl*1152 + ((ch ^ (tl & 7)) << 4)
// ---------------------------------------------------------------------------------------------------------
template <int NT, int NISS, typename T>
__device__ __forceinline__ void issue_tile_copy(uint32_t dst, const T *src, int64_t k_row_stride, int tok0, int seqlen_k, int iss_tid) {
#pragma unroll
    for (int it = 0; it < (NT * kChunksPerRow + NISS - 1) / NISS; ++it) {
        const int idx = iss_tid + it * NISS;
        if ((NT * kChunksPerRow) % NISS == 0 || idx < NT * kChunksPerRow) {
            const int tl = idx / kChunksPerRow;
            const int ch = idx - tl * kChunksPerRow;
            const bool valid = (tok0 + tl) < seqlen_k;
            cp_async_16(dst + tl * kRowBytes + ((ch ^ (tl & 7)) << 4), src + (int64_t)tl * k_row_stride + ch * 8, valid);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// One KV tile of one consumer warp: S partial -> exchange -> softmax -> PV.
// ---------------------------------------------------------------------------------------------------------
template <typename C, typename ET>
__device__ __forceinline__ void compute_tile(
    const uint32_t sbase, const int tok0, float4 *sS_mine, const float4 *sS_grp, const int grp,
    const uint32_t kq_off, const uint32_t (&vq_off)[4], const int t, const float sl2,
    const uint32_t (&qf)[9][4], float (&acc_o)[16][4], float &m0, float &m1, float &l0, float &l1,
    const int lim0, const int lim1, const int mask_min) {
    constexpr int NT = C::kNt;
    constexpr int NT8 = C::kNT8;

    // ---- S partial = Q[:, slice j] * K^T[slice j, tile] ----
    float sacc[NT8][4];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) { sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.f; }
#pragma unroll
    for (int i = 0; i < 9; ++i) {
#pragma unroll
        for (int np = 0; np < NT / 16; ++np) {
            uint32_t b[4];
            ldsm_x4(b, sbase + kq_off + np * 16 * kRowBytes + i * 128);
            ET::mma(sacc[2 * np], qf[i], b[0], b[1]);
            ET::mma(sacc[2 * np + 1], qf[i], b[2], b[3]);
        }
    }
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt)
        sS_mine[nt * 32] = make_float4(sacc[nt][0], sacc[nt][1], sacc[nt][2], sacc[nt][3]);
    bar_sync(1 + grp, 128);

    // ---- S = sum of the 4 partials (fixed order => bit-identical in all 4 warps) ----
    float s[NT8][4];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
        const float4 p0 = sS_grp[(0 * NT8 + nt) * 32];
        const float4 p1 = sS_grp[(1 * NT8 + nt) * 32];
        const float4 p2 = sS_grp[(2 * NT8 + nt) * 32];
        const float4 p3 = sS_grp[(3 * NT8 + nt) * 32];
        s[nt][0] = ((p0.x + p1.x) + p2.x) + p3.x;
        s[nt][1] = ((p0.y + p1.y) + p2.y) + p3.y;
        s[nt][2] = ((p0.z + p1.z) + p2.z) + p3.z;
        s[nt][3] = ((p0.w + p1.w) + p2.w) + p3.w;
    }

    // ---- masking (only tiles that cross the sequence end / causal diagonal) ----
    if (tok0 + NT > mask_min) {
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) {
            const int c = tok0 + nt * 8 + 2 * t;
            if (c >= lim0) s[nt][0] = -INFINITY;
            if (c + 1 >= lim0) s[nt][1] = -INFINITY;
            if (c >= lim1) s[nt][2] = -INFINITY;
            if (c + 1 >= lim1) s[nt][3] = -INFINITY;
        }
    }

    // ---- online softmax (log2 domain) ----
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
        mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
        mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float nm0 = fmaxf(m0, mx0), nm1 = fmaxf(m1, mx1);
    const float ms0 = (nm0 == -INFINITY) ? 0.f : nm0;
    const float ms1 = (nm1 == -INFINITY) ? 0.f : nm1;
    const float sc0 = ex2_approx((m0 - ms0) * sl2);
    const float sc1 = ex2_approx((m1 - ms1) * sl2);
    m0 = nm0; m1 = nm1;
    const float off0 = ms0 * sl2, off1 = ms1 * sl2;
    float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
        s[nt][0] = ex2_approx(fmaf(s[nt][0], sl2, -off0));
        s[nt][1] = ex2_approx(fmaf(s[nt][1], sl2, -off0));
        s[nt][2] = ex2_approx(fmaf(s[nt][2], sl2, -off1));
        s[nt][3] = ex2_approx(fmaf(s[nt][3], sl2, -off1));
        ps0 += s[nt][0] + s[nt][1];
        ps1 += s[nt][2] + s[nt][3];
    }
    l0 = l0 * sc0 + ps0;
    l1 = l1 * sc1 + ps1;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        acc_o[i][0] *= sc0; acc_o[i][1] *= sc0;
        acc_o[i][2] *= sc1; acc_o[i][3] *= sc1;
    }

    // ---- O[:, V cols 128j..128j+127] += P * V ----
#pragma unroll
    for (int kc = 0; kc < NT / 16; ++kc) {
        uint32_t pa[4];
        pa[0] = ET::pack(s[2 * kc][0], s[2 * kc][1]);
        pa[1] = ET::pack(s[2 * kc][2], s[2 * kc][3]);
        pa[2] = ET::pack(s[2 * kc + 1][0], s[2 * kc + 1][1]);
        pa[3] = ET::pack(s[2 * kc + 1][2], s[2 * kc + 1][3]);
#pragma unroll
        for (int np = 0; np < 8; ++np) {
            uint32_t b[4];
            ldsm_x4_trans(b, sbase + vq_off[np & 3] + (np >> 2) * 128 + kc * 16 * kRowBytes);
            ET::mma(acc_o[2 * np], pa, b[0], b[1]);
            ET::mma(acc_o[2 * np + 1], pa, b[2], b[3]);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------------------------------------
template <typename C>
__global__ void __launch_bounds__(C::kNThreads, C::kMinB)
mla_decode_kernel(const __grid_constant__ Flash_fwd_mla_params params) {
    using T = typename C::T;
    using ET = ElemTraits<T>;
    constexpr int NT = C::kNt;
    constexpr int NT8 = C::kNT8;
    constexpr int NS = C::kStages;
    constexpr bool SPEC = C::kSpec;

    extern __shared__ __align__(128) unsigned char smem_buf[];
    const uint32_t ring_u32 = smem_u32(smem_buf);
    float4 *sS_all = reinterpret_cast<float4 *>(smem_buf + C::kRingBytes);
    const uint32_t bar_u32 = smem_u32(smem_buf + C::kBarOff);   // full[NS], empty[NS] (v2b only)

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const bool is_producer = SPEC && (warp == C::kConsWarps);
    const int grp = (warp >> 2) % C::kGroups, j = warp & 3;
    const int g = lane >> 2, t = lane & 3;
    const int m_block = blockIdx.x, bidh = blockIdx.y, part = blockIdx.z;

    const int *meta = params.tile_scheduler_metadata_ptr + part * TileSchedulerMetaDataSize;
    const int4 md = __ldg(reinterpret_cast<const int4 *>(meta));
    const int begin_n_split_idx = __ldg(meta + 4);
    const int begin_idx = md.x, begin_seqlen = md.y, end_idx = md.z, end_seqlen = md.w;
    if (begin_idx >= params.b) return;

    if constexpr (SPEC) {
        if (tid == 0) {
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                mbar_init(bar_u32 + s * 8, 32);                       // full: 32 producer-thread arrivals
                mbar_init(bar_u32 + (NS + s) * 8, C::kConsWarps);     // empty: one arrival per consumer warp
            }
        }
        __syncthreads();
    }

    const int M = params.seqlen_q;   // rows = s_q * ngroups
    const int row0 = m_block * C::kBM + grp * 16 + g;
    const int row1 = row0 + 8;
    const bool rv0 = row0 < M, rv1 = row1 < M;
    const int s_orig = M / params.ngroups;
    const int ps_shift = 31 - __clz(params.page_block_size);
    const int ps_mask = params.page_block_size - 1;
    const float sl2 = params.scale_softmax_log2;

    // ldmatrix lane constants: K tile  B-frag(QK) = ldmatrix.x4 at row (8*(mi>>1)+ri), chunk 8i+2j+(mi&1)
    //                                  B-frag(PV) = ldmatrix.x4.trans at row (8*(mi&1)+ri), chunk 16j+2np+(mi>>1)
    const int mi = lane >> 3, ri = lane & 7;
    const uint32_t kq_off = (8 * (mi >> 1) + ri) * kRowBytes + ((((2 * j) + (mi & 1)) ^ ri) << 4);
    uint32_t vq_off[4];
#pragma unroll
    for (int p = 0; p < 4; ++p)
        vq_off[p] = (8 * (mi & 1) + ri) * kRowBytes + (((2 * p + (mi >> 1)) ^ ri) << 4) + 2 * j * 128;

    float4 *sS_mine_base = sS_all + ((grp * 4 + j) * NT8) * 32 + lane;
    const float4 *sS_grp_base = sS_all + (grp * 4 * NT8) * 32 + lane;
    constexpr int kSBufElems = C::kSBytes / 16;   // float4 elements per S buffer

    const T *kbase = reinterpret_cast<const T *>(params.k_ptr) + (int64_t)bidh * params.k_head_stride;
    const int64_t k_row_stride = params.k_row_stride;
    const int64_t k_batch_stride = params.k_batch_stride;

    // v2b: ring position (global across segments): slot index and fill-round
    int gslot = 0, ground = 0;
    int sbuf = 0;   // S exchange buffer parity (v2b)

#pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int seqlen_k = __ldg(params.cu_seqlens_k + batch_id);
        const int tok_begin = (batch_id == begin_idx) ? begin_seqlen : 0;
        const int tok_end = (batch_id == end_idx) ? min(end_seqlen, seqlen_k) : seqlen_k;
        const int n_split_idx = (batch_id == begin_idx) ? begin_n_split_idx : 0;
        const bool split = !(tok_begin == 0 && tok_end == seqlen_k);
        const int n_tiles = (tok_end - tok_begin + NT - 1) / NT;

        const int *bt = params.block_table + (int64_t)batch_id * params.block_table_batch_stride;
        const int bt_last = (int)params.block_table_batch_stride - 1;
        // page id of tile n (index clamped to the row so speculative look-ahead loads never leave the table)
        auto load_pg = [&](int n) { return __ldg(bt + min((tok_begin + n * NT) >> ps_shift, bt_last)); };

        // ================================ v2b producer warp ================================
        if constexpr (SPEC) {
            if (is_producer) {
                int pg_next = load_pg(0);
#pragma unroll 1
                for (int n = 0; n < n_tiles; ++n) {
                    const int pg_cur = pg_next;
                    pg_next = load_pg(n + 1);   // consumed one iteration later
                    if (ground > 0) mbar_wait(bar_u32 + (NS + gslot) * 8, (ground - 1) & 1);   // slot released by all consumers
                    const int tok0 = tok_begin + n * NT;
                    const T *src = kbase + (int64_t)pg_cur * k_batch_stride + (int64_t)(tok0 & ps_mask) * k_row_stride;
                    issue_tile_copy<NT, 32>(ring_u32 + gslot * C::kStageBytes, src, k_row_stride, tok0, seqlen_k, lane);
                    cp_async_mbar_arrive_noinc(bar_u32 + gslot * 8);
                    if (++gslot == NS) { gslot = 0; ++ground; }
                }
                continue;
            }
        } else {
            if (batch_id > begin_idx) __syncthreads();   // previous segment is done with the smem ring
        }

        // ================================ consumers ================================
        // ---- Q -> registers (A fragments of k-steps 4i+j) ----
        uint32_t qf[9][4];
        {
            const T *qb = reinterpret_cast<const T *>(params.q_ptr) + (int64_t)batch_id * params.q_batch_stride +
                          (int64_t)bidh * params.q_head_stride;
            const T *q0 = qb + (int64_t)row0 * params.q_row_stride + 16 * j + 2 * t;
            const T *q1 = qb + (int64_t)row1 * params.q_row_stride + 16 * j + 2 * t;
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                qf[i][0] = rv0 ? ldg_nc_u32(q0 + 64 * i) : 0u;
                qf[i][1] = rv1 ? ldg_nc_u32(q1 + 64 * i) : 0u;
                qf[i][2] = rv0 ? ldg_nc_u32(q0 + 64 * i + 8) : 0u;
                qf[i][3] = rv1 ? ldg_nc_u32(q1 + 64 * i + 8) : 0u;
            }
        }

        int pg_pending = 0;
        if constexpr (!SPEC) {
            int pgv[NS];   // page ids of the tiles in flight; pgv[NS-1] is the look-ahead for the next prefetch
#pragma unroll
            for (int s = 0; s < NS; ++s) pgv[s] = load_pg(s);
#pragma unroll
            for (int s = 0; s < NS - 1; ++s) {
                if (s < n_tiles) {
                    const int tok0 = tok_begin + s * NT;
                    const T *src = kbase + (int64_t)pgv[s] * k_batch_stride + (int64_t)(tok0 & ps_mask) * k_row_stride;
                    issue_tile_copy<NT, C::kNThreads>(ring_u32 + s * C::kStageBytes, src, k_row_stride, tok0, seqlen_k, tid);
                }
                cp_async_commit();
            }
            pg_pending = pgv[NS - 1];
        }

        // per-row causal / split column limits (absolute token index; col >= limit is masked)
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
        for (int n = 0; n < n_tiles; ++n) {
            const int tok0 = tok_begin + n * NT;
            if constexpr (!SPEC) {
                cp_async_wait<NS - 2>();
                __syncthreads();
                {
                    const int pg_cur = pg_pending;
                    pg_pending = load_pg(n + NS);   // consumed one iteration later: hides the load latency
                    if (n + NS - 1 < n_tiles) {
                        const int tok_p = tok_begin + (n + NS - 1) * NT;
                        const T *src = kbase + (int64_t)pg_cur * k_batch_stride + (int64_t)(tok_p & ps_mask) * k_row_stride;
                        issue_tile_copy<NT, C::kNThreads>(ring_u32 + wr_slot * C::kStageBytes, src, k_row_stride, tok_p, seqlen_k, tid);
                    }
                }
                cp_async_commit();
                compute_tile<C, ET>(ring_u32 + rd_slot * C::kStageBytes, tok0, sS_mine_base, sS_grp_base, grp, kq_off, vq_off, t, sl2, qf,
                                    acc_o, m0, m1, l0, l1, lim0, lim1, mask_min);
                rd_slot = (rd_slot + 1 == NS) ? 0 : rd_slot + 1;
                wr_slot = (wr_slot + 1 == NS) ? 0 : wr_slot + 1;
            } else {
                mbar_wait(bar_u32 + gslot * 8, ground & 1);   // tile landed in smem
                const int boff = sbuf * kSBufElems;
                compute_tile<C, ET>(ring_u32 + gslot * C::kStageBytes, tok0, sS_mine_base + boff, sS_grp_base + boff, grp, kq_off, vq_off, t,
                                    sl2, qf, acc_o, m0, m1, l0, l1, lim0, lim1, mask_min);
                __syncwarp();
                if (lane == 0) mbar_arrive(bar_u32 + (NS + gslot) * 8);   // this warp is done reading the slot
                if (++gslot == NS) { gslot = 0; ++ground; }
                sbuf ^= 1;
            }
        }

        // ---- epilogue ----
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
        if (!split) {
            T *ob = reinterpret_cast<T *>(params.o_ptr) + (int64_t)batch_id * params.o_batch_stride + (int64_t)bidh * params.o_head_stride;
            T *o0 = ob + (int64_t)row0 * params.o_row_stride + 128 * j + 2 * t;
            T *o1 = ob + (int64_t)row1 * params.o_row_stride + 128 * j + 2 * t;
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                if (rv0) *reinterpret_cast<uint32_t *>(o0 + nt * 8) = ET::pack(acc_o[nt][0] * inv0, acc_o[nt][1] * inv0);
                if (rv1) *reinterpret_cast<uint32_t *>(o1 + nt * 8) = ET::pack(acc_o[nt][2] * inv1, acc_o[nt][3] * inv1);
            }
            if (j == 0 && t == 0) {
                float *lp = reinterpret_cast<float *>(params.softmax_lse_ptr) + ((int64_t)batch_id * hh + bidh) * M;
                if (rv0) lp[row0] = lse0;
                if (rv1) lp[row1] = lse1;
            }
        } else {
            const int split_offset = __ldg(params.num_splits_ptr + batch_id);
            const int64_t slot = split_offset + n_split_idx;
            float *oa = reinterpret_cast<float *>(params.oaccum_ptr) + ((slot * hh + bidh) * M) * kDV;
            float *o0 = oa + (int64_t)row0 * kDV + 128 * j + 2 * t;
            float *o1 = oa + (int64_t)row1 * kDV + 128 * j + 2 * t;
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                if (rv0) *reinterpret_cast<float2 *>(o0 + nt * 8) = make_float2(acc_o[nt][0] * inv0, acc_o[nt][1] * inv0);
                if (rv1) *reinterpret_cast<float2 *>(o1 + nt * 8) = make_float2(acc_o[nt][2] * inv1, acc_o[nt][3] * inv1);
            }
            if (j == 0 && t == 0) {
                float *la = reinterpret_cast<float *>(params.softmax_lseaccum_ptr) + (slot * hh + bidh) * M;
                if (rv0) la[row0] = lse0;
                if (rv1) la[row1] = lse1;
            }
        }
    }
    if constexpr (SPEC) {
        if (is_producer) cp_async_wait<0>();
    }
}

// ---------------------------------------------------------------------------------------------------------
// Separate split-KV combine kernel: one CTA (128 threads) per (batch, head, row).
// ---------------------------------------------------------------------------------------------------------
template <typename T>
__global__ void __launch_bounds__(128)
mla_combine_kernel(const __grid_constant__ Flash_fwd_mla_params params) {
    const int tid = threadIdx.x;
    const int bidx = blockIdx.x;
    const int M = params.seqlen_q;
    const int hs = params.h * M;
    const int batch_idx = bidx / hs;
    const int hs_idx = bidx - batch_idx * hs;

    const int split_offset = __ldg(params.num_splits_ptr + batch_idx);
    const int nsp = __ldg(params.num_splits_ptr + batch_idx + 1) - split_offset;
    if (nsp <= 1) return;
    FLASH_DEVICE_ASSERT(nsp <= kMaxSplits);

    __shared__ float sScale[kMaxSplits];
    const float *lse_acc = reinterpret_cast<const float *>(params.softmax_lseaccum_ptr) + (int64_t)split_offset * hs + hs_idx;

    if (tid < 32) {
        constexpr int kPer = kMaxSplits / 32;   // 8 splits per lane at most
        float lv[kPer];
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int s = tid + 32 * i;
            lv[i] = (s < nsp) ? __ldcg(lse_acc + (int64_t)s * hs) : -INFINITY;
        }
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < kPer; ++i) mx = fmaxf(mx, lv[i]);
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float mxs = (mx == -INFINITY) ? 0.f : mx;
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < kPer; ++i) sum += expf(lv[i] - mxs);
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        const float glse = (sum == 0.f || sum != sum) ? INFINITY : logf(sum) + mxs;
        if (tid == 0) reinterpret_cast<float *>(params.softmax_lse_ptr)[bidx] = glse;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int s = tid + 32 * i;
            if (s < nsp) sScale[s] = expf(lv[i] - glse);
        }
    }
    __syncthreads();

    const float *oa = reinterpret_cast<const float *>(params.oaccum_ptr) + ((int64_t)split_offset * hs + hs_idx) * kDV + tid * 4;
    const int64_t sstride = (int64_t)hs * kDV;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    int sp = 0;
    for (; sp + 8 <= nsp; sp += 8) {   // 8 independent loads in flight, accumulation order stays 0..nsp-1
        float4 v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = __ldcg(reinterpret_cast<const float4 *>(oa + (int64_t)(sp + u) * sstride));
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float w = sScale[sp + u];
            acc.x += w * v[u].x; acc.y += w * v[u].y; acc.z += w * v[u].z; acc.w += w * v[u].w;
        }
    }
    for (; sp < nsp; ++sp) {
        const float4 v = __ldcg(reinterpret_cast<const float4 *>(oa + (int64_t)sp * sstride));
        const float w = sScale[sp];
        acc.x += w * v.x; acc.y += w * v.y; acc.z += w * v.z; acc.w += w * v.w;
    }
    const int head_idx = hs_idx / M;
    const int row = hs_idx - head_idx * M;
    T *o = reinterpret_cast<T *>(params.o_ptr) + (int64_t)batch_idx * params.o_batch_stride + (int64_t)head_idx * params.o_head_stride +
           (int64_t)row * params.o_row_stride + tid * 4;
    using ET = ElemTraits<T>;
    uint2 pk;
    pk.x = ET::pack(acc.x, acc.y);
    pk.y = ET::pack(acc.z, acc.w);
    *reinterpret_cast<uint2 *>(o) = pk;
}

// ---------------------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------------------
template <typename C>
void launch_main(Flash_fwd_mla_params &p, cudaStream_t stream) {
    auto kernel = &mla_decode_kernel<C>;
    static bool attr_done[16] = {false};
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    if (dev < 16 && !attr_done[dev]) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem));
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        attr_done[dev] = true;
    }
    const int num_m_blocks = (p.seqlen_q + C::kBM - 1) / C::kBM;
    kernel<<<dim3(num_m_blocks, p.h, p.num_sm_parts), C::kNThreads, C::kSmem, stream>>>(p);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template <typename T>
void mla_sm86_run(Flash_fwd_mla_params &p, cudaStream_t stream) {
    FLASH_ASSERT(p.d == kD && p.d_v == kDV);
    FLASH_ASSERT(p.k_ptr == p.v_ptr);
    FLASH_ASSERT(p.page_block_size == 16 || p.page_block_size == 32 || p.page_block_size == 64);
    FLASH_ASSERT((p.k_row_stride % 8) == 0 && (p.k_batch_stride % 8) == 0);
    FLASH_ASSERT(p.kv_dtype == kKVNative);
    const Plan plan = pick_plan(p.seqlen_q, p.page_block_size);

#define MLA_SM86_CASE(BM_, NT_, NS_, MB_, SP_)                                                             \
    if (plan.bm == BM_ && plan.nt == NT_ && plan.ns == NS_ && plan.minb == MB_ && plan.spec == SP_) {      \
        launch_main<Cfg<T, BM_, NT_, NS_, MB_, SP_>>(p, stream);                                           \
        launched = true;                                                                                   \
    }
    bool launched = false;
    // v2a
    MLA_SM86_CASE(16, 16, 3, 1, false)
    MLA_SM86_CASE(16, 16, 4, 1, false)
    MLA_SM86_CASE(32, 16, 3, 1, false)
    MLA_SM86_CASE(32, 16, 4, 1, false)
    MLA_SM86_CASE(16, 16, 2, 2, false)
    MLA_SM86_CASE(16, 32, 2, 1, false)
    MLA_SM86_CASE(32, 32, 2, 1, false)
    // v2b (BM=32 / 2-CTA-per-SM variants spill: the 9th warp lowers the per-thread register cap to 168, so they are not instantiated)
    MLA_SM86_CASE(16, 16, 3, 1, true)
    MLA_SM86_CASE(16, 16, 4, 1, true)
    MLA_SM86_CASE(16, 32, 2, 1, true)
#undef MLA_SM86_CASE
    FLASH_ASSERT(launched);

    mla_combine_kernel<T><<<p.b * p.h * p.seqlen_q, 128, 0, stream>>>(p);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace mla_sm86
