// Tile-scheduler metadata for the persistent split-KV MLA decode kernels (parallel, device-side).
//
// Every batch element i costs  cost_i = max(1, ceil(seqlen_i / G)) + ov  "units" (G = scheduler granule in tokens,
// ov = fixed per-segment overhead expressed in blocks).  The units of all batches form one line of T units; the line
// is cut into P_eff <= num_sm_parts ranges [floor(p*T/P_eff), floor((p+1)*T/P_eff)) (sizes differ by at most one unit,
// so no part is left empty by rounding) and range p becomes the work of persistent CTA p.  A range may start / end in the
// middle of a batch element (split-KV).  All num_sm_parts ranges are located in parallel by binary search on the
// prefix sums of the costs, instead of the single-thread serial loop of the original implementation.
//
// Output format is identical to the legacy kernels:
//   tile_scheduler_metadata[p] = {begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, 0, 0, 0}
//   num_splits[i]              = number of splits of all batch elements < i (prefix sum), num_splits[b] = total.
// A CTA whose range is empty gets begin_idx = b (the main kernels return immediately).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "static_switch.h"
#include "flash_mla.h"

static constexpr int kMaxBatchMeta = 12000;   // (b + 1) ints of dynamic smem must stay below 48 KB

template <int kMetaThreads>
__global__ void __launch_bounds__(kMetaThreads)
get_mla_metadata_kernel(__grid_constant__ const Mla_metadata_params params) {
    extern __shared__ int C[];                       // exclusive prefix of costs, C[b] = total
    __shared__ int warp_sums[kMetaThreads / 32];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int b = params.batch_size;
    const int G = params.block_size_n;
    const int ov = params.fixed_overhead_num_blocks;
    const int P = params.num_sm_parts;
    const int *seqlens = params.seqlens_k_ptr;

    auto cost_of = [&](int i) {
        const int nb = (seqlens[i] + G - 1) / G;
        return (nb > 0 ? nb : 1) + ov;
    };

    // ---- exclusive scan of the costs ----
    const int per = (b + kMetaThreads - 1) / kMetaThreads;
    int local = 0;
    for (int k = 0; k < per; ++k) {
        const int i = tid * per + k;
        if (i < b) local += cost_of(i);
    }
    int v = local;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += y;
    }
    if (lane == 31) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = lane < kMetaThreads / 32 ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        if (lane < kMetaThreads / 32) warp_sums[lane] = w;
    }
    __syncthreads();
    const int total = warp_sums[kMetaThreads / 32 - 1];
    int run = v - local + (warp > 0 ? warp_sums[warp - 1] : 0);
    for (int k = 0; k < per; ++k) {
        const int i = tid * per + k;
        if (i < b) { C[i] = run; run += cost_of(i); }
    }
    if (tid == 0) C[b] = total;
    __syncthreads();

    // number of active parts: every part should own at least `min_blocks_per_part` units
    // (Pe <= total guarantees that no part owns zero units, so the split indices of a batch element are dense.)
    long long Pe = P < total ? P : total;
    if (params.min_blocks_per_part > 0) {
        const long long cap = total / params.min_blocks_per_part;
        if (cap < Pe) Pe = cap < 1 ? 1 : cap;
    }
    // part containing unit x (parts are [floor(p*T/Pe), floor((p+1)*T/Pe)))
    auto part_of = [&](int x) -> int { return (int)((((long long)x + 1) * Pe - 1) / total); };

    // position of the first block unit >= x:  (batch index, block index inside the batch); (b, 0) if x >= total
    auto locate = [&](int x, int &bi, int &blk) {
        if (x >= total) { bi = b; blk = 0; return; }
        int lo = 0, hi = b - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (C[mid] <= x) lo = mid; else hi = mid - 1;
        }
        const int off = x - C[lo];
        bi = lo;
        blk = off < ov ? 0 : off - ov;
    };

    for (int p = tid; p < P; p += kMetaThreads) {
        int bi, bblk, ei, eblk;
        int *out = params.tile_scheduler_metadata_ptr + p * TileSchedulerMetaDataSize;
        int4 m0 = make_int4(b, 0, b - 1, 0);
        int m4 = 0, m5 = 0;
        if (p < Pe) {
            const int lo_x = (int)((long long)p * total / Pe);
            const int hi_x = (int)((long long)(p + 1) * total / Pe);
            locate(lo_x, bi, bblk);
            locate(hi_x, ei, eblk);
            const bool empty = (bi >= b) || (bi > ei) || (bi == ei && bblk >= eblk);
            if (!empty) {
                m0.x = bi;
                m0.y = bblk * G;
                if (eblk > 0) { m0.z = ei; m0.w = eblk * G; }
                else          { m0.z = ei - 1; m0.w = seqlens[ei - 1]; }
                m4 = p - part_of(C[bi] + ov);
                m5 = seqlens[bi];   // saves the main kernel one dependent load at start-up
            }
        }
        *reinterpret_cast<int4 *>(out) = m0;
        *reinterpret_cast<int4 *>(out + 4) = make_int4(m4, m5, 0, 0);
    }

    // ---- num_splits: splits_i = part(last unit) - part(first block unit) + 1; exclusive scan ----
    int locl = 0;
    for (int k = 0; k < per; ++k) {
        const int i = tid * per + k;
        if (i < b) locl += part_of(C[i + 1] - 1) - part_of(C[i] + ov) + 1;
    }
    v = locl;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += y;
    }
    __syncthreads();
    if (lane == 31) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = lane < kMetaThreads / 32 ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        if (lane < kMetaThreads / 32) warp_sums[lane] = w;
    }
    __syncthreads();
    int run2 = v - locl + (warp > 0 ? warp_sums[warp - 1] : 0);
    for (int k = 0; k < per; ++k) {
        const int i = tid * per + k;
        if (i < b) {
            params.num_splits_ptr[i] = run2;
            run2 += part_of(C[i + 1] - 1) - part_of(C[i] + ov) + 1;
        }
    }
    if (tid == 0) params.num_splits_ptr[b] = warp_sums[kMetaThreads / 32 - 1];
}

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.batch_size < kMaxBatchMeta);
    FLASH_ASSERT(params.block_size_n > 0 && params.num_sm_parts > 0);
    const size_t smem = (size_t)(params.batch_size + 1) * sizeof(int);
    if (params.batch_size <= 128) {   // small batches: a small block starts and synchronises faster
        get_mla_metadata_kernel<128><<<1, 128, smem, stream>>>(params);
    } else {
        get_mla_metadata_kernel<1024><<<1, 1024, smem, stream>>>(params);
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}
