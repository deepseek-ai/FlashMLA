// Tile-scheduler metadata for the persistent split-KV MLA decode kernels (parallel, device-side).
//
// Every batch element i costs  cost_i = max(1, ceil(seqlen_i / G)) + ov  "units" (G = scheduler granule in tokens,
// ov = fixed per-segment overhead expressed in blocks).  The units of all batches form one line; the line is cut
// into num_sm_parts equal ranges and range p becomes the work of persistent CTA p.  A range may start / end in the
// middle of a batch element (split-KV).  All num_sm_parts ranges are located in parallel by binary search on the
// prefix sums of the costs, instead of the single-thread serial loop of the original implementation.
//
// Output format is identical to the legacy kernels:
//   tile_scheduler_metadata[p] = {begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, 0, 0, 0}
//   num_splits[i]              = number of splits of all batch elements < i (prefix sum), num_splits[b] = total.
// A CTA whose range is empty gets begin_idx = b (the main kernels return immediately).

#include "static_switch.h"
#include "flash_mla.h"

static constexpr int kMetaThreads = 1024;
static constexpr int kMaxBatchMeta = 12000;   // (b + 1) ints of dynamic smem must stay below 48 KB

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

    const int payload = (total + P - 1) / P;

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
        const long long lo_x = (long long)p * payload;
        const long long hi_x = lo_x + payload;
        locate(lo_x >= total ? total : (int)lo_x, bi, bblk);
        locate(hi_x >= total ? total : (int)hi_x, ei, eblk);
        int *out = params.tile_scheduler_metadata_ptr + p * TileSchedulerMetaDataSize;
        int4 m0;
        int m4 = 0;
        const bool empty = (bi >= b) || (bi > ei) || (bi == ei && bblk >= eblk);
        if (empty) {
            m0 = make_int4(b, 0, b - 1, 0);
        } else {
            m0.x = bi;
            m0.y = bblk * G;
            if (eblk > 0) { m0.z = ei; m0.w = eblk * G; }
            else          { m0.z = ei - 1; m0.w = seqlens[ei - 1]; }
            m4 = p - (C[bi] + ov) / payload;
        }
        *reinterpret_cast<int4 *>(out) = m0;
        *reinterpret_cast<int4 *>(out + 4) = make_int4(m4, 0, 0, 0);
    }

    // ---- num_splits: splits_i = part(last unit) - part(first block unit) + 1; exclusive scan ----
    int locl = 0;
    for (int k = 0; k < per; ++k) {
        const int i = tid * per + k;
        if (i < b) locl += (C[i + 1] - 1) / payload - (C[i] + ov) / payload + 1;
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
            run2 += (C[i + 1] - 1) / payload - (C[i] + ov) / payload + 1;
        }
    }
    if (tid == 0) params.num_splits_ptr[b] = warp_sums[kMetaThreads / 32 - 1];
}

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.batch_size < kMaxBatchMeta);
    FLASH_ASSERT(params.block_size_n > 0 && params.num_sm_parts > 0);
    const size_t smem = (size_t)(params.batch_size + 1) * sizeof(int);
    get_mla_metadata_kernel<<<1, kMetaThreads, smem, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
