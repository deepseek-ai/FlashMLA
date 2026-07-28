#pragma once
//
// k2q CSR builder for the block-sparse MLA backward (MSA-inspired; see
// docs/MSA_BLOCK_SPARSE_KV_OUTER_BWD_DESIGN.md).
//
// Converts the per-query block selection
//     q2k_indices [total_q, topk]   int32   (KV-block ids, -1 = pad/invalid)
// into the reverse CSR
//     k2q_row_ptr   [num_kv_blocks + 1]      int32   (exclusive prefix sums)
//     k2q_q_indices [total_q * topk]         int32   (attending q per block)
// so the KV-outer backward (one CTA per KV block) can iterate exactly the
// queries that attend each block -- enabling accumulate-then-store dK/dV with
// NO atomicAdd scatter (the current per-token sparse bwd's main cost / nondet
// source). Single KV head (MLA, h_kv == 1); extend with a head stride for GQA.
//
// Standard COO->CSR build: count (histogram) -> exclusive prefix sum -> scatter.
//
#include <cuda_runtime.h>

namespace sm100::block_bwd {

// counts[b] += 1 for every (q, slot) entry whose selected block id == b.
__global__ void k2q_count_kernel(
    const int* __restrict__ q2k, int total_q, int topk, int num_kv_blocks,
    int* __restrict__ counts) {
  int n = total_q * topk;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    int b = q2k[i];
    if (b >= 0 && b < num_kv_blocks) atomicAdd(&counts[b], 1);
  }
}

// Exclusive prefix sum counts[0..num_kv_blocks) -> row_ptr[0..num_kv_blocks],
// and seed the scatter write cursor at each row's start. num_kv_blocks is small
// (ceil(s_kv / kv_block_size); a few hundred to a few thousand) so a single-
// thread scan is fine; swap for CUB DeviceScan if it ever dominates.
__global__ void k2q_prefix_kernel(
    const int* __restrict__ counts, int num_kv_blocks,
    int* __restrict__ row_ptr, int* __restrict__ cursor) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int acc = 0;
    for (int b = 0; b < num_kv_blocks; ++b) {
      row_ptr[b] = acc;
      cursor[b] = acc;  // write cursor starts at this row's first slot
      acc += counts[b];
    }
    row_ptr[num_kv_blocks] = acc;
  }
}

// Scatter: place q at the next free slot of its selected block's CSR segment.
// (Order within a block is atomic-arrival order; the bwd accumulates dK/dV over
// the whole set so the SET is exact. Sort per-row offline if bitwise-determinism
// of the FP32 accumulation order is required.)
__global__ void k2q_scatter_kernel(
    const int* __restrict__ q2k, int total_q, int topk, int num_kv_blocks,
    int* __restrict__ cursor, int* __restrict__ q_indices) {
  int n = total_q * topk;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    int b = q2k[i];
    if (b >= 0 && b < num_kv_blocks) {
      int q = i / topk;
      int pos = atomicAdd(&cursor[b], 1);
      q_indices[pos] = q;
    }
  }
}

// Host launcher. Caller allocates (device): counts[num_kv_blocks],
// cursor[num_kv_blocks], row_ptr[num_kv_blocks+1], q_indices[total_q*topk].
inline void build_k2q_csr(
    const int* d_q2k, int total_q, int topk, int num_kv_blocks,
    int* d_counts, int* d_cursor, int* d_row_ptr, int* d_q_indices,
    cudaStream_t stream) {
  cudaMemsetAsync(d_counts, 0, (size_t)num_kv_blocks * sizeof(int), stream);
  int n = total_q * topk;
  int threads = 256;
  int grid = (n + threads - 1) / threads;
  if (grid > 0) {
    k2q_count_kernel<<<grid, threads, 0, stream>>>(
        d_q2k, total_q, topk, num_kv_blocks, d_counts);
  }
  k2q_prefix_kernel<<<1, 1, 0, stream>>>(
      d_counts, num_kv_blocks, d_row_ptr, d_cursor);
  if (grid > 0) {
    k2q_scatter_kernel<<<grid, threads, 0, stream>>>(
        d_q2k, total_q, topk, num_kv_blocks, d_cursor, d_q_indices);
  }
}

// Sort each CSR row's attending-index segment ascending so the dK/dV reduction
// order is FIXED -> bitwise-deterministic accumulation (the atomic-cursor scatter
// above leaves the per-row order racy).
__global__ void k2q_sort_kernel(const int* __restrict__ row_ptr, int* __restrict__ q_idx, int num_rows) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_rows) return;
  int lo = row_ptr[r], hi = row_ptr[r + 1];
  for (int i = lo + 1; i < hi; ++i) {
    int key = q_idx[i], j = i - 1;
    while (j >= lo && q_idx[j] > key) { q_idx[j + 1] = q_idx[j]; --j; }
    q_idx[j + 1] = key;
  }
}

inline void sort_k2q_csr(const int* d_row_ptr, int* d_q_indices, int num_rows, cudaStream_t stream) {
  if (num_rows > 0)
    k2q_sort_kernel<<<(num_rows + 127) / 128, 128, 0, stream>>>(d_row_ptr, d_q_indices, num_rows);
}

}  // namespace sm100::block_bwd
