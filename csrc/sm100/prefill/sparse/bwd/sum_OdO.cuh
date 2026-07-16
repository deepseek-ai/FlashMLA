#pragma once

// Per-token, per-head reduction: d_row[t, h] = sum_d(o[t, h, d] * do[t, h, d]).
// Standard FA3 "delta" precompute, used in the main bwd to form
//     dS = P * (dP - d_row)
//
// Layout: o, do are [s_q, h_q, d_v] bf16; d_row is [s_q, h_q] float.
//
// Grid: dim3(ceil_div(s_q, kBlockQ), h_q, 1)
// Block: dim3(kNumThreadsD, kNumThreadsQ, 1)
//
// Each thread handles `kBlockQ / kNumThreadsQ` rows and reduces D in
// chunks of `kElemsPerThread` via __shfl_xor_sync across kNumThreadsD lanes.
//
// kNumThreadsD must be a power of two (we use 8 for D_V=512: 8 lanes x 64 d/lane).

#include <cuda_bf16.h>
#include "params.h"

namespace sm100::sparse_bwd {

template<int D_V, int kBlockQ = 16, int kNumThreadsD = 8, int kNumThreadsQ = 16>
__global__ __launch_bounds__(kNumThreadsD * kNumThreadsQ, 4)
void sum_OdO_kernel(
    const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ do_grad,
    int s_q, int h_q,
    int stride_o_s_q, int stride_o_h,
    int stride_do_s_q, int stride_do_h,
    float* __restrict__ d_row
) {
    static_assert(D_V % (kNumThreadsD * 2) == 0, "D_V must be divisible by 2*kNumThreadsD for bf16x2 loads");
    constexpr int kElemsPerThread = D_V / kNumThreadsD;  // 64 for D_V=512

    const int tid_q = threadIdx.y;
    const int tid_d = threadIdx.x;
    const int block_q_start = blockIdx.x * kBlockQ;
    const int h = blockIdx.y;

    // Each warp-row owns kBlockQ / kNumThreadsQ query rows.
    constexpr int kRowsPerThread = kBlockQ / kNumThreadsQ;  // 1 for default config
    static_assert(kBlockQ % kNumThreadsQ == 0, "kBlockQ must be divisible by kNumThreadsQ");

    #pragma unroll
    for (int row_i = 0; row_i < kRowsPerThread; ++row_i) {
        const int q = block_q_start + tid_q * kRowsPerThread + row_i;
        if (q >= s_q) continue;

        const __nv_bfloat16* o_row  = o      + q * stride_o_s_q  + h * stride_o_h;
        const __nv_bfloat16* do_row = do_grad+ q * stride_do_s_q + h * stride_do_h;

        float acc = 0.0f;
        const int d_start = tid_d * kElemsPerThread;

        #pragma unroll
        for (int i = 0; i < kElemsPerThread; i += 2) {
            __nv_bfloat162 o_pair  = *reinterpret_cast<const __nv_bfloat162*>(o_row  + d_start + i);
            __nv_bfloat162 do_pair = *reinterpret_cast<const __nv_bfloat162*>(do_row + d_start + i);
            float2 o_f  = __bfloat1622float2(o_pair);
            float2 do_f = __bfloat1622float2(do_pair);
            acc += o_f.x * do_f.x + o_f.y * do_f.y;
        }

        // Reduce across kNumThreadsD lanes within the warp.
        #pragma unroll
        for (int offset = kNumThreadsD / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset, 32);
        }

        if (tid_d == 0) {
            // dense MLA bwd's compute() does `dS = P * (dP + sum_odo)`
            // expecting sum_odo to be NEGATED during this precompute. Store
            // -acc so downstream `cute::add(dif, dpt, odo)` yields dP - sum(O*dO).
            d_row[q * h_q + h] = -acc;
        }
    }
}

template<int D_V>
void launch_sum_OdO(const SparseAttnBwdParams& params) {
    constexpr int kBlockQ      = 16;
    constexpr int kNumThreadsD = 8;
    constexpr int kNumThreadsQ = 16;

    dim3 grid((params.s_q + kBlockQ - 1) / kBlockQ, params.h_q, 1);
    dim3 block(kNumThreadsD, kNumThreadsQ, 1);

    sum_OdO_kernel<D_V, kBlockQ, kNumThreadsD, kNumThreadsQ><<<grid, block, 0, params.stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(params.o),
        reinterpret_cast<const __nv_bfloat16*>(params.do_grad),
        params.s_q, params.h_q,
        params.stride_o_s_q,  params.stride_o_h,
        params.stride_do_s_q, params.stride_do_h,
        params.d_row
    );
}

}  // namespace sm100::sparse_bwd
