#pragma once
// Weight-streaming GEMM for decode-sized token counts (T <= 64), batched over G groups (heads):
//     y[t, g, n] = sum_k x[t, g, k] * w[g, n, k]          bf16 / fp16 in, fp32 accumulate, one rounding on output
// x and w are both K-contiguous ("TN"), so the layer's projections map onto it directly:
//     qkv_a / o_proj : G = 1, w = the [N, K] nn.Linear weight
//     W_UK / W_UV    : G = heads, w = the per-head matrix pre-transposed to [G, N, K]; x / y may be strided views
//                      (q_nope inside the qkv output, y written straight into the [T, H * 128] o_proj input)
// Deterministic: fixed summation order everywhere; split-K partials are reduced by the last-arriving CTA of a tile in
// split order, and the per-tile counters are returned to zero (CUDA-graph replay safe).
#include <cstdint>
#include <cuda_runtime.h>

struct SkinnyGemmParams {
    const void *x;                    // x[t][g][k] at x + t * x_st + g * x_sg + k
    const void *w;                    // [G, N, K] contiguous
    void *y;                          // y[t][g][n] at y + t * y_st + g * y_sg + n
    int64_t x_st, x_sg, y_st, y_sg;   // element strides (x: multiples of 8, y: multiples of 2; checked by the host glue)
    int T, G, N, K;                   // T <= 64, N % n_tile == 0, K % 64 == 0
    int n_tile;                       // weight rows per CTA: 16 | 32 | 64
    int splits;                       // split-K factor, 1 <= splits <= K / 64
    float *ws;                        // splits > 1: >= G * N * splits * T floats of scratch
    int *counters;                    // splits > 1: >= G * (N / n_tile) ints, zero on entry, zero again on exit
    int is_bf16;                      // 1: bf16, 0: fp16
};

void launch_skinny_gemm(const SkinnyGemmParams &p, cudaStream_t stream);
