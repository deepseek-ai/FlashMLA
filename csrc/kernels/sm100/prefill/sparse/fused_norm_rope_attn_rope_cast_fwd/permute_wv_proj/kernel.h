#pragma once

#include <cutlass/float8.h>

#include "kernels/params.h"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_wv_proj {

// Local alias: `kernels/defines.h` calls this type `fp8`; this header needs the explicit name
using fp8_e4m3 = cutlass::float_e4m3_t;

struct Params {
    uint32_t d_o;           // O head dimension
    uint32_t input_gran;          // Scale granularity for input, must be 32
    uint32_t wv_group_size; // Number of O heads per WV group. Always equal to 8 for V4 pro/flash
    uint32_t n_wv_group;    // = h_o / wv_group_size
    uint32_t d_proj_out;    // dimension of wv_proj's output. Equal to vLLM's V33Attention.o_head_dim

    // Input tensors
    fp8_e4m3* __restrict__ wv_proj;    // [n_wv_group, d_proj_out, wv_group_size*d_o], contiguous on the last dim
    uint64_t stride_wv_proj_dim0, stride_wv_proj_dim1;
    int32_t* __restrict__ scale_factors;    // [n_wv_group, d_proj_out, wv_group_size*d_o/gran/4], contiguous on the SECOND dim (DeepGeMM's format)
    uint64_t stride_scale_factors_dim0, stride_scale_factors_dim2;

    // Output tensors
    fp8_e4m3* __restrict__ wv_proj_permuted;  // The same shape as wv_proj
    uint64_t stride_wv_proj_permuted_dim0, stride_wv_proj_permuted_dim1;
    int32_t* __restrict__ scale_factors_permuted;    // [n_wv_group, d_proj_out, wv_group_size * d_o / 32 / 4] - The granularity is fixed to 32
    uint64_t stride_scale_factors_permuted_dim0, stride_scale_factors_permuted_dim2;
    
    cudaStream_t stream;
};

void run_permute_wv_proj_kernel(const Params& params);

}
