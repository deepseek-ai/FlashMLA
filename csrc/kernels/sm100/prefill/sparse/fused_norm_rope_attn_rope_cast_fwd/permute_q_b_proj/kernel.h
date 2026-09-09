#pragma once

#include <cutlass/float8.h>

#include "kernels/params.h"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_q_b_proj {

// Local alias: `kernels/defines.h` calls this type `fp8`; this header needs the explicit name
using fp8_e4m3 = cutlass::float_e4m3_t;

struct Params {
    uint32_t h_q;           // Number of q heads
    uint32_t d_q;           // Q head dimension
    uint32_t q_lora_rank;   // Q LoRA rank (1024 for V4 Flash, 1536 for V4 Pro)
    uint32_t gran;          // Scale granularity, 32 or 128

    // Input tensors
    fp8_e4m3* __restrict__ q_b_proj;    // [h_q*d_q, q_lora_rank], contiguous on the last dim
    uint64_t stride_q_b_dim0;
    int32_t* __restrict__ scale_factors;    // [h_q*d_q, q_lora_rank/gran/4], contiguous on the FIRST dim (DeepGeMM's format)
    uint64_t stride_scale_factors_dim1;

    // Output tensors
    fp8_e4m3* __restrict__ q_b_proj_permuted;  // The same shape as q_b_proj
    uint64_t stride_q_b_permuted_dim0;
    int32_t* __restrict__ scale_factors_permuted;    // The same shape as scale_factors
    uint64_t stride_scale_factors_permuted_dim1;
    
    cudaStream_t stream;
};

void run_permute_q_b_proj_kernel(const Params& params);

}
