/*
Transform q_b_proj layout (include the weight and its scale factor) into layout required by "fused_norm_rope_attn_rope_cast_fwd" kernel

"fused_norm_rope_attn_rope_cast_fwd" 那边，为了最优秀的性能，我们希望输入的 q 的 layout 长这样：

[(h0d0 h0d1 ... h0d15) (h1d0 h1d1 ... h1d15) ... (hHd0 ... hHd15)] [(h0d16 ... h0d31) ... (hHd16 ... hHd31)] ... [(h0d496 ... h0d511) ... (hHd496 ... hHdD)]

where H = (head of q) - 1, D = (headdim of q) - 1

因此，我们需要把 q_b_proj 的行（如果假设 q_b_proj 的 shape 是 (H*D) * q_lora_rank 的话）互换。这个 kernel 负责该互换。

block dim: 32，一个 warp 负责 permute q_b_proj 的一行
grid dim: H*D
*/

#include "kernel.h"

#include <kerutils/kerutils.cuh>

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_q_b_proj {

__launch_bounds__(32)
__global__ void permute_q_b_proj_kernel(__grid_constant__ const Params params) {
    uint32_t row_idx = blockIdx.x;
    uint32_t head_idx = row_idx / params.d_q;
    uint32_t head_dim_idx = row_idx % params.d_q;
    uint32_t out_row_idx = head_dim_idx % 16 + head_idx * 16 + (head_dim_idx / 16u) * (params.h_q * 16);
    uint32_t num_scales_per_row = params.q_lora_rank / params.gran / 4;

    __shared__ uint64_t bar_storage;
    ku::transac_bar_t& bar = *(ku::transac_bar_t*)&bar_storage;
    if (cute::elect_one_sync()) {
        bar.init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    CUTE_ALIGNAS(1024) extern __shared__ fp8_e4m3 row_buf[];
    if (cute::elect_one_sync()) {
        cute::SM90_BULK_COPY_G2S::copy(
            params.q_b_proj + row_idx * params.stride_q_b_dim0,
            (uint64_t*)&bar,
            row_buf,
            params.q_lora_rank
        );
    }

    for (uint32_t scale_idx = threadIdx.x; scale_idx < num_scales_per_row; scale_idx += 32) {
        auto cur_scale = params.scale_factors[row_idx + scale_idx * params.stride_scale_factors_dim1];
        params.scale_factors_permuted[out_row_idx + scale_idx * params.stride_scale_factors_permuted_dim1] = cur_scale;
    }

    if (cute::elect_one_sync()) {
        bar.arrive_and_expect_tx(params.q_lora_rank);
        bar.wait(0);
        cutlass::arch::fence_view_async_shared();
        cute::SM90_BULK_COPY_S2G::copy(
            row_buf,
            params.q_b_proj_permuted + out_row_idx * params.stride_q_b_permuted_dim0,
            params.q_lora_rank
        );
    }
}

void run_permute_q_b_proj_kernel(const Params& params) {
    uint32_t smem_size = params.q_lora_rank;
    KU_CUDA_CHECK(cudaFuncSetAttribute(permute_q_b_proj_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    permute_q_b_proj_kernel<<<params.h_q*params.d_q, 32, smem_size, params.stream>>>(params);
}

}
