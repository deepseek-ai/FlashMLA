/*
Transform wv_proj layout (include the weight and its scale factor) into layout required by "fused_norm_rope_attn_rope_cast_fwd" kernel

"fused_norm_rope_attn_rope_cast_fwd" 那边，为了最优秀的性能，我们输出的 o 的 layout 长这样：

- 首先，我们只关注一个 token 对应的 o，其 shape 为 [h_o, d_o]
- 将 head 按照 wv group 分组，其 shape 变成 [n_wv_group, wv_group_size, d_o]
- 对于每个 wv group（此时 shape 为 [wv_group_size, d_o]），输出为
    [(h0d0 h0d1 ... h0d31) (h1d0 h1d1 ... h1d31) ... (hGd0 ... hGd31)] [(h0d32 ... h0d63) ... (hGd32 ... hGd63)] ... [(h0d480 ... h0d511) ... (hGd480 ... hGdD)]
- 这个操作相当于把 head dim 上的每 32 个元素打包后，进行转置操作

where G = wv_group_size - 1, D = (headdim of o) - 1

因此，我们需要对 wv_proj 中的每个 wv_group 的权重分别处理。对于一个 wv group 的权重，把 wv_proj 的不同列互换。这个 kernel 负责该互换。

关于 scale factor：由于这一次我们在 permute input channel（而不是像 q_b 一样 permute output channel），且 permute 粒度为 32

block dim: 32，一个 warp 负责变换 wv proj weight 的某个 wv_group 的一整行（如果认为 wv_proj 的 shape 是 [n_wv_group, d_proj_out, wv_group_size * d_o] 的话）
grid dim: (d_proj_out, n_wv_group)
*/

#include "kernel.h"

#include <kerutils/kerutils.cuh>

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_wv_proj {

static constexpr uint32_t INPUT_GRAN = 32;
static constexpr uint32_t CHUNK_SIZE = 32;  // 沿着 d_o 方向，每 32 个元素为一个 chunk
static constexpr uint32_t D_O = 512;

__launch_bounds__(32)
__global__ void permute_wv_proj_kernel(__grid_constant__ const Params params) {
    uint32_t row_idx = blockIdx.x;
    uint32_t wv_group_idx = blockIdx.y;
    uint32_t row_size = params.wv_group_size * D_O;

    __shared__ uint64_t bar_storage;
    ku::transac_bar_t& bar = *(ku::transac_bar_t*)&bar_storage;
    if (cute::elect_one_sync()) {
        bar.init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Load the row
    CUTE_ALIGNAS(1024) extern __shared__ fp8_e4m3 smem_buf[];
    fp8_e4m3* row_buf = smem_buf;   // [row_size]
    fp8_e4m3* permuted_row_buf = smem_buf + row_size;   // [row_size]
    if (cute::elect_one_sync()) {
        cute::SM90_BULK_COPY_G2S::copy(
            params.wv_proj + wv_group_idx * params.stride_wv_proj_dim0 + row_idx * params.stride_wv_proj_dim1,
            &bar_storage,
            row_buf,
            row_size
        );
        bar.arrive_and_expect_tx(row_size);
    }

    // Permute SF
    uint32_t num_sf_per_head = D_O / 32;  // The output granularity is fixed to 32
    uint32_t num_sf = params.wv_group_size * num_sf_per_head;
    for (uint32_t i = threadIdx.x; i < num_sf; i += 32) {
        uint32_t head_idx = i / num_sf_per_head;
        uint32_t head_dim_chunk_idx = i % num_sf_per_head;
        uint32_t input_sf_idx_in_row = i;
        uint8_t cur_sf = *((uint8_t*)(params.scale_factors + wv_group_idx * params.stride_scale_factors_dim0 + row_idx + (input_sf_idx_in_row / 4) * params.stride_scale_factors_dim2) + input_sf_idx_in_row % 4);
        uint32_t output_sf_idx_in_row = head_idx + head_dim_chunk_idx * params.wv_group_size;
        *((uint8_t*)(params.scale_factors_permuted + wv_group_idx * params.stride_scale_factors_permuted_dim0 + row_idx + (output_sf_idx_in_row / 4) * params.stride_scale_factors_permuted_dim2) + output_sf_idx_in_row % 4) = cur_sf;
    }

    // Wait for the row to be ready, and permute the row
    bar.wait(0);
    for (uint32_t i = threadIdx.x; i < params.wv_group_size * (D_O / CHUNK_SIZE); i += 32) {
        fp8_e4m3 data[CHUNK_SIZE];
        *(__int128_t*)(data +  0) = ku::ld_shared(row_buf + i * CHUNK_SIZE);
        *(__int128_t*)(data + 16) = ku::ld_shared(row_buf + i * CHUNK_SIZE + 16);
        uint32_t head_idx = i / (D_O / CHUNK_SIZE);
        uint32_t head_dim_chunk_idx = i % (D_O / CHUNK_SIZE);
        uint32_t chunk_idx_in_permuted_row = head_dim_chunk_idx * params.wv_group_size + head_idx;
        ku::st_shared(permuted_row_buf + chunk_idx_in_permuted_row * CHUNK_SIZE +  0, *(__int128_t*)(data +  0));
        ku::st_shared(permuted_row_buf + chunk_idx_in_permuted_row * CHUNK_SIZE + 16, *(__int128_t*)(data + 16));
    }

    // Store the row
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    if (cute::elect_one_sync()) {
        cute::SM90_BULK_COPY_S2G::copy(
            permuted_row_buf,
            params.wv_proj_permuted + wv_group_idx * params.stride_wv_proj_permuted_dim0 + row_idx * params.stride_wv_proj_permuted_dim1,
            row_size
        );
    }
}

void run_permute_wv_proj_kernel(const Params& params) {
    KU_ASSERT(params.d_o == D_O);
    KU_ASSERT(params.input_gran == INPUT_GRAN);
    KU_ASSERT(D_O % (params.input_gran * 4) == 0);
    uint32_t smem_size = 2 * params.wv_group_size * params.d_o;
    KU_CUDA_CHECK(cudaFuncSetAttribute(permute_wv_proj_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    permute_wv_proj_kernel<<<dim3(params.d_proj_out, params.n_wv_group), 32, smem_size, params.stream>>>(params);
}

}
