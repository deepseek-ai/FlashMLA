#include "get_decoding_sched_meta.h"

#include <cuda_runtime_api.h>
#include <cutlass/fast_math.h>
#include <kerutils/kerutils.cuh>

#include "utils.h"

namespace smxx::decode {

__global__ void __launch_bounds__(32, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const GetDecodeSchedMetaParams params) {
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    DecodingSchedMeta *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.b;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    extern __shared__ int shared_mem[];
    int* num_blocks_shared = shared_mem; // [batch_size]
    int* num_splits_shared = shared_mem + batch_size; // [batch_size+1]
    int* seqlens_k_shared = shared_mem + batch_size*2+1; // [batch_size]
    int* first_block_idx_shared = shared_mem + batch_size*3+1; // [batch_size]
    int* last_block_idx_shared = shared_mem + batch_size*4+1; // [batch_size]

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32) {
        int cur_s_k;
        if (params.topk == -1) {
            // Dense model, cur_s_k = actual s_k
            cur_s_k = __ldg(seqlens_k_ptr + i);
        } else {
            // Sparse model, cur_s_k = topk (+ extra topk)
            cur_s_k = params.topk_length ? __ldg(params.topk_length + i) : params.topk;
            if (cur_s_k == 0) cur_s_k = 1;  // Ensure the main loop will never be empty
            if (params.extra_topk) {
                cur_s_k = ku::ceil(cur_s_k, block_size_n);
                cur_s_k += params.extra_topk_length ? __ldg(params.extra_topk_length + i) : params.extra_topk;
            }
        }
        seqlens_k_shared[i] = cur_s_k;
        int first_token_idx = 0;
        int last_token_idx = max(cur_s_k-1, 0);
        int cur_first_block_idx = first_token_idx / block_size_n;
        int cur_last_block_idx = last_token_idx / block_size_n;
        // NOTE Should attend to tokens [first_token_idx, last_token_idx], i.e. blocks [cur_first_block_idx, cur_last_block_idx]
        // NOTE if seqlens_k is 0, then first_token_idx == last_token_idx == cur_first_block_idx == cur_last_block_idx == 0. So the sequence will have 1 block. We will correct this later in this kernel.
        int num_blocks = cur_last_block_idx - cur_first_block_idx + 1;
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
        first_block_idx_shared[i] = cur_first_block_idx;
        last_block_idx_shared[i] = cur_last_block_idx;
    }
    for (int offset = 16; offset >= 1; offset /= 2) {
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    }
    __syncwarp();

    if (threadIdx.x == 0) {
        int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;

        int now_req_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i) {
            DecodingSchedMeta cur_meta;
            cur_meta.begin_req_idx = now_req_idx;
            cur_meta.begin_block_idx = now_block + first_block_idx_shared[now_req_idx];
            cur_meta.begin_split_idx = now_n_split_idx;
            cur_meta.is_first_req_splitted = (now_block != 0);
            int remain_payload = payload;
            while (now_req_idx < batch_size) {
                int num_blocks = num_blocks_shared[now_req_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_req_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_req_idx;
                    now_block = 0;
                    now_n_split_idx = 0;
                } else {
                    if (remain_payload - fixed_overhead_num_blocks > 0) {
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx;
                        remain_payload = 0;
                    }
                    break;
                }
            }
            cur_meta.end_req_idx = now_block > 0 ? now_req_idx : now_req_idx - 1;
            cur_meta.end_block_idx = now_block > 0 ? now_block + first_block_idx_shared[now_req_idx] : (seqlens_k_shared[now_req_idx-1] == 0 ? 0 : last_block_idx_shared[now_req_idx-1] + 1);
            cur_meta.is_last_req_splitted = cur_meta.end_block_idx != last_block_idx_shared[cur_meta.end_req_idx] + 1 && seqlens_k_shared[cur_meta.end_req_idx] != 0;
            if (cur_meta.begin_req_idx == cur_meta.end_req_idx) {
                cur_meta.is_first_req_splitted = cur_meta.is_last_req_splitted = cur_meta.is_first_req_splitted || cur_meta.is_last_req_splitted;
            }
            tile_scheduler_metadata_ptr[i] = cur_meta;
        }
        FLASH_DEVICE_ASSERT(now_req_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();

    for (int i = threadIdx.x; i <= batch_size; i += 32) {
        num_splits_ptr[i] = num_splits_shared[i];
    }
}

void run_get_decoding_sched_meta_kernel(GetDecodeSchedMetaParams &params) {
    int smem_size = sizeof(int) * (params.b*5+1);
    CHECK_CUDA(cudaFuncSetAttribute(get_mla_metadata_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    get_mla_metadata_kernel<<<1, 32, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
