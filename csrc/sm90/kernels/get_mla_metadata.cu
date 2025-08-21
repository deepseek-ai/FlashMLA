#include "get_mla_metadata.h"

#include <cuda_runtime_api.h>
#include <cutlass/fast_math.h>

#include "utils.h"

__global__ void __launch_bounds__(32, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const Mla_metadata_params params) {
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.batch_size;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks; // 5
    int num_sm_parts = params.num_sm_parts;
    int window_left = params.window_left;
    bool is_causal = params.is_causal;
    int qlen = params.s_q;

    extern __shared__ int shared_mem[];
    int* num_blocks_shared = shared_mem; // [batch_size]
    // also set for start_block_idx
    int* num_splits_shared = shared_mem + batch_size; // [batch_size+1]

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32) {
        int kvlen = seqlens_k_ptr[i];
        int end_block_idx = cutlass::ceil_div(kvlen, block_size_n);
        int start_block_idx = 0;
        if (window_left >= 0) {
            int start_token_idx = is_causal ? kvlen - qlen + 1 - window_left : kvlen - window_left;
            start_block_idx = max(start_token_idx, 0) / block_size_n;
        }
        int num_blocks = end_block_idx - start_block_idx;
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
        num_splits_shared[i+1] = start_block_idx;
    }
    // one warp-level reduce
    for (int offset = 16; offset >= 1; offset /= 2) {
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    }
    __syncwarp();

    if (threadIdx.x == 0) {
        int payload = max(
            cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks,
            2*fixed_overhead_num_blocks
        );
        
        int now_idx = 0, now_n_split_idx = 0, cum_num_splits = 0;
        int now_block = 0;
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i) {
            int tile_scheduler_metadata0[4], tile_scheduler_metadata1;
            tile_scheduler_metadata0[0] = now_idx; // seq_idx
            int start_block_idx = now_idx >= batch_size ? 0 : num_splits_shared[now_idx+1];
            tile_scheduler_metadata0[1] = (start_block_idx+now_block) * block_size_n; // begin_token_idx
            tile_scheduler_metadata1 = now_n_split_idx;
            int remain_payload = payload;
            while (now_idx < batch_size) {
                int num_blocks = num_blocks_shared[now_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_idx;
                    start_block_idx = now_idx >= batch_size ? 0 : num_splits_shared[now_idx+1];
                    now_block = 0;
                    now_n_split_idx = 0;
                } else {
                    if (remain_payload - fixed_overhead_num_blocks > 0) {
                        // truncate this seq
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx;
                        remain_payload = 0;
                    }
                    // else: not worth truncating. this seq will not be processed on this SM, but on next SM.
                    break;
                }
            }
            tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1;
            tile_scheduler_metadata0[3] = now_block > 0 ?
                (start_block_idx+now_block) * block_size_n : 
                seqlens_k_ptr[now_idx - 1]
            ;
            *reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize) = *reinterpret_cast<int4 *>(tile_scheduler_metadata0);
            tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
        }
        FLASH_DEVICE_ASSERT(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();

    for (int i = threadIdx.x; i <= batch_size; i += 32) {
        num_splits_ptr[i] = num_splits_shared[i];
    }
}

void run_get_mla_metadata_kernel(Mla_metadata_params &params, cudaStream_t stream) {
    int smem_size = sizeof(int) * (params.batch_size*2+1);
    CHECK_CUDA(cudaFuncSetAttribute(get_mla_metadata_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    get_mla_metadata_kernel<<<1, 32, smem_size, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
