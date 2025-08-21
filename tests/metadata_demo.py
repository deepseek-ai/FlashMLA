"""
the same calculation as in get_mla_metadata.cu,
but perform in python in CPU no GPU
"""
import os, sys
import math
import torch

def main():
    seqlens_k = [ 3120, 11682,  1833,  2702, 11296,  4037, 13332,  7927,  9362,  8865,
        11318,   820,  5661,  3019,  9568,  3841]
    batch_size = len(seqlens_k)
    fixed_overhead_num_blocks = 5
    block_size_m = 64
    block_size_n = 64
    sm_num = 132
    num_heads_per_head_k = 256
    num_sm_parts = sm_num // math.ceil(num_heads_per_head_k/block_size_m)
    num_heads_k = 1
    window_left=512
    s_q = 4
    
    num_blocks_shared = [0] * batch_size
    num_splits_shared = [0] * (batch_size + 1)
    tile_scheduler_metadata = torch.zeros(num_sm_parts, 8, dtype=torch.int32)

    total_num_blocks = 0
    for i in range(batch_size):
        kvlen = seqlens_k[i]
        qlen = s_q
        end_block_idx = math.ceil(kvlen / block_size_n)
        start_block_idx = 0
        if window_left >= 0:
            start_block_idx = max(kvlen - qlen + 1 - window_left, 0) // block_size_n
        num_blocks = end_block_idx - start_block_idx
        total_num_blocks += num_blocks + fixed_overhead_num_blocks
        num_blocks_shared[i] = num_blocks
        num_splits_shared[i+1] = start_block_idx
    
    payload = max(
        math.ceil(total_num_blocks / num_sm_parts) + fixed_overhead_num_blocks,
        2 * fixed_overhead_num_blocks
    )
    # now_n_split_idx 表示这个seq被切成几段
    # num_splits_shared 是now_n_split_idx的前缀和
    now_idx = 0
    now_n_split_idx = 0
    cum_num_splits = 0
    now_block = 0
    num_splits_shared[0] = 0
    
    for i in range(num_sm_parts):
        tile_scheduler_metadata0 = [0] * 4
        tile_scheduler_metadata0[0] = now_idx  # seq_idx
        start_block_idx = 0 if now_idx >= batch_size else num_splits_shared[now_idx + 1]
        tile_scheduler_metadata0[1] = (start_block_idx + now_block) * block_size_n  # begin_token_idx
        tile_scheduler_metadata1 = now_n_split_idx
        remain_payload = payload
        
        while now_idx < batch_size:
            num_blocks = num_blocks_shared[now_idx]
            now_remain_blocks = num_blocks - now_block
            if remain_payload >= now_remain_blocks + fixed_overhead_num_blocks:
                # 这个seq结束，准备拿下一个seq
                cum_num_splits += now_n_split_idx + 1
                num_splits_shared[now_idx + 1] = cum_num_splits
                remain_payload -= now_remain_blocks + fixed_overhead_num_blocks
                now_idx += 1
                start_block_idx = 0 if now_idx >= batch_size else num_splits_shared[now_idx + 1]
                now_block = 0
                now_n_split_idx = 0
            else:
                # 这个SM已经不够分了
                if remain_payload - fixed_overhead_num_blocks > 0:
                    # 截断这个seq
                    now_block += remain_payload - fixed_overhead_num_blocks
                    now_n_split_idx += 1
                    remain_payload = 0
                # 不值得截断 这个seq不放在这个SM上做 放在下一个SM上
                break
        
        # 截断seq or batch结束
        tile_scheduler_metadata0[2] = now_idx if now_block > 0 else now_idx - 1
        tile_scheduler_metadata0[3] = (now_block+start_block_idx) * block_size_n if now_block > 0 else seqlens_k[now_idx - 1]
        for j in range(4):
            tile_scheduler_metadata[i, j] = tile_scheduler_metadata0[j]
        tile_scheduler_metadata[i, 4] = tile_scheduler_metadata1
        tile_scheduler_metadata[i, 5] = i

    print(tile_scheduler_metadata[:, :6])
    print(num_splits_shared)

if __name__ == "__main__":
    main()
