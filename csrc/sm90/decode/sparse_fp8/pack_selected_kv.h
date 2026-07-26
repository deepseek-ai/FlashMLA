#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace sm90::decode::sparse_fp8 {

struct PackSelectedKvParams {
    const uint8_t* src;
    const int* indices;
    const int* topk_length;
    uint8_t* dst;

    int num_source_tokens;
    int page_block_size;
    int64_t source_block_stride;
    int64_t source_row_stride;
    int batch_size;
    int query_len;
    int topk;
    cudaStream_t stream;
};

void run_pack_selected_kv_kernel(const PackSelectedKvParams &params);

}
