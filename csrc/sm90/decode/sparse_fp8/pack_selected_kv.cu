#include "pack_selected_kv.h"

#include <cuda_runtime.h>

#include <kerutils/kerutils.cuh>

namespace sm90::decode::sparse_fp8 {

namespace {

constexpr int kRecordBytes = 656;
constexpr int kVectorBytes = 16;
constexpr int kVectorsPerRecord = kRecordBytes / kVectorBytes;
constexpr int kTokensPerTile = 64;
constexpr int kVectorsPerCta = 2;
constexpr int kVectorGroups =
    (kVectorsPerRecord + kVectorsPerCta - 1) / kVectorsPerCta;
constexpr int kThreads = kTokensPerTile * kVectorsPerCta;

__device__ __forceinline__ uint4 load_128b_cg(const uint4* ptr) {
    uint4 value;
    asm volatile(
        "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(ptr)
    );
    return value;
}

__global__ void __launch_bounds__(kThreads)
pack_selected_kv_kernel(PackSelectedKvParams params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const int packed_page = blockIdx.x / kVectorGroups;
    const int vector_group = blockIdx.x % kVectorGroups;
    const int token_in_tile = threadIdx.x % kTokensPerTile;
    const int vector_in_group = threadIdx.x / kTokensPerTile;
    const int vector_index =
        vector_group * kVectorsPerCta + vector_in_group;

    if (vector_index >= kVectorsPerRecord) {
        return;
    }

    const int tiles_per_row = params.topk / kTokensPerTile;
    const int row = packed_page / tiles_per_row;
    const int tile_in_row = packed_page % tiles_per_row;
    const int batch_idx = row / params.query_len;
    const int selected_offset =
        tile_in_row * kTokensPerTile + token_in_tile;
    const int selected_idx = row * params.topk + selected_offset;
    const int active_topk = params.topk_length
        ? __ldg(params.topk_length + batch_idx)
        : params.topk;
    const int source_token = selected_offset < active_topk
        ? __ldg(params.indices + selected_idx)
        : -1;
    const bool valid =
        source_token >= 0 && source_token < params.num_source_tokens;

    uint4 value = {};
    if (valid) {
        const int source_block = source_token / params.page_block_size;
        const int source_row = source_token % params.page_block_size;
        value = load_128b_cg(reinterpret_cast<const uint4*>(
            params.src
            + source_block * params.source_block_stride
            + source_row * params.source_row_stride
            + vector_index * kVectorBytes
        ));
    }
    *reinterpret_cast<uint4*>(
        params.dst
        + (
            (packed_page * kVectorsPerRecord + vector_index)
                * kTokensPerTile
            + token_in_tile
        ) * kVectorBytes
    ) = value;
#endif
}

}

void run_pack_selected_kv_kernel(const PackSelectedKvParams &params) {
    const int packed_pages =
        params.batch_size * params.query_len
        * (params.topk / kTokensPerTile);
    pack_selected_kv_kernel<<<
        packed_pages * kVectorGroups,
        kThreads,
        0,
        params.stream
    >>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

}
