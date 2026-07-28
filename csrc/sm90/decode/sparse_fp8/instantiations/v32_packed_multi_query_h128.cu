#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse_fp8 {

template void run_flash_splitkv_mla_fp8_packed_multi_query_kernel<
    ModelType::V32,
    128
>(const SparseAttnDecodeParams &params);

}
