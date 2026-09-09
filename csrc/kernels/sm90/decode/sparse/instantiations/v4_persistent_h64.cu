#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V4, 64>(const SparseAttnDecodeParams &params);

}

