#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse_fp8 {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 64>(const SparseAttnDecodeParams &params);

}
