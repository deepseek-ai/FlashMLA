#include "../kernel.cuh"

namespace sm100::decode::sparse::head64 {

template
void run_flash_splitkv_mla_fp8_sparse_kernel<Config{ModelType::V32, ModelType::V32, true}>(const SparseAttnDecodeParams &params);

}
