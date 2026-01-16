#include "../kernel.cuh"

namespace sm100::decode::head64 {

template
void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32>(const SparseAttnDecodeParams &params);

}
