#include "../kernel.cuh"

namespace sm100::decode::head64_bf16 {
template void run_flash_splitkv_mla_bf16_sparse_kernel<ModelType::MODEL1>(const SparseAttnDecodeParams &params);
}
