#pragma once

#include "params.h"

namespace sm100::decode::head64_bf16 {

template<ModelType MODEL_TYPE>
void run_flash_splitkv_mla_bf16_sparse_kernel(const SparseAttnDecodeParams &params);

}

