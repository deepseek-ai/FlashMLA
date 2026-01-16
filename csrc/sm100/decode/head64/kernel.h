#pragma once

#include "params.h"

namespace sm100::decode::head64 {

template<ModelType MODEL_TYPE>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

}

