#pragma once

#include "params.h"
#include "cuda_kernels/defines.h"

namespace sm100::decode::sparse::head64 {

struct Config {
    ModelType MODEL_TYPE;         // Format of `kv`: V41
    ModelType EXTRA_MODEL_TYPE;   // Format of `extra_kv`
    bool ENABLE_SPLITKV;
};

template<Config CONFIG>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

}
