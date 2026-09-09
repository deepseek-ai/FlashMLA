#pragma once

#include "kernels/params.h"

namespace sm100::decode::sparse::head64 {

struct Config {
    ModelType MODEL_TYPE;         // Format of `kv`: V32, V4 or V41
    ModelType EXTRA_MODEL_TYPE;   // Format of `extra_kv`: MODEL_TYPE, or V41_FP4 with MODEL_TYPE == V41 (see is_valid_kv_format_pair)
    bool ENABLE_SPLITKV;
};

template<Config CONFIG>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

}
