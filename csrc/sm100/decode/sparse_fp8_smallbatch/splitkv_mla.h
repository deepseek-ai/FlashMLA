#pragma once

#include "deps/params_compat.h"

namespace nv_smallbatch::sm100 {

void run_flash_splitkv_mla_fp8_sparse_kernel(const DecodingParamsCompat &params);

}  // namespace nv_smallbatch::sm100
