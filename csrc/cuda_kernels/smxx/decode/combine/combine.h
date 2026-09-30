#pragma once

#include "params.h"
#include "cuda_kernels/defines.h"

namespace smxx::decode {

void run_flash_mla_combine_kernel(const SparseAttnDecodeParams &params);

}
