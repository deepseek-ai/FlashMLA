#pragma once

#include "kernels/params.h"

namespace sm90::decode::dense {

template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params);

}
