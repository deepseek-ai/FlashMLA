#pragma once

#include "params.h"

namespace sm90 {

template<typename InputT, typename OutputT = InputT>
void run_flash_splitkv_mla_kernel(DecodingParams &params, cudaStream_t stream);

}
