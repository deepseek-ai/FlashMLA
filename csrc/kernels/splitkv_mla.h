#pragma once

#include "params.h"

template<typename InputT>
void run_flash_splitkv_mla_kernel(Flash_fwd_mla_params &params, cudaStream_t stream);
