#pragma once

#include "flash_mla.h"

void run_get_mla_metadata_kernel(Mla_metadata_params &params, cudaStream_t stream);
