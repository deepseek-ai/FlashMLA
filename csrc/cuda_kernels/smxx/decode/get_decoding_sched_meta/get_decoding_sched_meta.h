#pragma once

#include "params.h"
#include "cuda_kernels/defines.h"

namespace smxx::decode {

void run_get_decoding_sched_meta_kernel(const GetDecodeSchedMetaParams& params);

}
