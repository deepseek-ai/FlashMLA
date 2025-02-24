// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_parameter.h"
#include "flash_run_fwd_template_impl.h"
#include <mctlass/numeric_types.h>

template void run_flash_splitkv_fwd_template<
                576,
                32,
                16,
                2,
                true,
                true,
                cutlass::bfloat16_t,
                true,
                512
            >(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params,cudaStream_t stream);

