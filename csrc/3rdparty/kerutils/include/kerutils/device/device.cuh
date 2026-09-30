#pragma once

#include "kerutils/common/common.h"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "cuda/common.h"
#include "cuda/sm80/intrinsics.cuh"
#include "cuda/sm80/helpers.cuh"
#include "cuda/sm90/intrinsics.cuh"
#include "cuda/sm90/helpers.cuh"
#include "cuda/sm100/intrinsics.cuh"
#include "cuda/sm100/helpers.cuh"
#include "cuda/sm100/gemm.cuh"
#include "cuda/sm100/tma_cta_group2_nosplit.cuh"
#endif

#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#include "ascend/common.h"
#endif
