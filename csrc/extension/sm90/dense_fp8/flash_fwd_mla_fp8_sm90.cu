/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

#include "flash_fwd_mla_kernel.h"

#ifndef FLASH_MLA_DISABLE_FP8
template void run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::bfloat16_t, 576>(DecodingParams_fp8 &params, cudaStream_t stream);
#endif