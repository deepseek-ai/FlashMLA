#include "../kernel.h"
#include "../kernel.cuh"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode, ModelType::V41, ModelType::V41_FP4, 128, false}>(const ParamT<SparseAttnFwdMode::Decode>& params);

}
