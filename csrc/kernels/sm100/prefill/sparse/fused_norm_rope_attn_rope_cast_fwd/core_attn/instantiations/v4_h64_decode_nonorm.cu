#include "../kernel.h"
#include "../kernel.cuh"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode, ModelType::V4, ModelType::V4, 64, false}>(const ParamT<SparseAttnFwdMode::Decode>& params);

}
