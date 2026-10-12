#include "mla_decode_sm86_q.cuh"

namespace mla_sm86 {
template void mla_sm86_run_q<__half>(Flash_fwd_mla_params &, cudaStream_t);
}
