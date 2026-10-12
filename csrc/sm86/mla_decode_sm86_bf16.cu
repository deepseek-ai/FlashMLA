#include "mla_decode_sm86.cuh"

namespace mla_sm86 {
template void mla_sm86_run<__nv_bfloat16>(Flash_fwd_mla_params &, cudaStream_t);
}

#ifdef MLA_SM86_TRACE
void mla_sm86_trace_copy(unsigned long long *host, int n) {
    cudaMemcpyFromSymbol(host, mla_sm86::g_mla_trace, n * sizeof(unsigned long long));
}
void mla_sm86_trace_clear() {
    static unsigned long long zeros[1024 * 8] = {0};
    cudaMemcpyToSymbol(mla_sm86::g_mla_trace, zeros, sizeof(zeros));
}
#endif
