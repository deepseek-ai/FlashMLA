#include <pybind11/pybind11.h>

void register_sparse_prefill(pybind11::module_& m);
void register_sparse_decode(pybind11::module_& m);
void register_dense_fwd(pybind11::module_& m);
void register_dense_bwd(pybind11::module_& m);
void register_dense_decode(pybind11::module_& m);
void register_fused_norm_rope_attn_rope_cast_fwd(pybind11::module_& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    register_sparse_prefill(m);
    register_sparse_decode(m);
    register_dense_fwd(m);
    register_dense_bwd(m);
    register_dense_decode(m);
    register_fused_norm_rope_attn_rope_cast_fwd(m);
}
