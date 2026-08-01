#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("sparse_prefill_fwd", [](
        const at::Tensor &q,
        const at::Tensor &kv,
        const at::Tensor &indices,
        float sm_scale,
        int d_v,
        const at::Tensor &attn_sink,
        const at::Tensor &topk_length) {
        return sparse_attn_prefill_interface(
            q, kv, indices, sm_scale, d_v, attn_sink, topk_length);
    });
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
