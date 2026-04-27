#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def(
        "sparse_prefill_fwd",
        &sparse_attn_prefill_interface,
        pybind11::arg("q"),
        pybind11::arg("kv"),
        pybind11::arg("indices"),
        pybind11::arg("sm_scale"),
        pybind11::arg("d_v"),
        pybind11::arg("attn_sink"),
        pybind11::arg("topk_length"),
        pybind11::arg("indexer_topk") = 0
    );
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
