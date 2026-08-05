#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "persistent_sparse_attention/sparse_attention.h"
#ifdef FLASH_MLA_PSA_WITH_PIPELINE
#include "persistent_sparse_attention/pipeline_bindings.h"
#endif

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Persistent sparse-attention kernel suite for FlashMLA nv_dev";

#ifdef FLASH_MLA_PSA_WITH_PIPELINE
    persistent_sparse_attention::register_pipeline_apis(m);
#endif

    m.def(
        "sparse_prefill_fwd",
        &persistent_sparse_attention::sparse_prefill_fwd,
        py::arg("q"),
        py::arg("kv"),
        py::arg("indices"),
        py::arg("sm_scale"),
        py::arg("d_v") = 512,
        py::arg("attn_sink") = std::nullopt,
        py::arg("topk_length") = std::nullopt,
        py::arg("out") = std::nullopt,
        py::arg("num_sms") = std::nullopt,
        py::arg("q_ready") = std::nullopt,
        py::arg("q_ready_chunk_size") = 0,
        py::arg("q_positions") = std::nullopt,
        py::arg("q_cos_sin_cache") = std::nullopt,
        py::arg("q_norm_eps") = 1.0e-6,
        py::arg("o_ready") = std::nullopt,
        py::arg("fused_o_fp8") = std::nullopt,
        py::arg("fused_o_scale") = std::nullopt,
        py::arg("fused_o_positions") = std::nullopt,
        py::arg("fused_o_skip_bf16") = false,
        py::arg("extra_kv") = std::nullopt,
        py::arg("extra_indices") = std::nullopt,
        py::arg("extra_topk_length") = std::nullopt,
        py::arg("psa") = false,
        py::arg("csa_lane") = 0
    );

    m.def(
        "sparse_decode_ready_fwd",
        &persistent_sparse_attention::sparse_decode_ready_fwd,
        py::arg("q"),
        py::arg("kv"),
        py::arg("indices"),
        py::arg("topk_length"),
        py::arg("attn_sink"),
        py::arg("extra_kv"),
        py::arg("extra_indices"),
        py::arg("extra_topk_length"),
        py::arg("d_v"),
        py::arg("sm_scale"),
        py::arg("out"),
        py::arg("num_sm_parts"),
        py::arg("row_ready"),
        py::arg("initial_ready_rows"),
        py::arg("q_positions"),
        py::arg("q_cos_sin_cache"),
        py::arg("q_norm_eps"),
        py::arg("fused_o_fp8"),
        py::arg("fused_o_scale"),
        py::arg("fused_o_positions"),
        py::arg("fused_o_skip_bf16")
    );
}
