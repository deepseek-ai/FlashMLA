// SPDX-License-Identifier: MIT
// Python bindings for the build-time-instantiated CSA adapters.

#include "adapters.h"

#include "logits_out_aot.cpp"
#include "persistent_frontend_aot.cpp"
#include "wqb_ready_aot_host.cpp"

namespace py = pybind11;

namespace flash_mla::csa::deepgemm_adapters {

void register_apis(pybind11::module_& module) {
  module.def(
      "_persistent_attention_frontend", &persistent_attention_frontend,
      py::arg("hidden"), py::arg("hidden_fp8"), py::arg("hidden_sf"),
      py::arg("w_qkv_a"), py::arg("qkv"), py::arg("q_weight"),
      py::arg("kv_weight"), py::arg("q_norm_fp8"), py::arg("q_norm_sf"),
      py::arg("kv_out"), py::arg("w_q_b"), py::arg("q_2d"),
      py::arg("positions"), py::arg("cos_sin_cache"), py::arg("workspace"),
      py::arg("rms_eps") = 1.0e-6f, py::arg("num_sms") = 0);
  module.def(
      "_index_logits_out", &deepgemm_fp8_fp4_mqa_logits_out,
      py::arg("q_fp"), py::arg("q_scale"), py::arg("kv_fp"),
      py::arg("kv_scale"), py::arg("weights"), py::arg("row_starts"),
      py::arg("row_ends"), py::arg("logits_out"),
      py::arg("num_sms") = 0);
  module.def(
      "_wqb_ready_out", &wqb_ready_out,
      py::arg("values"), py::arg("scales"), py::arg("weight"),
      py::arg("weight_scales"), py::arg("output"), py::arg("counters"),
      py::arg("ready"), py::arg("clocks"), py::arg("chunk_size") = 128,
      py::arg("steady_sms") = 0, py::arg("head_tiles") = 4688,
      py::arg("num_sms") = 0);
  module.def(
      "_wqb_ready_packed_out", &wqb_ready_packed_out,
      py::arg("values"), py::arg("packed_scales"), py::arg("weight"),
      py::arg("packed_weight_scales"), py::arg("output"),
      py::arg("counters"), py::arg("ready"), py::arg("clocks"),
      py::arg("chunk_size") = 128, py::arg("steady_sms") = 0,
      py::arg("head_tiles") = 4688, py::arg("num_sms") = 0);
}

}  // namespace flash_mla::csa::deepgemm_adapters
