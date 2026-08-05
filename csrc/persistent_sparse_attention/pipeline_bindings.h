#pragma once

#include <pybind11/pybind11.h>

namespace persistent_sparse_attention {

void register_pipeline_apis(pybind11::module_& module);

}  // namespace persistent_sparse_attention
