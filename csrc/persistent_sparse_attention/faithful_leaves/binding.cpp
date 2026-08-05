#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>

#include "faithful_leaves.h"

namespace py = pybind11;
using torch::Tensor;

namespace {

void check_tensor(const Tensor& value, c10::ScalarType dtype, const char* name) {
  TORCH_CHECK(value.is_cuda() && value.layout() == torch::kStrided,
              name, " must be a strided CUDA tensor");
  TORCH_CHECK(value.scalar_type() == dtype, name, " has the wrong dtype");
}

void check_same_device(const Tensor& reference,
                       std::initializer_list<const Tensor*> values) {
  for (const Tensor* value : values) {
    TORCH_CHECK(value->device() == reference.device(),
                "all faithful leaf tensors must be on one CUDA device");
  }
}

cudaStream_t stream_for(const Tensor& value) {
  return at::cuda::getCurrentCUDAStream(value.get_device()).stream();
}

void check_faithful_grid(const Tensor& value, int64_t grid,
                         const char* operation) {
  const auto* properties = at::cuda::getDeviceProperties(value.get_device());
  TORCH_CHECK(properties->major == 10, operation, " requires an sm_100 GPU");
  TORCH_CHECK(grid > 0, operation, " grid must be positive");
  TORCH_CHECK(grid <= properties->multiProcessorCount,
              operation, " grid exceeds physical SM count");
}

void check_launch(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, " launch failed: ",
              cudaGetErrorString(status));
}

void index_query_quant(const Tensor& positions, const Tensor& index_query,
                       const Tensor& cos_sin, const Tensor& packed,
                       const Tensor& scales, int64_t num_sms) {
  check_tensor(positions, torch::kInt64, "positions");
  check_tensor(index_query, torch::kBFloat16, "index_query");
  check_tensor(cos_sin, torch::kFloat32, "cos_sin_cache");
  check_tensor(packed, torch::kUInt8, "packed");
  check_tensor(scales, torch::kUInt8, "scales");
  const int64_t tokens = index_query.size(0);
  TORCH_CHECK(index_query.dim() == 3 && index_query.size(1) == 64 &&
                  index_query.size(2) == 128 && index_query.is_contiguous(),
              "index_query must be contiguous BF16 [tokens,64,128]");
  TORCH_CHECK(positions.dim() == 1 && positions.numel() == tokens &&
                  positions.is_contiguous(),
              "positions must be contiguous int64 [tokens]");
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) >= 64 &&
                  cos_sin.stride(1) == 1,
              "cos_sin_cache must be float32 [positions,>=64]");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({tokens, 64, 64}) &&
                  packed.is_contiguous(),
              "packed must be contiguous uint8 [tokens,64,64]");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({tokens, 64, 4}) &&
                  scales.is_contiguous(),
              "scales must be contiguous uint8 [tokens,64,4]");
  TORCH_CHECK(tokens <= std::numeric_limits<uint32_t>::max(),
              "token count exceeds uint32");
  check_same_device(index_query, {&positions, &cos_sin, &packed, &scales});
  TORCH_CHECK(!index_query.is_alias_of(packed) &&
                  !index_query.is_alias_of(scales) &&
                  !packed.is_alias_of(scales),
              "index query quant inputs and outputs must not alias");
  check_faithful_grid(index_query, num_sms, "index_query_quant");
  c10::cuda::CUDAGuard guard(index_query.device());
  check_launch(launch_index_query_quant(
                   positions.const_data_ptr<int64_t>(),
                   index_query.const_data_ptr(), cos_sin.const_data_ptr<float>(),
                   packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
                   static_cast<uint32_t>(tokens), static_cast<uint32_t>(num_sms),
                   stream_for(index_query)),
               "index_query_quant");
}

void index_weight_scale(const Tensor& input, const Tensor& output,
                        double softmax_scale, double head_scale,
                        int64_t num_sms) {
  check_tensor(input, torch::kBFloat16, "index_weights");
  check_tensor(output, torch::kFloat32, "weights_out");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous() &&
                  input.sizes() == output.sizes(),
              "index weights must be contiguous with matching shapes");
  TORCH_CHECK(input.numel() <= std::numeric_limits<uint32_t>::max(),
              "weight count exceeds uint32");
  check_same_device(input, {&output});
  TORCH_CHECK(!input.is_alias_of(output), "index weight buffers must not alias");
  check_faithful_grid(input, num_sms, "index_weight_scale");
  c10::cuda::CUDAGuard guard(input.device());
  check_launch(launch_index_weight_scale(
                   input.const_data_ptr(), output.data_ptr<float>(),
                   static_cast<uint32_t>(input.numel()),
                   static_cast<float>(softmax_scale),
                   static_cast<float>(head_scale),
                   static_cast<uint32_t>(num_sms), stream_for(input)),
               "index_weight_scale");
}

void index_cache_pack(const Tensor& source, const Tensor& values,
                      const Tensor& scales, int64_t num_rows,
                      int64_t source_row_offset, int64_t destination_row_offset,
                      int64_t num_sms) {
  check_tensor(source, torch::kUInt8, "index_cache");
  check_tensor(values, torch::kUInt8, "index_k_values");
  check_tensor(scales, torch::kInt32, "index_k_scales");
  TORCH_CHECK(source.dim() == 2 && source.stride(1) == 1 &&
                  source.stride(0) >= 64 * (64 + 4),
              "index_cache must be uint8 [pages,>=4352]");
  TORCH_CHECK(values.dim() == 2 && values.size(1) >= 64 &&
                  values.stride(1) == 1 && values.stride(0) >= 64,
              "index_k_values must be uint8 [rows,>=64]");
  TORCH_CHECK(scales.dim() == 1 && scales.stride(0) > 0,
              "index_k_scales must be int32 [rows]");
  TORCH_CHECK(num_rows >= 0 && source_row_offset >= 0 &&
                  destination_row_offset >= 0,
              "row counts and offsets must be non-negative");
  TORCH_CHECK(source_row_offset + num_rows <= source.size(0) * 64 &&
                  destination_row_offset + num_rows <= values.size(0) &&
                  destination_row_offset + num_rows <= scales.size(0),
              "index cache pack range exceeds a source or output surface");
  TORCH_CHECK(source_row_offset + num_rows <=
                  std::numeric_limits<uint32_t>::max() &&
                  destination_row_offset + num_rows <=
                  std::numeric_limits<uint32_t>::max(),
              "index cache pack range exceeds uint32");
  check_same_device(source, {&values, &scales});
  TORCH_CHECK(!source.is_alias_of(values) && !source.is_alias_of(scales) &&
                  !values.is_alias_of(scales),
              "index cache pack surfaces must not alias");
  check_faithful_grid(source, num_sms, "index_cache_pack");
  const bool vectorized =
      (reinterpret_cast<std::uintptr_t>(source.data_ptr()) & 15u) == 0 &&
      (reinterpret_cast<std::uintptr_t>(values.data_ptr()) & 15u) == 0 &&
      (reinterpret_cast<std::uintptr_t>(scales.data_ptr()) & 3u) == 0 &&
      source.stride(0) % 16 == 0 && values.stride(0) % 16 == 0;
  c10::cuda::CUDAGuard guard(source.device());
  check_launch(launch_index_cache_pack(
                   source.const_data_ptr<uint8_t>(), values.data_ptr<uint8_t>(),
                   scales.data_ptr<int32_t>(), source.stride(0), source.stride(1),
                   values.stride(0), values.stride(1), scales.stride(0),
                   static_cast<uint32_t>(source_row_offset),
                   static_cast<uint32_t>(num_rows),
                   static_cast<uint32_t>(destination_row_offset),
                   static_cast<uint32_t>(num_sms), vectorized, stream_for(source)),
               "index_cache_pack");
}

void kv_cache_gather_persistent(
    const Tensor& output, const Tensor& cache, const Tensor& seq_lens,
    const std::optional<Tensor>& gather_lens, const Tensor& block_table,
    int64_t cache_block_size, int64_t output_offset,
    const std::optional<Tensor>& block_table_base_offsets, int64_t num_sms) {
  check_tensor(output, torch::kBFloat16, "output");
  check_tensor(cache, torch::kUInt8, "cache");
  check_tensor(seq_lens, torch::kInt32, "seq_lens");
  check_tensor(block_table, torch::kInt32, "block_table");
  const int64_t requests = seq_lens.numel();
  TORCH_CHECK(output.dim() == 3 && output.size(0) == requests &&
                  output.size(2) == 512 && output.stride(2) == 1,
              "output must be BF16 [requests,rows,512]");
  TORCH_CHECK(cache.dim() == 2 && cache.stride(1) == 1 &&
                  cache.stride(0) >= cache_block_size * 584,
              "cache pages must provide 584 bytes per token");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.stride(0) > 0,
              "seq_lens must be int32 [requests]");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == requests &&
                  block_table.stride(1) > 0,
              "block_table must be int32 [requests,max_blocks]");
  TORCH_CHECK(cache_block_size > 0 &&
                  cache_block_size <= std::numeric_limits<int32_t>::max() &&
                  output_offset >= 0 && output_offset <= output.size(1),
              "cache block size or output offset is invalid");
  if (gather_lens) {
    check_tensor(*gather_lens, torch::kInt32, "gather_lens");
    TORCH_CHECK(gather_lens->dim() == 1 && gather_lens->numel() == requests &&
                    gather_lens->stride(0) > 0,
                "gather_lens must be int32 [requests]");
  }
  if (block_table_base_offsets) {
    check_tensor(*block_table_base_offsets, torch::kInt32,
                 "block_table_base_offsets");
    TORCH_CHECK(block_table_base_offsets->dim() == 1 &&
                    block_table_base_offsets->numel() == requests &&
                    block_table_base_offsets->stride(0) > 0,
                "block_table_base_offsets must be int32 [requests]");
  }
  check_same_device(output, {&cache, &seq_lens, &block_table});
  if (gather_lens) check_same_device(output, {&*gather_lens});
  if (block_table_base_offsets)
    check_same_device(output, {&*block_table_base_offsets});
  TORCH_CHECK(!output.is_alias_of(cache), "gather output and cache must not alias");
  TORCH_CHECK(requests <= std::numeric_limits<int32_t>::max() &&
                  block_table.size(1) <= std::numeric_limits<int32_t>::max(),
              "gather dimensions exceed int32");
  check_faithful_grid(output, num_sms, "kv_cache_gather_persistent");
  c10::cuda::CUDAGuard guard(output.device());
  check_launch(launch_kv_cache_gather(
                   output.data_ptr(), cache.const_data_ptr<uint8_t>(),
                   seq_lens.const_data_ptr<int32_t>(),
                   gather_lens ? gather_lens->const_data_ptr<int32_t>() : nullptr,
                   block_table.const_data_ptr<int32_t>(),
                   block_table_base_offsets
                       ? block_table_base_offsets->const_data_ptr<int32_t>()
                       : nullptr,
                   output.stride(0), output.stride(1), cache.stride(0),
                   seq_lens.stride(0), gather_lens ? gather_lens->stride(0) : 0,
                   block_table.stride(0), block_table.stride(1),
                   block_table_base_offsets
                       ? block_table_base_offsets->stride(0)
                       : 0,
                   static_cast<int32_t>(requests),
                   static_cast<int32_t>(block_table.size(1)),
                   static_cast<int32_t>(cache_block_size),
                   static_cast<int32_t>(output_offset),
                   static_cast<uint32_t>(num_sms), stream_for(output)),
               "kv_cache_gather_persistent");
}

}  // namespace

void register_faithful_leaves(py::module_& module) {
  module.def("index_query_quant", &index_query_quant, py::arg("positions"),
             py::arg("index_query"), py::arg("cos_sin_cache"),
             py::arg("packed"), py::arg("scales"), py::arg("num_sms") = 104);
  module.def("index_weight_scale", &index_weight_scale,
             py::arg("index_weights"), py::arg("weights_out"),
             py::arg("softmax_scale"), py::arg("head_scale"),
             py::arg("num_sms") = 44);
  module.def("index_cache_pack", &index_cache_pack, py::arg("index_cache"),
             py::arg("index_k_values"), py::arg("index_k_scales"),
             py::arg("num_rows"), py::arg("source_row_offset") = 0,
             py::arg("destination_row_offset") = 0,
             py::arg("num_sms") = 132);
  module.def("kv_cache_gather_persistent", &kv_cache_gather_persistent,
             py::arg("output"), py::arg("cache"), py::arg("seq_lens"),
             py::arg("gather_lens"), py::arg("block_table"),
             py::arg("cache_block_size"), py::arg("output_offset") = 0,
             py::arg("block_table_base_offsets") = py::none(),
             py::arg("num_sms") = 16);
}
