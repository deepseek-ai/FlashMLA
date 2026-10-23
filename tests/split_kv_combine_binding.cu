#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define FLASH_MLA_IS_BUILD_ON_CUDA
#define KERUTILS_IS_BUILD_ON_CUDA
#include "csrc/cuda_kernels/smxx/decode/combine/combine.cu"

std::vector<torch::Tensor> combine(torch::Tensor partial_out,
                                   torch::Tensor partial_lse,
                                   torch::Tensor splits) {
  TORCH_CHECK(partial_out.is_cuda() && partial_lse.is_cuda() && splits.is_cuda());
  TORCH_CHECK(partial_out.is_contiguous() && partial_lse.is_contiguous() && splits.is_contiguous());
  TORCH_CHECK(partial_out.scalar_type() == torch::kFloat32 && partial_lse.scalar_type() == torch::kFloat32);
  TORCH_CHECK(splits.scalar_type() == torch::kInt32);
  TORCH_CHECK(partial_out.dim() == 4 && partial_out.size(1) == 1 && partial_out.size(3) == 512);
  TORCH_CHECK(partial_lse.sizes() == partial_out.sizes().slice(0, 3));
  TORCH_CHECK(splits.numel() == 2 && partial_out.size(0) >= 2 && partial_out.size(0) <= 32);
  c10::cuda::CUDAGuard guard(partial_out.device());
  TORCH_CHECK(partial_lse.device() == partial_out.device() && splits.device() == partial_out.device());
  const int n = partial_out.size(0), h = partial_out.size(2);
  TORCH_CHECK(h > 0);
  TORCH_CHECK(splits[0].item<int>() == 0 && splits[1].item<int>() == n);
  auto out = torch::zeros({1, 1, h, 512}, partial_out.options().dtype(torch::kBFloat16));
  auto lse = torch::zeros({1, 1, h}, partial_lse.options());
  SparseAttnDecodeParams params{};
  params.b = 1; params.s_q = 1; params.h_q = h; params.h_kv = 1;
  params.d_qk = 512; params.d_v = 512; params.enable_split_kv = true;
  params.o_accum = partial_out.data_ptr<float>();
  params.lse_accum = partial_lse.data_ptr<float>();
  params.num_splits_ptr = splits.data_ptr<int>();
  params.num_sm_parts = n;
  params.out = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
  params.lse = lse.data_ptr<float>();
  params.attn_sink = nullptr;
  params.stride_o_b = out.stride(0); params.stride_o_s_q = out.stride(1); params.stride_o_h_q = out.stride(2);
  params.stride_lse_b = lse.stride(0); params.stride_lse_s_q = lse.stride(1);
  params.stride_o_accum_split = partial_out.stride(0);
  params.stride_o_accum_s_q = partial_out.stride(1);
  params.stride_o_accum_h_q = partial_out.stride(2);
  params.stride_lse_accum_split = partial_lse.stride(0);
  params.stride_lse_accum_s_q = partial_lse.stride(1);
  params.stream = c10::cuda::getCurrentCUDAStream().stream();
  smxx::decode::run_flash_mla_combine_kernel(params);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
  return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("combine", &combine); }
