"""Regression for #239 using controlled split-KV intermediate buffers.

Run `python tests/test-split-kv-combine.py` with CUDA PyTorch, Ninja, NVCC,
the CUTLASS submodule initialized, and a GPU with compute capability >= 9.0.
Only the native combine kernel is compiled; no full FlashMLA build is needed.
Compilation uses the production build's --use_fast_math flag.
This does not exercise the public sparse-attention API or the sink path.
"""
import math
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.cpp_extension import load


def main():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("This test requires CUDA PyTorch and a CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError("The unchanged PDL launch requires compute capability >= 9.0")

    root = Path(__file__).resolve().parents[1]
    cutlass = root / "csrc/3rdparty/cutlass"
    if not (cutlass / "include/cutlass/cutlass.h").is_file():
        raise RuntimeError("Initialize csrc/3rdparty/cutlass before running this test")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    os.environ.setdefault("MAX_JOBS", "1")
    # A fresh build directory ensures the binding uses this checkout's kernel.
    with tempfile.TemporaryDirectory(prefix="flashmla-combine-test-") as build:
        module = load(
            name="flashmla_split_kv_combine_test",
            sources=[str(root / "tests/split_kv_combine_binding.cu")],
            build_directory=build,
            extra_include_paths=[str(root), str(root / "csrc"),
                                 str(cutlass / "include"), str(cutlass / "tools/util/include"),
                                 str(root / "csrc/3rdparty/kerutils/include")],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3", "-std=c++20", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=True,
        )
        cases = 0
        for splits in [2, 3, 5, 16]:
            for heads in [1, 8]:
                cumulative = torch.tensor([0, splits], dtype=torch.int32, device="cuda")
                partial = torch.ones((splits, 1, heads, 512), dtype=torch.float32, device="cuda")
                for offset in [0., 16., 1024., 65536., 1048576.]:
                    partial_lse = torch.full((splits, 1, heads), offset, dtype=torch.float32, device="cuda")
                    out, lse = module.combine(partial, partial_lse, cumulative)
                    torch.cuda.synchronize()
                    # A weighted average of identical values must preserve them.
                    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0,
                                               msg=f"{splits=}, {heads=}, {offset=}")
                    expected_lse = (torch.tensor(offset, device="cuda") + math.log2(splits)) / math.log2(math.e)
                    torch.testing.assert_close(lse, expected_lse.expand_as(lse), rtol=1e-6, atol=1e-6)
                    cases += 1

        partial = torch.zeros((3, 1, 8, 512), dtype=torch.float32, device="cuda")
        partial_lse = torch.full((3, 1, 8), -math.inf, dtype=torch.float32, device="cuda")
        cumulative = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
        out, lse = module.combine(partial, partial_lse, cumulative)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
        assert torch.isposinf(lse).all().item()
        print(f"PASS: {cases} constant-output cases and the all-minus-infinity control")


if __name__ == "__main__":
    main()
