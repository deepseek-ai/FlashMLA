import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, find_packages

THIS_DIR = Path(__file__).resolve().parent

# The pybind11 API of the extension. The fused kernels live in their own translation unit.
API_SOURCES = [
    "csrc/api/api.cpp",
    "csrc/api/sparse_prefill.cpp",
    "csrc/api/sparse_decode.cpp",
    "csrc/api/dense_fwd.cpp",
    "csrc/api/dense_bwd.cpp",
    "csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp",
]

# The extension modules, one per build platform
EXT_NAME_CUDA = "flash_mla.cuda"
EXT_NAME_ASCEND = "flash_mla.npu"

# `-gencode` targets. We use the architecture-specific ones (sm_100a / sm_103a) instead of the
# family-specific one (sm_100f) because the kernels use `cvt` with `.scaled::n2::ue8m0`, and
# sm_103a additionally enables better code generation for `exp`.
CUDA_ARCHS = ["sm_100a", "sm_103a"]

# Head counts (H_Q) the fused core_attn kernel is instantiated for: 128 (2-CTA cluster) and 64
HEADS_FOR_FUSED_CORE_ATTN = (64, 128)

CUDA_KERNEL_SOURCES = [
    # sm100 dense prefill (cutlass FMHA): forward and backward
    "csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",
    "csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",

    # Small kernels for decoding: tile scheduler metadata generation and the split-KV combine
    "csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "csrc/cuda_kernels/smxx/decode/combine/combine.cu",

    # sm100 sparse prefill
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k512.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu",

    # sm100 sparse prefill for small topk, and head128 decoding
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_k512.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41_splitkv.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4_splitkv.cu",

    # sm100 sparse decode (head 64, with and without split-KV, on the fp8 and the fp4 cache)
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64_no_split.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64_no_split.cu",

    # Fused Norm + RoPE + Core Attn + RoPE + Cast: 6 instantiations per head count
    # (V4.1 fp8 prefill, V4.1 fp8 decode and V4.1 fp4 decode, each with and without q norm)
    *[f"csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/{model}_h{h}_{mode}_{norm}.cu"
      for h in HEADS_FOR_FUSED_CORE_ATTN
      for model, mode in (("v41", "prefill"), ("v41", "decode"), ("v41fp4", "decode"))
      for norm in ("norm", "nonorm")],
    "csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.cu",
]

# The Ascend instantiations are explicit template instantiations of `run_sparse_fwd_kernel`.
# Kept for: h64 x {prefill, decode} x {V4.1 fp8, V4.1 fp4 (decode only)} x {sink, nosink}
ASCEND_KERNEL_SOURCES = [
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41_h64_prefill_nosink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41_h64_prefill_sink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41_h64_decode_nosink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41_h64_decode_sink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41fp4_h64_decode_nosink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41fp4_h64_decode_sink.asc",
]

def get_cuda_arch_flags():
    arch_flags = []
    for arch in CUDA_ARCHS:
        compute_arch = arch.removeprefix("sm_")
        arch_flags.extend(["-gencode", f"arch=compute_{compute_arch},code={arch}"])
    return arch_flags


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


def build_on_cuda_platform():
    from torch.utils.cpp_extension import (
        BuildExtension,
        CUDAExtension,
        CUDA_HOME,
    )

    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable.
    # For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    nvcc_version_number = nvcc_version.split("release ")[1].split(",")[0].strip()
    major, minor = map(int, nvcc_version_number.split("."))
    print(f"Compiling using NVCC {major}.{minor}")
    assert major > 13 or (major == 13 and minor >= 1), \
        "Compiling the kernels requires NVCC 13.1 or higher (cvt with .scaled::n2::ue8m0)."

    class BuildExtensionWithPostActions(BuildExtension):
        """
        Run the register-spill check after building.
        """
        def run(self):
            BuildExtension.run(self)

            if not self.dry_run:
                from tests.kernelkit import check_kernel_reg_spill_in_artifact

                so_paths = [
                    os.path.join(self.build_lib, self.get_ext_filename(EXT_NAME_CUDA)),
                    self.get_ext_fullpath(EXT_NAME_CUDA),    # May point to the extension in the source tree for editable installs
                ]
                so_paths = list(set(so_paths))  # deduplicate
                assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
                for so_path in so_paths:
                    spilled_kernels = check_kernel_reg_spill_in_artifact(
                        so_path, quiet=True, suppress_checking_env_var="FLASH_MLA_SKIP_REG_SPILL_CHECK"
                    )
                    # The restored upstream CUTLASS FMHA kernels (cutlass::device_kernel<...>) keep
                    # their stack frames, so they are exempt. Only the kernels defined by this
                    # repository have to be spill-free.
                    own_spilled_kernels = [name for name in spilled_kernels if not name.startswith("_ZN7cutlass")]
                    num_exempt_kernels = len(spilled_kernels) - len(own_spilled_kernels)
                    if num_exempt_kernels > 0:
                        print(f"{num_exempt_kernels} upstream CUTLASS kernel(s) keep their stack frames (exempt)")
                    if len(own_spilled_kernels) > 0:
                        for kernel_name in own_spilled_kernels:
                            print(f"  {kernel_name}")
                        print("Register spilling detected. Build failed!")
                        sys.exit(1)

    ext_modules = [
        CUDAExtension(
            name=EXT_NAME_CUDA,
            sources=[
                *API_SOURCES,
                *CUDA_KERNEL_SOURCES,
            ],
            extra_compile_args={
                # The host-side (`.cpp`) translation units are not compiled by nvcc, so `__CUDACC__`
                # is not defined and neither kerutils' nor our own platform switch is set automatically.
                "cxx": [
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-Wno-deprecated-declarations",
                    "-DFLASH_MLA_IS_BUILD_ON_CUDA",
                    "-DKERUTILS_IS_BUILD_ON_CUDA",
                ],
                "nvcc": get_nvcc_thread_args() + [
                    "-O3",
                    "-std=c++20",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use",
                    "-lineinfo",
                    "--source-in-ptx",
                    "--diag-suppress=445",  # "constant "N" is not used in declaring the parameter types of function template"
                    "-DNDEBUG",
                    "-DFLASH_MLA_IS_BUILD_ON_CUDA",
                ] + get_cuda_arch_flags(),
            },
            include_dirs=[
                THIS_DIR / "csrc",
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "tools" / "util" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "kerutils" / "include",
                Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cccl",    # for cuda/std headers in CUDA 13+
                Path(CUDA_HOME) / "targets" / "sbsa-linux" / "include" / "cccl",
            ],
            # On a machine without the NVIDIA driver (e.g. a CPU-only build node) `-lcuda` needs
            # the stub libraries that ship with the toolkit
            extra_link_args=[
                f"-L{Path(CUDA_HOME) / 'targets' / 'x86_64-linux' / 'lib' / 'stubs'}",
                f"-L{Path(CUDA_HOME) / 'targets' / 'sbsa-linux' / 'lib' / 'stubs'}",
                "-lcuda",
            ],
        )
    ]

    return ext_modules, BuildExtensionWithPostActions


def build_on_ascend_platform():
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    import torch_npu

    arch = os.environ.get("ASCEND_NPU_ARCH", "dav-3510")
    asc_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    if not os.path.isdir(asc_home):
        raise RuntimeError(
            f"Ascend toolkit not found at '{asc_home}'. "
            "Please set ASCEND_HOME_PATH to the root of your CANN installation "
            "(e.g. /usr/local/Ascend/cann-9.1.0)."
        )

    class AscendBuildExtension(BuildExtension):
        def build_extensions(self):
            self.compiler.src_extensions += [".asc"]
            os.environ["CXX"] = "bisheng"
            os.environ["CC"] = "bisheng"
            super().build_extensions()

    ext_modules = [
        CppExtension(
            name=EXT_NAME_ASCEND,
            sources=[
                *API_SOURCES,
                *ASCEND_KERNEL_SOURCES,
            ],
            extra_compile_args={
                "cxx": [
                    "-Ofast",
                    "-std=c++20",
                    f"--npu-arch={arch}",
                    "-DFLASH_MLA_IS_BUILD_ON_ASCEND",
                    "-isystem", str(Path(asc_home) / "aarch64-linux" / "include"),  # `-isystem` suppresses warnings
                    "-isystem", str(Path(asc_home) / "aarch64-linux" / "asc" / "include"),
                    "-isystem", str(Path(torch_npu.__file__).parent / "include"),
                    "-mllvm", "-enable-hiipu-vf-loop-unroll",
                    "--cce-res-usage",
                    # Otherwise some `__simd_callee__` are incorrectly marked as "function 'XXX' is not needed and will not be emitted"
                    "-Wno-unneeded-internal-declaration",
                    "-Wno-microsoft-template",
                    "-Wno-deprecated-declarations",
                    # Disable argument preload at the beginning of kernels to avoid a hardware bug
                    "-mllvm", "-cce-aicore-dcpreload-args=false",
                ],
            },
            extra_link_args=[
                str(Path(asc_home) / "aarch64-linux" / "lib64" / "libascendc_runtime.a"),
            ],
            include_dirs=[
                THIS_DIR / "csrc",
                THIS_DIR / "csrc" / "3rdparty" / "kerutils" / "include",
            ],
            libraries=[
                "torch_npu",
            ],
            library_dirs=[
                str(Path(torch_npu.__file__).parent / "lib"),
                str(Path(asc_home) / "aarch64-linux" / "lib64"),
            ],
        )
    ]

    return ext_modules, AscendBuildExtension


def get_build_target_platform() -> str:
    """
    The target platform of the build, "CUDA" or "ASCEND".

    `FLASH_MLA_BUILD_TARGET_PLATFORM` takes precedence. Otherwise the platform is detected
    without any dependency probing: the Ascend driver exposes `/dev/davinci_manager`,
    so its absence means CUDA.
    """
    overrided_platform = os.environ.get("FLASH_MLA_BUILD_TARGET_PLATFORM")
    if overrided_platform is not None:
        available = ["CUDA", "ASCEND"]
        if overrided_platform not in available:
            raise ValueError(
                f"Invalid `FLASH_MLA_BUILD_TARGET_PLATFORM`: {overrided_platform}. "
                f"Available values are {available}"
            )
        return overrided_platform
    return "ASCEND" if os.path.exists("/dev/davinci_manager") else "CUDA"


def get_version() -> str:
    try:
        rev = "+" + subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=THIS_DIR, stderr=subprocess.DEVNULL
        ).decode("ascii").rstrip()
    except Exception:
        rev = ""
    return "1.0.0" + rev


def main():
    current_platform = get_build_target_platform()
    print(f"Build target: {current_platform}")

    if current_platform == "CUDA":
        ext_modules, build_ext = build_on_cuda_platform()
    else:
        ext_modules, build_ext = build_on_ascend_platform()

    setup(
        name="flash_mla",
        version=get_version(),
        packages=find_packages(include=["flash_mla"]),
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )


if __name__ == "__main__":
    main()
