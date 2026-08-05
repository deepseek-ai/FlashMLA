import subprocess
from pathlib import Path

from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME


DEEPGEMM_COMMIT = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
CUTLASS_COMMIT = "147f5673d0c1c3dcf66f78d677fd647e4a020219"

_ATTENTION_SOURCES = [
    "csrc/persistent_sparse_attention/api.cpp",
    "csrc/persistent_sparse_attention/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu",
    "csrc/persistent_sparse_attention/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu",
    "csrc/persistent_sparse_attention/psa_fwd/sparse_attention.cu",
]

_PIPELINE_SOURCES = [
    "csrc/persistent_sparse_attention/pipeline_bindings.cpp",
    "csrc/persistent_sparse_attention/deepgemm/adapters_aot.cpp",
    "csrc/persistent_sparse_attention/deepgemm/frontend_aot.cu",
    "csrc/persistent_sparse_attention/deepgemm/logits_aot.cu",
    "csrc/persistent_sparse_attention/deepgemm/wqb_ready_aot.cu",
    "csrc/persistent_sparse_attention/root_fusion/root_fusion.cu",
    "csrc/persistent_sparse_attention/faithful_leaves/binding.cpp",
    "csrc/persistent_sparse_attention/faithful_leaves/kv_cache_gather.cu",
    "csrc/persistent_sparse_attention/faithful_leaves/index_weight_scale.cu",
    "csrc/persistent_sparse_attention/faithful_leaves/index_query_quant.cu",
    "csrc/persistent_sparse_attention/faithful_leaves/index_cache_pack.cu",
    "csrc/persistent_sparse_attention/attention_output_quant/attention_output_quant.cu",
    "csrc/persistent_sparse_attention/combine_indices/combine_indices.cu",
    "csrc/persistent_sparse_attention/index_cache_gather/index_cache_gather.cu",
    "csrc/persistent_sparse_attention/learned_topk/learned_topk.cu",
    "csrc/persistent_sparse_attention/main_compress/main_compress.cu",
    "csrc/persistent_sparse_attention/output_quant/output_quant.cu",
    "csrc/persistent_sparse_attention/reset_ready/reset_ready.cu",
    "csrc/persistent_sparse_attention/state_save/state_save.cu",
    "csrc/persistent_sparse_attention/swa_cache_insert/swa_cache_insert.cu",
]


def _git_head(path):
    return subprocess.check_output(
        ["git", "-c", "safe.directory={}".format(path), "rev-parse", "HEAD"],
        cwd=str(path),
        universal_newlines=True,
    ).strip()


def _cccl_include():
    cuda_root = Path(CUDA_HOME)
    candidates = (
        cuda_root / "include" / "cccl",
        cuda_root / "targets" / "x86_64-linux" / "include" / "cccl",
        cuda_root / "targets" / "sbsa-linux" / "include" / "cccl",
    )
    for candidate in candidates:
        if (candidate / "cuda" / "std").is_dir():
            return candidate
    return candidates[0]


def make_persistent_sparse_attention_extension(
    repo_root,
    cxx_args,
    feature_args,
    arch_flags,
    nvcc_thread_args,
    build_full_suite=False,
):
    """Build one SM100 persistent-attention extension.

    The base build contains sparse prefill/decode. The opt-in full suite adds
    the persistent pipeline kernels and fixed AOT frontend/logits/WqB
    instantiations to the same extension. DeepGEMM and CUTLASS are pinned
    build-only submodules; no dependency headers or runtime compiler are
    packaged in the wheel.
    """
    root = Path(repo_root)
    sources = list(_ATTENTION_SOURCES)
    defines = []
    libraries = []
    include_dirs = [
        _cccl_include(),
        root / "csrc",
        root / "csrc" / "kerutils" / "include",
        root / "csrc" / "sm90",
        root / "csrc" / "cutlass" / "include",
        root / "csrc" / "cutlass" / "tools" / "util" / "include",
    ]
    nvcc_args = [
        "-O3",
        "-std=c++20",
        "-DNDEBUG",
        "-D_USE_MATH_DEFINES",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use",
        "-lineinfo",
        "--source-in-ptx",
    ]

    if build_full_suite:
        deep_gemm = root / "csrc" / "deep_gemm"
        cutlass = root / "csrc" / "cutlass"
        if _git_head(deep_gemm) != DEEPGEMM_COMMIT:
            raise RuntimeError("DeepGEMM submodule must be at " + DEEPGEMM_COMMIT)
        if _git_head(cutlass) != CUTLASS_COMMIT:
            raise RuntimeError("CUTLASS submodule must be at " + CUTLASS_COMMIT)
        sources.extend(_PIPELINE_SOURCES)
        defines.extend([
            ("FLASH_MLA_PSA_WITH_PIPELINE", "1"),
            ("FLASH_MLA_CSA_WITH_ROOT_FUSION", "1"),
            ("FLASH_MLA_CSA_WITH_DEEPGEMM", "1"),
            ("FLASH_MLA_CSA_DEEPGEMM_ROOT_ADAPTED", "1"),
        ])
        libraries.append("cuda")
        include_dirs[1:1] = [
            root / "csrc" / "persistent_sparse_attention",
            root / "csrc" / "persistent_sparse_attention" / "deepgemm" / "include_overlays",
            deep_gemm / "deep_gemm" / "include",
        ]
        nvcc_args.extend(["-gencode", "arch=compute_100a,code=sm_100a"])
    else:
        nvcc_args.append("--use_fast_math")
        nvcc_args.extend(arch_flags)

    return CUDAExtension(
        name="flash_mla.persistent_sparse_attention_cuda",
        sources=sources,
        libraries=libraries,
        define_macros=defines,
        extra_compile_args={
            "cxx": cxx_args + feature_args,
            "nvcc": nvcc_args + feature_args + nvcc_thread_args,
        },
        include_dirs=include_dirs,
    )
