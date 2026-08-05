"""CPU-only packaging contracts for the persistent sparse-attention suite."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PSA_SOURCE = ROOT / "csrc" / "persistent_sparse_attention"
PSA_PACKAGE = ROOT / "flash_mla" / "persistent_sparse_attention"


def test_deepgemm_and_cutlass_are_pinned_build_only_submodules():
    modules = (ROOT / ".gitmodules").read_text()
    assert "path = csrc/cutlass" in modules
    assert "url = https://github.com/NVIDIA/cutlass.git" in modules
    assert "path = csrc/deep_gemm" in modules
    assert "url = https://github.com/deepseek-ai/DeepGEMM.git" in modules

    build = (ROOT / "persistent_sparse_attention_build.py").read_text()
    assert "559d79fb6994a58b8a15b4b93bf13ccc16edf247" in build
    assert "147f5673d0c1c3dcf66f78d677fd647e4a020219" in build
    assert 'deep_gemm / "deep_gemm" / "include"' in build
    assert 'root / "csrc" / "cutlass" / "include"' in build
    assert 'os.environ.get("DEEPGEMM_ROOT")' not in build

    setup = (ROOT / "setup.py").read_text()
    assert 'submodules = ["csrc/cutlass"]' in setup
    assert 'submodules.append("csrc/deep_gemm")' in setup
    assert 'missing_submodules = [' in setup
    assert 'if missing_submodules:' in setup
    assert 'path / ".git").exists()' in setup
    assert '"submodule", "update", "--init"' in setup
    assert "safe.directory={this_dir}" in setup
    assert "cwd=this_dir" in setup
    assert "check=True" in setup


def test_one_source_tree_one_python_package_one_extension():
    assert not (ROOT / "csrc" / "csa").exists()
    assert not (ROOT / "flash_mla" / "csa").exists()
    assert (PSA_SOURCE / "pipeline_bindings.cpp").is_file()
    assert (PSA_SOURCE / "psa_fwd" / "sparse_attention.cu").is_file()
    assert (PSA_PACKAGE / "ops.py").is_file()

    build = (ROOT / "persistent_sparse_attention_build.py").read_text()
    setup = (ROOT / "setup.py").read_text()
    assert 'name="flash_mla.persistent_sparse_attention_cuda"' in build
    assert "flash_mla.csa_cuda" not in build + setup
    assert "make_csa_extension" not in setup
    assert "FLASH_MLA_BUILD_PERSISTENT_SPARSE_ATTENTION_SUITE" in setup


def test_full_suite_is_aot_and_cuda13_compatible():
    build = (ROOT / "persistent_sparse_attention_build.py").read_text()
    adapters = (PSA_PACKAGE / "aot_adapters.py").read_text()
    assert '"arch=compute_100a,code=sm_100a"' in build
    assert 'cuda_root / "include" / "cccl"' in build
    assert 'libraries.append("cuda")' in build
    assert "nvrtc" not in build.lower()
    assert "_initialize_jit" not in adapters
    assert "nvcc" not in adapters.lower()
    assert not (PSA_PACKAGE / "include").exists()
    assert not (PSA_PACKAGE / "jit_adapters.py").exists()

    obsolete = {
        "adapters.cpp",
        "logits_out.cpp",
        "persistent_frontend.hpp",
        "wqb_ready.cpp",
        "wqb_ready_runtime.h",
    }
    assert not any((PSA_SOURCE / "deepgemm" / name).exists() for name in obsolete)


def test_project_overlay_is_explicit_and_minimal():
    overlay = PSA_SOURCE / "deepgemm" / "include_overlays"
    actual = {
        str(path.relative_to(overlay))
        for path in overlay.rglob("*")
        if path.is_file()
    }
    assert actual == {
        "deep_gemm/comm/barrier.cuh",
        "deep_gemm/epilogue/sm100_csa_prefill_c0_c4.cuh",
        "deep_gemm/epilogue/sm100_persistent_frontend_elementwise.cuh",
        "deep_gemm/impls/sm100_bf16_gemm.cuh",
        "deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh",
        "deep_gemm/impls/sm100_persistent_frontend_fused.cuh",
        "deep_gemm/ptx/ld_st.cuh",
        "deep_gemm/scheduler/gemm.cuh",
        "flash_mla/persistent_sparse_attention/wqb_ready.cuh",
    }


def test_wheel_packages_configs_but_no_dependency_headers():
    setup = (ROOT / "setup.py").read_text()
    assert "'flash_mla.persistent_sparse_attention': ['configs/*.json']" in setup
    assert "include/**/*" not in setup


if __name__ == "__main__":
    checks = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for check in checks:
        check()
    print("packaging contracts passed:", len(checks))
