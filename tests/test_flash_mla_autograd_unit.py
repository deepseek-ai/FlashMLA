import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def load_interface(cuda_backend):
    """Load the Python wrapper with an isolated fake CUDA extension."""
    package = types.ModuleType("flash_mla")
    package.__path__ = []
    package.cuda = cuda_backend

    saved_package = sys.modules.get("flash_mla")
    saved_cuda = sys.modules.get("flash_mla.cuda")
    sys.modules["flash_mla"] = package
    sys.modules["flash_mla.cuda"] = cuda_backend
    try:
        path = Path(__file__).parents[1] / "flash_mla" / "flash_mla_interface.py"
        spec = importlib.util.spec_from_file_location("flash_mla_interface_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved_package is None:
            sys.modules.pop("flash_mla", None)
        else:
            sys.modules["flash_mla"] = saved_package
        if saved_cuda is None:
            sys.modules.pop("flash_mla.cuda", None)
        else:
            sys.modules["flash_mla.cuda"] = saved_cuda


class AutogradContractTests(unittest.TestCase):
    def test_lse_is_marked_non_differentiable(self):
        backend = types.ModuleType("flash_mla.cuda")
        backend.dense_prefill_fwd = lambda *args: None
        interface = load_interface(backend)

        q = torch.empty(1, 1, 128, requires_grad=True)
        k = torch.empty(1, 1, 128, requires_grad=True)
        v = torch.empty(1, 1, 128, requires_grad=True)
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

        out, lse = interface.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            1,
            1,
        )

        self.assertTrue(out.requires_grad)
        self.assertFalse(lse.requires_grad)
        self.assertIsNone(lse.grad_fn)


if __name__ == "__main__":
    unittest.main()
