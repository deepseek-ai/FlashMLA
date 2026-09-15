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


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.backend = types.ModuleType("flash_mla.cuda")
        self.backend.calls = 0

        def sparse_decode_fwd(q, *args):
            self.backend.calls += 1
            return q, q, None, None

        self.backend.sparse_decode_fwd = sparse_decode_fwd
        self.interface = load_interface(self.backend)

    def test_invalid_sparse_options_do_not_reach_backend(self):
        sched_meta, _ = self.interface.get_mla_metadata()
        q = torch.empty(1, 1, 64, 576)
        k_cache = torch.empty(1, 64, 1, 656)
        indices = torch.empty(1, 1, 64, dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "causal must be False"):
            self.interface.flash_mla_with_kvcache(
                q,
                k_cache,
                None,
                None,
                512,
                sched_meta,
                causal=True,
                is_fp8_kvcache=False,
                indices=indices,
            )

        self.assertFalse(sched_meta.have_initialized)
        with self.assertRaisesRegex(ValueError, "is_fp8_kvcache must be True"):
            self.interface.flash_mla_with_kvcache(
                q,
                k_cache,
                None,
                None,
                512,
                sched_meta,
                is_fp8_kvcache=False,
                indices=indices,
            )

        self.assertFalse(sched_meta.have_initialized)
        self.assertEqual(self.backend.calls, 0)

    def test_legacy_num_splits_placeholder_is_rejected(self):
        sched_meta, _ = self.interface.get_mla_metadata()
        with self.assertRaisesRegex(ValueError, "num_splits must be None"):
            self.interface.flash_mla_with_kvcache(None, None, None, None, 512, sched_meta, num_splits=1)

    def test_unsupported_prefill_options_are_rejected(self):
        calls = (
            (self.interface.flash_attn_varlen_func, (None, None, None, None, None, None, None)),
            (self.interface.flash_attn_varlen_qkvpacked_func, (None, None, None, None)),
            (self.interface.flash_attn_varlen_kvpacked_func, (None, None, None, None, None, None, None)),
        )
        for function, args in calls:
            with self.subTest(function=function.__name__), self.assertRaisesRegex(ValueError, "dropout_p must be 0.0"):
                function(*args, dropout_p=0.1)


if __name__ == "__main__":
    unittest.main()
