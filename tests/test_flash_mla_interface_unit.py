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


class SchedulerMetadataTests(unittest.TestCase):
    def test_failed_first_call_does_not_commit_scheduler_config(self):
        backend = types.ModuleType("flash_mla.cuda")
        backend.calls = 0

        def sparse_decode_fwd(q, *args):
            backend.calls += 1
            if backend.calls == 1:
                raise RuntimeError("injected backend failure")
            return q, q, torch.tensor([1]), torch.tensor([1])

        backend.sparse_decode_fwd = sparse_decode_fwd
        interface = load_interface(backend)
        sched_meta, _ = interface.get_mla_metadata()

        def invoke(batch_size):
            q = torch.empty(batch_size, 1, 64, 576)
            k_cache = torch.empty(1, 64, 1, 656)
            indices = torch.empty(batch_size, 1, 64, dtype=torch.int32)
            return interface.flash_mla_with_kvcache(
                q,
                k_cache,
                None,
                None,
                512,
                sched_meta,
                is_fp8_kvcache=True,
                indices=indices,
            )

        with self.assertRaisesRegex(RuntimeError, "injected backend failure"):
            invoke(1)

        self.assertFalse(sched_meta.have_initialized)
        self.assertIsNone(sched_meta.config)
        self.assertIsNone(sched_meta.tile_scheduler_metadata)
        self.assertIsNone(sched_meta.num_splits)

        out, _ = invoke(2)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(backend.calls, 2)
        self.assertTrue(sched_meta.have_initialized)
        self.assertEqual(sched_meta.config.b, 2)


if __name__ == "__main__":
    unittest.main()
