import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("flash_mla_setup", ROOT / "setup.py")
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)

PATCH = (ROOT / "csrc" / "3rdparty" / "cutlass-windows.patch").read_text(encoding="utf-8").encode("utf-8")
BARRIER = b"""#pragma once

#include <cutlass/arch/memory_sm75.h>
#include <cute/arch/cluster_sm90.hpp>
"""
ADAPTER = (
    b"  /// Fills a buffer in Global Memory with a byte sequence copied from host memory\n"
    b"  template<class FillValueType>\n"
    b"  CUTLASS_HOST_DEVICE\n"
    b"  Status memsetDevice(\n"
    b"      void* destination,\n"
    b"      FillValueType fill_value, \n"
    b"      size_t count,\n"
)


class WindowsPatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        thirdparty = self.root / "csrc" / "3rdparty"
        self.cutlass = thirdparty / "cutlass"
        self.barrier = self.cutlass / "include" / "cutlass" / "arch" / "barrier.h"
        self.adapter = self.cutlass / "include" / "cutlass" / "cuda_host_adapter.hpp"
        self.barrier.parent.mkdir(parents=True)
        self.barrier.write_bytes(BARRIER)
        self.adapter.write_bytes(ADAPTER)
        self.patch = thirdparty / "cutlass-windows.patch"
        self.patch.write_bytes(PATCH)
        for patcher in (
            mock.patch.object(SETUP, "THIS_DIR", self.root),
            mock.patch.object(SETUP.sys, "platform", "win32"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_applies_both_fixes_and_is_idempotent(self):
        SETUP.apply_cutlass_windows_patch()
        self.assertIn(b"#if !defined(_MSC_VER)", self.barrier.read_bytes())
        self.assertNotIn(b"CUTLASS_HOST_DEVICE", self.adapter.read_bytes())
        patched = (self.barrier.read_bytes(), self.adapter.read_bytes())
        SETUP.apply_cutlass_windows_patch()
        self.assertEqual(patched, (self.barrier.read_bytes(), self.adapter.read_bytes()))

    def test_normalizes_windows_patch_line_endings(self):
        self.patch.write_bytes(PATCH.replace(b"\n", b"\r\n"))
        SETUP.apply_cutlass_windows_patch()
        self.assertIn(b"#if !defined(_MSC_VER)", self.barrier.read_bytes())
        self.assertNotIn(b"CUTLASS_HOST_DEVICE", self.adapter.read_bytes())

    def test_rejects_incompatible_sources_without_partial_changes(self):
        self.adapter.write_bytes(ADAPTER.replace(b"memsetDevice(", b"renamedDevice("))
        with self.assertRaisesRegex(RuntimeError, "could not be applied"):
            SETUP.apply_cutlass_windows_patch()
        self.assertEqual(self.barrier.read_bytes(), BARRIER)

    def test_rejects_partially_applied_patch(self):
        self.adapter.write_bytes(ADAPTER.replace(b"  CUTLASS_HOST_DEVICE\n", b""))
        with self.assertRaisesRegex(RuntimeError, "could not be applied"):
            SETUP.apply_cutlass_windows_patch()
        self.assertEqual(self.barrier.read_bytes(), BARRIER)

    def test_requires_initialized_submodule(self):
        self.adapter.unlink()
        with self.assertRaisesRegex(RuntimeError, "submodule is missing"):
            SETUP.apply_cutlass_windows_patch()

    def test_requires_patch_file(self):
        self.patch.unlink()
        with self.assertRaises(FileNotFoundError):
            SETUP.apply_cutlass_windows_patch()

    def test_other_platforms_do_not_modify_cutlass(self):
        with mock.patch.object(SETUP.sys, "platform", "linux"):
            with mock.patch.object(subprocess, "run") as run:
                SETUP.apply_cutlass_windows_patch()
        run.assert_not_called()
        self.assertEqual(self.barrier.read_bytes(), BARRIER)
        self.assertEqual(self.adapter.read_bytes(), ADAPTER)


if __name__ == "__main__":
    unittest.main()
