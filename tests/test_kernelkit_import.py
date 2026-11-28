"""Kernelkit import must not require CUDA's optional profiler dependency."""
import subprocess
import sys
import unittest
from pathlib import Path

PREFIX = r'''
import importlib.abc
import json
import subprocess
import sys
import types
import torch
sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'tests'))
original = subprocess.check_output
def inventory(args, *a, **kw):
    return '' if args == ['lspci'] else original(args, *a, **kw)
subprocess.check_output = inventory
class RejectTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'triton' or fullname.startswith('triton.'):
            raise ModuleNotFoundError('optional Triton unavailable', name=fullname)
'''


class KernelkitImportTests(unittest.TestCase):
    def run_code(self, code):
        result = subprocess.run([sys.executable, '-c', PREFIX + code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_import_without_triton(self):
        self.run_code('''
sys.meta_path.insert(0, RejectTriton())
import kernelkit
assert 'triton' not in sys.modules
assert callable(kernelkit.bench)
''')

    def test_cuda_marker_still_requires_triton_when_called(self):
        self.run_code('''
sys.meta_path.insert(0, RejectTriton())
import importlib
bench = importlib.import_module('kernelkit.bench')
try:
    bench._run_profiler_range_start_marker_kernel()
except ModuleNotFoundError as error:
    assert error.name == 'triton'
else:
    raise AssertionError('CUDA marker silently skipped its Triton dependency')
''')

    def test_cuda_marker_is_jitted_and_launched(self):
        self.run_code('''
launches = []
jits = []
class Kernel:
    def __getitem__(self, grid):
        return lambda: launches.append(grid)
fake = types.ModuleType('triton')
def jit(function):
    jits.append(function.__name__)
    return Kernel()
fake.jit = jit
sys.modules['triton'] = fake
import importlib
bench = importlib.import_module('kernelkit.bench')
bench._run_profiler_range_start_marker_kernel()
bench._run_profiler_range_start_marker_kernel()
assert launches == [(1,), (1,)]
assert jits == ['profiler_range_start_marker_kernel']
''')


if __name__ == '__main__':
    unittest.main()
