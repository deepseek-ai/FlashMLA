"""Compile the sparse kernel's BF16 call sites against the installed CANN API."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


class AscendConversionCompileTests(unittest.TestCase):
    def test_conversion_symbols_are_exported_by_cann(self):
        compiler = shutil.which('bisheng')
        if compiler is None:
            self.skipTest('requires the CANN Bisheng compiler on PATH')
        header = Path(__file__).resolve().parents[1] / 'csrc/ascend_kernels/prefill/sparse/kernel_body.h'
        calls = sorted(set(re.findall(r'\b(asc_float2bfloat16_rn(?:_v2)?(?:_impl)?)\(', header.read_text())))
        self.assertEqual(len(calls), 2)
        source = '#include <c_api/asc_simd.h>\n'
        source += '__simd_callee__ inline void probe(vector_bf16 &dst, vector_float src, vector_bool mask) {\n'
        source += ''.join(name + '(dst, src, mask);\n' for name in calls) + '}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probe.asc'
            path.write_text(source)
            result = subprocess.run([compiler, '--npu-arch=' + os.environ.get('ASCEND_NPU_ARCH', 'dav-3510'),
                                     '--cce-aicore-only', '-std=c++20', '-O2', '-c', str(path),
                                     '-o', str(Path(directory) / 'probe.o')],
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
