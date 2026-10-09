import os
import subprocess
import re
import sys
from typing import List, Optional

from .platform import Platform, requires_platform, get_current_platform

if get_current_platform() == Platform.CUDA:
    from torch.utils.cpp_extension import BuildExtension
else:
    # Don't import `torch.utils.cpp_extension` since it prints "No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'", which is annoying
    class BuildExtension:
        pass

@requires_platform([Platform.CUDA, Platform.CPU_ONLY])
def check_kernel_reg_spill_in_artifact(artifact_path: str, stack_baseline: int = 0, quiet: bool = False, suppress_checking_env_var: Optional[str] = None) -> List[str]:
    """
    Check the compiled artifact (e.g. a .cubin or .so file) for register spilling.

    Returns a list of kernel names that spill (local memory > 0, or stack > stack_baseline).

    If `quiet` is True, no output is printed.
    If `suppress_checking_env_var` is not None, the check is skipped entirely when
    the environment variable named by `suppress_checking_env_var` is set to "1", "yes", or "true".
    """
    from torch.utils.cpp_extension import CUDA_HOME

    if suppress_checking_env_var is not None and os.environ.get(suppress_checking_env_var, '0').lower() in ['1', 'yes', 'true']:
        return []
    
    if not quiet:
        print(f"Checking register spills in: {artifact_path}")

    cuda_home = CUDA_HOME if CUDA_HOME is not None else '/usr/local/cuda'
    cuobjdump_path = os.path.join(cuda_home, 'bin/cuobjdump')
    if not os.path.exists(cuobjdump_path):
        raise FileNotFoundError(f"cuobjdump not found (looked at {cuobjdump_path})")
    
    try:
        result = subprocess.run(
            [cuobjdump_path, "-res-usage", artifact_path],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("cuobjdump timed out during spill checking")
        raise RuntimeError()

    if result.returncode != 0:
        print(f"cuobjdump failed:\n{result.stdout}\n{result.stderr}")
        raise RuntimeError()

    def _parse_cuobjdump(output: str) -> list[tuple[str, int, int, int]]:
        """Parse cuobjdump output, returning [(name, reg, stack, local)] for kernels that spill."""
        spills = []
        current_name = None
        func_re = re.compile(r"^\s*Function\s+(\S+)")
        resource_re = re.compile(
            r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:\d+\s+LOCAL:(\d+)"
        )

        for line in output.splitlines():
            m = func_re.match(line)
            if m:
                current_name = m.group(1)
                continue
            m = resource_re.search(line)
            if m and current_name:
                reg = int(m.group(1))
                stack = int(m.group(2))
                local = int(m.group(3))
                if stack > stack_baseline or local > 0:
                    spills.append((current_name[:-1], reg, stack, local))
                current_name = None

        return spills

    spills = _parse_cuobjdump(result.stdout)
    if not spills:
        if not quiet:
            print("No register spills detected.")
        return []

    if not quiet:
        print(f"Found {len(spills)} kernel(s) with register spilling:\n")
        print(f"{'REG':>6}  {'STACK':>6}  {'LOCAL':>6}  Kernel")
        print("-" * 60)
        for name, reg, stack, local in spills:
            print(f"{reg:>6}  {stack:>6}  {local:>6}  {name}")

        print("\nRegister spilling can significantly degrade kernel performance.")
        print("This is often caused by differences in the compiler version or toolchain.")
        print("If you see this message, you may:")
        print("  - Investigate the cause of the spill")
        if suppress_checking_env_var is not None:
            print(f"  - Or, suppress this check by setting {suppress_checking_env_var}=1")

    return [s[0] for s in spills]

class SpillCheckBuildExtension(BuildExtension):
    @requires_platform(Platform.CUDA)
    def __init__(self, stack_baseline: int = 0, suppress_checking_env_var: Optional[str] = None):
        self.stack_baseline = stack_baseline
        self.suppress_checking_env_var = suppress_checking_env_var
        super().__init__()
    
    def run(self):
        super().run()
        for ext in self.extensions:
            so_path = self.get_ext_fullpath(ext.name)

            spilled_kernels = check_kernel_reg_spill_in_artifact(so_path, self.stack_baseline, False, self.suppress_checking_env_var)
            if len(spilled_kernels) > 0:
                print('Register spilling detected. Build failed!')
                sys.exit(1)
        print('Register spill check passed.')
        