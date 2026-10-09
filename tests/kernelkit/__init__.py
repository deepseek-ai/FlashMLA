from . import bench
from . import build
from . import compare
from . import generate
from . import platform
from . import precision
from . import utils

from .bench import bench, bench_by_cuda_events
from .build import check_kernel_reg_spill_in_artifact, SpillCheckBuildExtension
from .compare import get_cos_diff, check_is_bitwise_equal, check_is_allclose, check_is_bitwise_equal_comparator, check_is_allclose_comparator
from .generate import gen_non_contiguous_randn_tensor, gen_non_contiguous_tensor, non_contiguousify
from .platform import Platform, get_current_platform, assert_current_platform, requires_platform, is_on_cuda_platform, is_on_ascend_platform, is_on_cpu_only_platform
from .precision import LowPrecisionMode, is_low_precision_mode, optional_cast_to_bf16_and_cast_back
from .utils import colors, cdiv, is_using_profiling_tools, set_random_seed, Counter
