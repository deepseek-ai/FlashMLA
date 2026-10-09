from typing import Tuple, List, Callable, Union, Optional, Dict, overload
import csv
import dataclasses
import itertools
import os
import pathlib
import shutil
import tempfile

import torch
import triton

from .platform import Platform, get_current_platform, requires_platform
from .utils import is_using_profiling_tools, suppress_stdout_stderr

class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

@triton.jit
def profiler_range_start_marker_kernel():
    pass

def _run_profiler_range_start_marker_kernel():
    profiler_range_start_marker_kernel[(1,)]()

@dataclasses.dataclass
class BenchResult:
    """
    A struct holding the result of `bench_kineto`
    """

    num_tests: int
    time_ranges: Dict[str, List[Tuple[float, float]]]

    def _get_matched_kernel_name(self, name_substr: str, allow_no_match: bool = False, allow_multiple_match: bool = False) -> List[str]:
        matched_names = [name for name in self.time_ranges.keys() if name_substr in name]
        if not allow_no_match and len(matched_names) == 0:
            all_kernel_names_str = '\n  - ' + '\n  - '.join(self.time_ranges.keys())
            raise ValueError(f"Error: No kernel name matched for substring {name_substr}.\nAvailable kernels are: {all_kernel_names_str}")
        if not allow_multiple_match and len(matched_names) > 1:
            raise ValueError(f"Error: Multiple kernel matched for substring {name_substr}: {', '.join(matched_names)}")
        return matched_names
    
    def get_kernel_names(self) -> List[str]:
        return list(self.time_ranges.keys())
    
    def get_kernel_times(self, kernel_names_substr: List[str], allow_indivisible_run_count: bool = False, allow_missing: bool = False, allow_multiple_match: bool = False, return_avg_individual_run: bool = False) -> List[float]:
        """
        Get the average each-run time usage of each kernel provided in `kernel_names`

        If return_avg_individual_run is False, return sum(time) / num_tests, else return sum(time) / len(time)
        If is_using_profiling_tools (which is conflict with bench_kineto), return a series of 1 seconds
        """
        if is_using_profiling_tools():
            return [1 for _ in range(len(kernel_names_substr))]
        
        result = []
        for substr in kernel_names_substr:
            matched_names = self._get_matched_kernel_name(substr, allow_no_match=allow_missing, allow_multiple_match=allow_multiple_match)
            if len(matched_names) == 0:
                assert allow_missing
                result.append(0)
            else:
                time_usage_sum = 0
                run_cnt_sum = 0
                for matched_name in matched_names:
                    run_cnt = len(self.time_ranges[matched_name])
                    if not allow_indivisible_run_count and run_cnt % self.num_tests != 0:
                        raise RuntimeError(f"Error: the number of runs for kernel {matched_name} ({run_cnt}) is indivisible by `num_tests` ({self.num_tests})")
                    time_usage_sum += sum([end-start for (start, end) in self.time_ranges[matched_name]])
                    run_cnt_sum += run_cnt
                denominator = run_cnt_sum if return_avg_individual_run else self.num_tests
                result.append(time_usage_sum / denominator)
        return result
    
    def get_kernel_time(self, kernel_name_substr: str) -> float:
        return self.get_kernel_times([kernel_name_substr])[0]

    def get_e2e_time(self, kernel_names: list[str]) -> float:
        """
        Get the end-to-end time usage for a sequence of kernels
        defined as "last kernel end time" - "first kernel start time"
        If is_using_profiling_tools (which is conflict with bench_kineto), return 1 second
        """
        if is_using_profiling_tools():
            return 1
        
        matched_kernel_names = list(set(itertools.chain.from_iterable(self._get_matched_kernel_name(t, allow_no_match=False, allow_multiple_match=True) for t in kernel_names)))
        for kernel_name in matched_kernel_names:
            num_kernels = len(self.time_ranges[kernel_name])
            if num_kernels % self.num_tests != 0:
                raise RuntimeError(f"Error: the number of runs for kernel {kernel_name} ({num_kernels}) is indivisible by `num_tests` ({self.num_tests})")
        
        time_spans = []
        for i in range(self.num_tests):
            cur_run_time_spans = []
            for kernel_name in matched_kernel_names:
                num_cur_kernels = len(self.time_ranges[kernel_name])
                num_cur_kernels_each_run = num_cur_kernels // self.num_tests
                cur_run_time_spans.extend(self.time_ranges[kernel_name][num_cur_kernels_each_run*i: num_cur_kernels_each_run*(i+1)])
            cur_run_start_time = min(s[0] for s in cur_run_time_spans)
            cur_run_end_time = max(s[1] for s in cur_run_time_spans)
            time_spans.append((cur_run_start_time, cur_run_end_time))

        result = sum([end-start for (start, end) in time_spans]) / self.num_tests
        return result

def _bench_kineto(fn: Callable, num_tests: int = 30,
                 flush_l2: bool = True) -> BenchResult:
    """
    Run `fn` for `num_tests` times under `bench_kineto` (CUPTI), and returns a BenchKinetoRawResult
    """
    is_using_nsys = is_using_profiling_tools()

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1) if not is_using_nsys else None
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True) if not is_using_nsys else empty_suppress()

    is_kineto_log_level_unset = 'KINETO_LOG_LEVEL' not in os.environ
    if is_kineto_log_level_unset:
        os.environ['KINETO_LOG_LEVEL'] = '6'  # Suppress those "profiler_start" / "profiler_end" messages
    
    with profiler:
        for i in range(2):
            if i == 1 and not is_using_nsys:
                _run_profiler_range_start_marker_kernel()    # This marks the start of the profiling range
            for _ in range(num_tests):
                if flush_l2:
                    torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()
                enable_nvtx_range = i == 1 and _ == num_tests-1
                if enable_nvtx_range:
                    torch.cuda.nvtx.range_push("profile_target")
                fn()
                if enable_nvtx_range:
                    torch.cuda.nvtx.range_pop()
            if not is_using_nsys:
                if i == 0:
                    torch.cuda.synchronize()
                profiler.step()

    if is_kineto_log_level_unset:
        os.environ.pop('KINETO_LOG_LEVEL')

    if is_using_nsys:
        return BenchResult(num_tests, {})

    from torch.autograd.profiler_util import EventList, FunctionEvent   # pylint: disable=import-outside-toplevel
    events: EventList = profiler.events() # type: ignore

    # Filter out all events that are not function events
    events: List[FunctionEvent] = [event for event in events if isinstance(event, FunctionEvent)]

    # Filter out all events before the range marker
    for idx, event in enumerate(events):
        if event.name == "profiler_range_start_marker_kernel":
            events = events[idx+1:]
            break
    else:
        raise RuntimeError("Could not find profiler range start marker kernel event")

    # Get time ranges of each kernel
    kernel_times = {}
    for event in events:
        kernel_name = event.name
        if kernel_name not in kernel_times:
            kernel_times[kernel_name] = []
        kernel_times[kernel_name].append((event.time_range.start/1e6, event.time_range.end/1e6))
    
    return BenchResult(num_tests, kernel_times)


def _bench_msprof(fn: Callable, num_tests: int = 30,
                 flush_l2: bool = True) -> BenchResult:
    """
    Run `fn` for `num_tests` times under the Ascend profiler, and returns a BenchResult

    NOTE The Ascend profiler has no in-process event list like kineto's `profiler.events()`.
    Instead it dumps raw profiling data and analyses it into
    `ASCEND_PROFILER_OUTPUT/kernel_details.csv`, which holds exactly one row per kernel
    launch (`Name` / `Start Time(us)` / `Duration(us)`). Note that the analysis shells out to
    `msprof`, so each call costs a few seconds regardless of `num_tests`.
    """
    import torch_npu.profiler   # pylint: disable=import-outside-toplevel
    is_using_msprof = is_using_profiling_tools()

    flush_l2_size = int(8e9 // 4)

    def run_once():
        if flush_l2:
            torch.empty(flush_l2_size, dtype=torch.int, device='npu').zero_()
        fn()

    # Warm up outside the profiled range: the Ascend profiler records everything between
    # `start()` and `stop()`, it has none of kineto's `schedule`-based warmup
    for _ in range(num_tests):
        run_once()
    torch.npu.synchronize()

    prof_dir = tempfile.mkdtemp(prefix='kernelkit_bench_msprof_')
    try:
        # Pairing `schedule(active=1)` with a single `step()` records exactly `num_tests` iterations.
        # `on_trace_ready` runs `msprof --analyze` when the profiler stops, which is what actually
        # writes `kernel_details.csv`. Everything is wrapped in `suppress_stdout_stderr` because a
        # zero-warmup schedule and the `msprof` progress bar both print noisy messages.
        with suppress_stdout_stderr(True):
            profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                ),
            ) if not is_using_msprof else empty_suppress()

            with profiler as prof:
                for _ in range(num_tests):
                    run_once()
                torch.npu.synchronize()
                if not is_using_msprof:
                    prof.step()

        if is_using_msprof:
            return BenchResult(num_tests, {})

        kernel_details_csv = next(pathlib.Path(prof.prof_if.prof_path).rglob('kernel_details.csv'), None)
        if kernel_details_csv is None:
            raise RuntimeError(f"Could not find kernel_details.csv under {prof.prof_if.prof_path}")

        result = BenchResult(num_tests, {})
        with open(kernel_details_csv, newline='') as f:
            for row in csv.DictReader(f):
                start = float(row['Start Time(us)']) * 1e-6
                end = start + float(row['Duration(us)']) * 1e-6
                result.time_ranges.setdefault(row['Name'].strip(), []).append((start, end))
        return result
    finally:
        shutil.rmtree(prof_dir, ignore_errors=True)


def bench(fn: Callable, num_tests: int = 30,
          flush_l2: bool = True) -> BenchResult:
    current_platform = get_current_platform()
    if current_platform == Platform.CUDA:
        return _bench_kineto(fn, num_tests, flush_l2)
    elif current_platform == Platform.ASCEND:
        return _bench_msprof(fn, num_tests, flush_l2)
    else:
        raise RuntimeError(f"Unknown platform: {current_platform}")


@overload
def bench_by_cuda_events(kernels: List[Callable], num_warmups_each: int, num_runs_each: int) -> List[float]: ...

@overload
def bench_by_cuda_events(kernels: Callable, num_warmups_each: int, num_runs_each: int) -> float: ...

@requires_platform(Platform.CUDA)
def bench_by_cuda_events(kernels: Union[List[Callable], Callable], num_warmups_each: int = 10, num_runs_each: int = 30) -> Union[List[float], float]:
    buf_for_l2_clear = torch.empty(int(256e6//4), dtype=torch.int32, device='cuda')

    is_kernel_single_callable = isinstance(kernels, Callable)
    if is_kernel_single_callable:
        kernels = [kernels]

    torch.cuda.synchronize()
    for i in range(num_warmups_each):
        for kernel in kernels:
            kernel()
            if i == 0:
                # Ensure the first run is successful
                try:
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"Kernel {kernel.__name__} failed on warmup run {i}: {e}")
                    return []

    start_events = [[torch.cuda.Event(enable_timing=True) for _ in range(num_runs_each)] for _ in kernels]
    end_events = [[torch.cuda.Event(enable_timing=True) for _ in range(num_runs_each)] for _ in kernels]
    for i in range(num_runs_each):
        for j, kernel in enumerate(kernels):
            buf_for_l2_clear.random_()
            if i == num_runs_each-1:
                torch.cuda.nvtx.range_push("profile_target")
            start_events[j][i].record()
            kernel()
            end_events[j][i].record()
            if i == num_runs_each-1:
                torch.cuda.nvtx.range_pop()
    
    torch.cuda.synchronize()
    time_usages = [
        sum([start_events[j][i].elapsed_time(end_events[j][i])*1e-3 for i in range(num_runs_each)]) / num_runs_each
        for j in range(len(kernels))
    ]
    if is_kernel_single_callable:
        time_usages = time_usages[0]
    return time_usages
