import time
import dataclasses
from typing import Tuple, List, Dict, Optional
import copy

import rich.console
import rich.table

import torch
import kernelkit as kk

import flash_mla

import lib
from lib import TestParam
from lib import RawTestParamForDecode as RawTestParam
import ref

"""
Generate testcase for unit test
"""

def gen_testcase() -> List[RawTestParam]:
    correctness_cases = []
    corner_cases = []
    for d_qk in [576, 512]:
        for have_extra_k in ([False, True] if d_qk == 512 else [False]):
            for have_extra_topk_len in ([False, True] if have_extra_k else [False]):
                for have_topk_len in ([False, True] if d_qk == 512 else [False]):
                    for h_q in [64, 128]:
                        cur_correctness_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=True,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0)
                            for (s_k, topk, block_size) in [
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [1, 3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                        ]
                        correctness_cases.extend(cur_correctness_cases)

                        cur_corner_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        is_all_indices_invalid=is_all_indices_invalid,
                                        have_zero_seqlen_k=have_zero_seqlen_k,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=enable_attn_sink,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0,
                            )
                            for (s_k, topk, block_size) in [
                                (512, 64, 61),
                                (650, 576, 53),
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 61),
                                (650, 576, 53),
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                            for is_all_indices_invalid in [True, False]
                            for have_zero_seqlen_k in [True, False]
                            for enable_attn_sink in [True, False]
                            if (is_all_indices_invalid or have_zero_seqlen_k or enable_attn_sink)
                        ]
                        corner_cases.extend(cur_corner_cases)

    base_and_bszs = [
        # V3.2
        (RawTestParam(0, 128, 2, 1, 32768, True, topk=2048, d_qk=576), [2, 64, 74, 128]),
        # MODEL1 CONFIG1
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=512, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG2
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG3
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG4
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
    ]
    performance_cases = [
        # Production cases
        dataclasses.replace(base, b=b)
        for base, bszs in base_and_bszs
        for b in bszs
    ] + [
        # Peak perf cases
        RawTestParam(74*2, h_q, 2, 1, 32768, True, topk=16384, d_qk=d_qk)
        for h_q in [64, 128]
        for d_qk in [512, 576]
    ]

    return correctness_cases + corner_cases + performance_cases


@dataclasses.dataclass
class Result:
    is_correct: bool
    compute_memory_ratio: float
    time_usage_per_us: float
    splitkv_time_usage_us: float
    combine_time_usage_us: float
    achieved_tflops: float
    achieved_gBps: float

_counter = kk.Counter()

@torch.inference_mode()
def test_flash_mla(p: TestParam) -> Result:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()
    assert p.decode

    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase_for_decode(p)

    tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    def run_decode():
        return lib.run_flash_mla_decode(p, t, tile_scheduler_metadata, None)
    
    # We first run the kernel once to generate output data for the correctness test
    # We must do this first, otherwise when allocating tensors for storing answers,
    # it may re-use memory that contains the correct answer, leading to false positives
    if p.check_correctness:
        torch.cuda.synchronize()
        out_ans, lse_ans = run_decode()
        torch.cuda.synchronize()
        # torch.set_printoptions(profile='full')
        # print(tile_scheduler_metadata.tile_scheduler_metadata[:, :7])
    
    # We run the performance test before generating the answer for the correctness test to avoid interference
    performance_result = Result(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if p.num_runs == 0:
        performance_result = Result(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        result = kk.bench_kineto(run_decode, p.num_runs)

        splitkv_kernel_name = "flash_fwd_splitkv_mla_fp8_sparse_kernel"
        combine_kernel_name = "flash_fwd_mla_combine_kernel"
        
        # Get individual kernel time usages
        kernel_time_usages_us: Dict[str, Optional[float]] = {}
        def pick_kernel_time_usage(kernel_name: str):
            t = [kernel_name in s for s in result.get_kernel_names()]
            if any(t):
                assert sum(t) == 1
                kernel_time_usages_us[kernel_name] = result.get_kernel_time(kernel_name) * 1e6
            else:
                kernel_time_usages_us[kernel_name] = None
        pick_kernel_time_usage(splitkv_kernel_name)
        pick_kernel_time_usage(combine_kernel_name)

        # Get E2E time usages
        def have_kernel(name: str):
            return kernel_time_usages_us[name] is not None
        
        if kk.is_using_profiling_tools():
            e2e_time_usage_us = 1e6
        else:
            assert have_kernel(splitkv_kernel_name)
            if have_kernel(combine_kernel_name):
                e2e_time_usage_us = result.get_e2e_time(splitkv_kernel_name, combine_kernel_name) * 1e6
            else:
                e2e_time_usage_us = kernel_time_usages_us[splitkv_kernel_name]
        assert e2e_time_usage_us is not None

        flops_and_mem_vol = lib.count_flop_and_mem_vol_for_decode(p, t)

        e2e_time_usage_s = e2e_time_usage_us / 1e6
        theoritical_compute_memory_ratio = flops_and_mem_vol.flop / flops_and_mem_vol.mem_vol
        achieved_tflops = flops_and_mem_vol.flop / e2e_time_usage_s / 1e12
        achieved_gBps = flops_and_mem_vol.mem_vol / e2e_time_usage_s / 1e9
        def print_kernel_time_usage(name: str, short_name: str):
            if kernel_time_usages_us[name] is not None:
                print(f'{short_name} time: {kernel_time_usages_us[name]:.1f} us')
        print(f'Compute/Memory: {theoritical_compute_memory_ratio:.2f}')
        print(f'Time (per): {e2e_time_usage_us:.1f} us')
        print_kernel_time_usage(splitkv_kernel_name, "Splitkv")
        print_kernel_time_usage(combine_kernel_name, "Combine")
        print(f'TFlops: {achieved_tflops:.1f}')
        print(f'GB/s: {achieved_gBps:.0f}')

        performance_result = Result(True, theoritical_compute_memory_ratio, e2e_time_usage_us, kernel_time_usages_us[splitkv_kernel_name] or 0.0, kernel_time_usages_us[combine_kernel_name] or 0.0, achieved_tflops, achieved_gBps)
    
    is_correct = True
    if p.check_correctness:
        torch.cuda.synchronize()
        with torch.profiler.record_function("reference_flash_mla"):
            out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)

        is_out_correct = kk.check_is_allclose("out", out_ans, out_ref, abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6)
        is_lse_correct = kk.check_is_allclose("lse", lse_ans, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)
        is_correct &= is_out_correct and is_lse_correct

    performance_result.is_correct = is_correct
    return performance_result


def main():
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(32)

    raw_testcases = gen_testcase()
    testcases = [t.to_test_param() for t in raw_testcases]

    print(f"{kk.colors['CYAN_BG']}{len(testcases)} testcases to run{kk.colors['CLEAR']}")

    is_no_cooldown = lib.is_no_cooldown()
    num_testcases_len = len(str(len(testcases)))
    failed_cases = []
    results: List[Tuple[TestParam, Result]] = []
    for testcase_idx, testcase in enumerate(testcases):
        if testcase != testcases[0] and testcase.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3) # Cooldown
        print(f"[{testcase_idx+1:{num_testcases_len}d}/{len(testcases)}, {testcase_idx/len(testcases)*100:3.0f}%]  ", end='')
        result = test_flash_mla(testcase)
        results.append((testcase, result))
        if not result.is_correct:
            failed_cases.append(testcase)
            import sys
            sys.exit(1)

    console = rich.console.Console(width=120)
    table = rich.table.Table(show_header=True, header_style="bold cyan")
    table.add_column("topk")
    table.add_column("Bsz")
    table.add_column("h_q&k")
    table.add_column("sq")
    table.add_column("sk")
    table.add_column("d_qk")
    table.add_column("Feats")
    table.add_column("C/M")
    table.add_column("TFlops")
    table.add_column("GBps")
    table.add_column("us")
    table.add_column(" ")

    for testcase, result in results:
        assert testcase.decode
        topk_str = f"{testcase.topk}" if testcase.decode.extra_topk is None else f"{testcase.topk}+{testcase.decode.extra_topk}"
        table.add_row(
            topk_str,
            str(testcase.decode.b),
            f"{testcase.h_q:3d} {testcase.h_kv}",
            str(testcase.s_q),
            str(testcase.s_kv),
            str(testcase.d_qk),
            " V"[testcase.decode.is_varlen] + " L"[testcase.have_topk_length] + " E"[testcase.decode.have_extra_topk_length],
            f"{result.compute_memory_ratio:3.0f}",
            f"{result.achieved_tflops:3.0f}",
            f"{result.achieved_gBps:4.0f}",
            f"{result.time_usage_per_us:4.1f}",
            "" if result.is_correct else "X"
        )
    console.print(table)

    def geomean(l) -> float:
        import numpy
        return numpy.exp(numpy.mean(numpy.log(l)))
    
    num_correct_testcases = [result.is_correct for t, result in results if t.check_correctness].count(True)
    num_correctness_cases = sum([1 for t in testcases if t.check_correctness])
    if num_correct_testcases == num_correctness_cases:
        print(f"{kk.colors['GREEN_BG']}{num_correct_testcases}/{num_correctness_cases} correctness cases passed{kk.colors['CLEAR']}")
    else:
        print(f"{kk.colors['RED_BG']}{num_correct_testcases}/{num_correctness_cases} correctness cases passed{kk.colors['CLEAR']}")
        for t in failed_cases:
            print(f"\t{t},")

    valid_achieved_tflops = [result.achieved_tflops for _, result in results if result.achieved_tflops > 0.1]
    if len(valid_achieved_tflops) > 0:
        achieved_tflops_geomean = geomean(valid_achieved_tflops)    # > 0.1 to prune out correctness cases
        print(f"TFlops     geomean: {achieved_tflops_geomean:.1f}")
    

if __name__ == "__main__":
    main()
