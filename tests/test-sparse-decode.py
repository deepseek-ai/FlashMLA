import time
import dataclasses
from typing import Tuple, List, Dict, Optional
import sys
import random
import copy

import argparse
import rich.console
import rich.table

import torch
import kernelkit as kk

import flash_mla

import lib
from lib import TestParam
from lib import RawTestParamForDecode as RawTestParam
import quant
import ref

"""
Generate testcase for unit test
"""

def gen_testcase() -> List[RawTestParam]:
    kk.set_random_seed(0)

    correctness_cases = []
    corner_cases = []
    for have_extra_k in [False, True]:
        for kvcache_layout, extra_kvcache_layout in [
            (quant.KVCacheLayout.V41_FP8Sparse, None),
            (quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4Sparse),   # fp4 extra_kv_cache
        ]:
            if extra_kvcache_layout is not None and not have_extra_k:
                continue    # The extra layout is meaningless without an extra cache
            for h_q in [64, 128]:
                have_topk_len = random.choice([False, True])
                have_extra_topk_len = random.choice([False, True, True]) if have_extra_k else False
                cur_correctness_cases = [
                    RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                have_topk_length=have_topk_len,
                                enable_attn_sink=random.randint(0, 1) == 1,
                                extra_s_k=extra_s_k,
                                extra_topk=extra_topk,
                                block_size=block_size,
                                extra_block_size=extra_block_size,
                                have_extra_topk_length=have_extra_topk_len,
                                d_qk=512,
                                kvcache_layout=kvcache_layout,
                                extra_kvcache_layout=extra_kvcache_layout,
                                check_correctness=True,
                                num_runs=0)
                    for (s_k, topk, block_size) in [
                        (512, 64, 5),
                        (512, 64, 69),
                        (1024, 576, 2),
                        (1024, 576, 61),
                        (2046, 2048, 1),
                        (2046, 2048, 64),
                        (2046, 2048, 576)
                    ]
                    for (extra_s_k, extra_topk, extra_block_size) in ([
                        (512, 64, 5),
                        (512, 64, 69),
                        (1024, 576, 2),
                        (1024, 576, 123),
                        (2046, 2048, 1),
                        (2046, 2048, 576)
                    ] if have_extra_k else [(None, None, None)])
                    for b in [4, 74, 321]
                    for s_q in [1, 3]
                    for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])    # With b=74 and no varlen/topk_len/extra_topk_len, no request should be split-kv
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
                                d_qk=512,
                                kvcache_layout=kvcache_layout,
                                extra_kvcache_layout=extra_kvcache_layout,
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
                    for b in [1, 74, 321]
                    for s_q in [3]
                    for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                    for is_all_indices_invalid in [True, False]
                    for have_zero_seqlen_k in [True, False]
                    for enable_attn_sink in [True, False]
                    if (is_all_indices_invalid or have_zero_seqlen_k or enable_attn_sink)
                ]
                corner_cases.extend(cur_corner_cases)

    # head128 with extra_topk not a multiple of B_TOPK (64): the partial extra block goes through the ceil + mask
    # scheduling of phase1 (h_q == 64 takes the head64 kernel, which requires the multiple)
    partial_extra_block_cases = [
        RawTestParam(4, 128, 3, 1, 512, True, 128,
                    extra_s_k=512, extra_topk=100, block_size=64, extra_block_size=64,
                    d_qk=512,
                    kvcache_layout=kvcache_layout,
                    extra_kvcache_layout=extra_kvcache_layout,
                    check_correctness=True, num_runs=0)
        for kvcache_layout, extra_kvcache_layout in [
            (quant.KVCacheLayout.V41_FP8Sparse, None),
            (quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4Sparse),
        ]
    ]

    base_and_bszs = [
        (RawTestParam(0, 64, 4, 1, 256, True, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512, block_size=256, extra_block_size=64, kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse), [64, 128, 256, 512]),
        (RawTestParam(0, 128, 4, 1, 256, True, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512, block_size=256, extra_block_size=64, kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse), [64, 128, 256, 512]),
        (RawTestParam(0, 64, 4, 1, 256, True, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512, block_size=256, extra_block_size=64, kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse, extra_kvcache_layout=quant.KVCacheLayout.V41_FP4Sparse), [64, 128, 256, 512]),
        (RawTestParam(0, 128, 4, 1, 256, True, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512, block_size=256, extra_block_size=64, kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse, extra_kvcache_layout=quant.KVCacheLayout.V41_FP4Sparse), [64, 128, 256, 512]),
    ]
    performance_cases = [
        # Production cases
        dataclasses.replace(base, b=b)
        for base, bszs in base_and_bszs
        for b in bszs
    ]
    performance_cases_prefill_as_decode = [
        RawTestParam(1, h_q, 4096, 1, 4096, False, 128, extra_s_k=4096, extra_topk=512, extra_block_size=64, d_qk=512, kvcache_layout=kvcache_layout, extra_kvcache_layout=extra_kvcache_layout)
        for h_q in [64, 128]
        for kvcache_layout, extra_kvcache_layout in [
            (quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP8Sparse),
            (quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4Sparse),
        ]
    ]

    return correctness_cases + corner_cases + partial_extra_block_cases + performance_cases + performance_cases_prefill_as_decode


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
    if kk.is_on_cuda_platform():
        torch.cuda.empty_cache()

    t = lib.generate_testcase_for_decode(p)

    tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    def run_decode(bsz_start: int = 0, bsz_end: Optional[int] = None):
        return lib.run_flash_mla_decode(p, t, tile_scheduler_metadata, None, bsz_start, bsz_end)
    
    # We first run the kernel once to generate output data for the correctness test
    # We must do this first, otherwise when allocating tensors for storing answers,
    # it may re-use memory that contains the correct answer, leading to false positives
    if p.check_correctness:
        lib.device_synchronize()
        out_ans, lse_ans = run_decode()
        lib.device_synchronize()
    
    # We run the performance test before generating the answer for the correctness test to avoid interference
    performance_result = Result(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if p.num_runs == 0:
        performance_result = Result(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        result = kk.bench(run_decode, p.num_runs)

        splitkv_kernel_name = {
            kk.Platform.CUDA: "flash_fwd_splitkv_mla_fp8_sparse_kernel",
            kk.Platform.ASCEND: "sparse_attn_fwd_kernel"
        }[kk.get_current_platform()]
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
                e2e_time_usage_us = result.get_e2e_time([splitkv_kernel_name, combine_kernel_name]) * 1e6
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
        lib.device_synchronize()
        out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)

        is_out_correct = kk.check_is_allclose("out", out_ans, out_ref, abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6)
        is_lse_correct = kk.check_is_allclose("lse", lse_ans, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)
        is_correct &= is_out_correct and is_lse_correct
        
    performance_result.is_correct = is_correct
    return performance_result


def main():
    device = torch.device("cuda:0") if kk.get_current_platform() == kk.Platform.CUDA else torch.device("npu:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    if kk.is_on_cuda_platform():
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(32)

    parser = argparse.ArgumentParser()
    lib.stick_unit_test_args(parser)
    args = parser.parse_args()

    raw_testcases = gen_testcase()

    # Prune out unsupported cases
    testcases = [t.to_test_param() for t in raw_testcases]
    filtered_testcases = []
    for t in testcases:
        t = copy.deepcopy(t)
        if t.can_run_on_and_clamp(lib.TestTarget.DECODE):
            filtered_testcases.append(t)
    testcases: List[TestParam] = filtered_testcases

    print(f"{kk.colors['CYAN_BG']}{len(testcases)} testcases to run{kk.colors['CLEAR']}")

    num_testcases_len = len(str(len(testcases)))
    failed_cases = []
    results: List[Tuple[TestParam, Result]] = []
    for testcase_idx, testcase in enumerate(testcases):
        if testcase_idx != 0 and testcase.num_runs > 0 and not args.no_cooldown:
            time.sleep(0.3) # Cooldown
        print(f"[{testcase_idx+1:{num_testcases_len}d}/{len(testcases)}, {testcase_idx/len(testcases)*100:3.0f}%]  ", end='')
        result = test_flash_mla(testcase)
        results.append((testcase, result))
        if not result.is_correct:
            failed_cases.append(testcase)
            if not args.run_to_finish:
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
    table.add_column("KV")
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
            "FP8+FP4" if testcase.decode.extra_kvcache_layout == quant.KVCacheLayout.V41_FP4Sparse else "FP8",
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
