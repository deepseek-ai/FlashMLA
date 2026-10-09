import time
import sys

import argparse
import torch
import kernelkit as kk

from lib import TestParam
import lib
import ref

_counter = kk.Counter()

@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print("================")
    print(f"Running on {p}")
    if kk.is_on_cuda_platform():
        torch.cuda.empty_cache()

    t = lib.generate_testcase(p)
    lib.device_synchronize()

    ans_out, ans_max_logits, ans_lse = lib.run_flash_mla_sparse_fwd(p, t)
    lib.device_synchronize()

    is_correct = True
    if p.check_correctness:
        lib.device_synchronize()
        ref_out, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        lib.device_synchronize()

        out_criteria = ({'abs_tol': 8e-4, 'rel_tol': 3.01/128} if p.k_amplifier_portion == 0.0 else {'abs_tol': 1.0, 'rel_tol': 1.0}) | {'cos_diff_tol': 1e-5, 'dtype_for_cos_diff_calc': torch.float}
        max_logits_criteria = {'abs_tol': 1e-6, 'rel_tol': 2.01/65536} if p.k_amplifier_portion == 0.0 else {'abs_tol': 1.0, 'rel_tol': 1.0}
        lse_criteria = {'abs_tol': 1e-6, 'rel_tol': 2.01/65536} if p.k_amplifier_portion == 0 else {'abs_tol': 1e-5, 'rel_tol': 4.01/65536}
        is_correct &= kk.check_is_allclose("out", ans_out.float(), ref_out_fp32, **out_criteria)
        is_correct &= kk.check_is_allclose("max_logits", ans_max_logits, ref_max_logits, **max_logits_criteria)
        is_correct &= kk.check_is_allclose("lse", ans_lse, ref_lse, **lse_criteria)

    if p.num_runs > 0:
        flops_and_mem_vol = lib.count_flop_and_mem_vol(p, t)
        ans_time = kk.bench(lambda: lib.run_flash_mla_sparse_fwd(p, t), num_tests=p.num_runs).get_kernel_time("sparse_attn_fwd")
        flops = flops_and_mem_vol.fwd_flop/ans_time/1e12
        mem_bw = flops_and_mem_vol.fwd_prefill_mem_vol/ans_time/1e12
        print(f"Prefill:  {ans_time*1e6:4.0f} us, {flops:6.1f} TFlops, {mem_bw:4.2f} TBps")

    return is_correct


if __name__ == '__main__':
    device = torch.device("cuda:0") if kk.get_current_platform() == kk.Platform.CUDA else torch.device("npu:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    if kk.is_on_cuda_platform():
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    lib.stick_unit_test_args(parser)
    args = parser.parse_args()

    correctness_cases = [
        # Regular shapes
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, d_qk=d_qk)
        for d_qk in [512]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            # Regular shapes
            (128, 128),
            (256, 256),
            (512, 512),

            # Irregular shapes
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            # Irregular shapes with OOB TopK
            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [
            1, 62, 213
        ]
    ]

    correctness_cases_with_features = [
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, have_attn_sink=have_attn_sink, have_topk_length=have_topk_length, d_qk=d_qk, k_amplifier_portion=0.02 if enable_k_amplifier else 0.0, k_amplifier_ratio=2**8 if enable_k_amplifier else 1.0)
        for d_qk in [512]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [62, 213]
        for have_attn_sink in [False, True]
        for have_topk_length in [False, True]
        for enable_k_amplifier in [False, True]
    ]

    corner_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=True, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk, k_amplifier_portion=0.02 if enable_k_amplifier else 0.0, k_amplifier_ratio=2**8 if enable_k_amplifier else 1.0)
        for d_qk in [512]
        for h_q in [
            128, 64
        ]
        for s_q, s_kv, topk in [
            (1, 128, 128),
            (1, 256, 256),
            (1234, 4321, 4096),
            (4096, 2048, 2048)
        ]
        for enable_k_amplifier in [False, True]
    ] + [
        # In these cases, some blocks may not have any valid topk indices
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            (32, 2048),
            (64, 8192)
        ]
        for s_q in [1, 1024]
    ] + [
        # In this testcase, s_q is really large, so we cannot put it on the second dimension of grid shape
        TestParam(70000, 256, 256, h_q=h_q, check_correctness=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512]
        for h_q in [
            128, 64
        ]
    ]

    performance_case_templates = [
        (512, 64, 512+128, [4096, 8192, 32768]),
        (512, 128, 512+128, [4096, 8192, 32768]),
    ]

    performance_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, d_qk=d_qk, have_attn_sink=True)
        for (d_qk, h_q, topk, s_kv_list) in performance_case_templates
        for s_q in [4096]
        for s_kv in s_kv_list
    ]

    testcases = correctness_cases + correctness_cases_with_features + corner_cases + performance_cases

    filtered_testcases = []
    for t in testcases:
        if t.can_run_on_and_clamp(lib.TestTarget.SPARSE_FWD):
            filtered_testcases.append(t)
    testcases = filtered_testcases

    failed_cases = []
    for test in testcases:
        if test != testcases[0] and test.num_runs > 0 and not args.no_cooldown:
            time.sleep(0.3)
        is_correct = run_test(test)
        if not is_correct:
            failed_cases.append(test)
            if not args.run_to_finish:
                sys.exit(1)
    
    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")
