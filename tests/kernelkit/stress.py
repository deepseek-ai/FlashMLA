import dataclasses
import os
import random
import time
import datetime
import json
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import hfai
import argparse

try:
    os.environ["RAY_DEDUP_LOGS"] = "0"
    import ray
    HAVE_RAY = True
except ImportError:
    class ray:
        @staticmethod
        def remote(*args, **kwargs):
            def f(t):
                return t
            return f
    HAVE_RAY = False

from .utils import cdiv, colors


@dataclasses.dataclass
class StressResult:
    test_params: Any
    is_passed: bool
    msg: str

def save_test_data(path: str, test_data: Any):
    os.makedirs(path, exist_ok=True)
    for key in test_data.__dir__():
        # TODO This function is broken
        if not key.startswith('_') and isinstance(getattr(test_data, key), torch.Tensor):
            tensor: torch.Tensor = getattr(test_data, key)
            torch.save(tensor, os.path.join(path, f"test-data-{key}.pt"))

have_warned_about_cpu_result_tensor = False
have_warned_about_non_tensor_comparator = False
def run_batched_stress_test(
    test_params: Any,
    test_data: Any,
    run_func: Callable[[], Union[List, Tuple]],
    comparators: List[Callable[[Any, Any, torch.Tensor], None]],
    num_runs: int,
    num_runs_per_batch: int,
    pause: bool = True,
) -> StressResult:
    num_values = len(comparators)
    num_runs = cdiv(num_runs, num_runs_per_batch) * num_runs_per_batch

    first_outputs = run_func()
    assert len(first_outputs) == num_values

    for base_run_idx in range(0, num_runs, num_runs_per_batch):
        # Generate the execution order - We interleave runs with compares for maximum entropy
        order = list('R'*num_runs_per_batch + 'C'*num_runs_per_batch) # 'R' means "Run", 'C' means "Compare"
        if num_runs_per_batch > 1 and random.randint(0, 3) == 0:
            # Randomly shuffle the order but make sure that in every prefix, #R >= #C
            random.shuffle(order)
            new_order = []
            num_r_minus_num_c = 0
            for c in order:
                if c == 'R':
                    num_r_minus_num_c += 1
                    new_order.append(c)
                else:
                    if num_r_minus_num_c > 0:
                        num_r_minus_num_c -= 1
                        new_order.append(c)
            new_order.extend(['C'] * (2*num_runs_per_batch - len(new_order)))
            assert new_order.count('R') == num_runs_per_batch and new_order.count('C') == num_runs_per_batch
            order = new_order
        
        # Run
        this_batch_outputs = [] # [num_runs_per_batch, num_values]
        correctness_map = torch.empty((num_runs_per_batch, num_values), dtype=torch.bool, device='cuda')
        nxt_compare_outputs_idx = 0
        for i in range(num_runs_per_batch*2):
            if order[i] == 'R':
                # Execute
                this_batch_outputs.append(run_func())
            elif order[i] == 'C':
                # Compare
                victim_outputs = this_batch_outputs[nxt_compare_outputs_idx]
                assert len(victim_outputs) == num_values
                for value_idx in range(num_values):
                    comparators[value_idx](victim_outputs[value_idx], first_outputs[value_idx], correctness_map[nxt_compare_outputs_idx][value_idx])
                nxt_compare_outputs_idx += 1
        
        # 秋后算账
        cur_batch_correctness = correctness_map.cpu().tolist()
        for i in range(num_runs_per_batch):
            cur_run_correctness = cur_batch_correctness[i]
            if not all(cur_run_correctness):
                msg = f'error: WRONG ANSWER during `run_batched_test`. {test_params=}, {base_run_idx=}, {i=}, {order=}\n'
                for value_idx in range(num_values):
                    msg += f'  Value {value_idx}: {"PASS" if cur_run_correctness[value_idx] else "FAIL"}\n'
                print(f"{colors['RED_FG']}{msg}{colors['CLEAR']}")

                path = f"failed_tests/test_{str(datetime.datetime.now())}.{random.randint(0, 1000000)}"
                print(f"Saving test data and outputs to {path}...")
                os.makedirs(path)
                with open(os.path.join(path, "test_params.txt"), "w", encoding="utf-8") as f:
                    f.write(str(test_params) + "\n")
                save_test_data(path, test_data)
                for value_idx in range(num_values):
                    torch.save(first_outputs[value_idx], os.path.join(path, f"output-{value_idx}-first.pt"))
                    torch.save(this_batch_outputs[i][value_idx], os.path.join(path, f"output-run{i}-value{value_idx}-cur.pt"))
                return StressResult(test_params, False, msg)
        
        if pause:
            if random.randint(0, 1000) == 0:
                time.sleep(random.random()/2)
            elif random.randint(0, 300) == 0:
                time.sleep(random.random()/20)

    return StressResult(test_params, True, "PASSED")


@ray.remote(num_cpus=1, num_gpus=0)
class Glitcher:
    """
    A glitcher that launches some random workload on a GPU
    """
    def __init__(self, cuda_visible_devices: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.num_devices = len(cuda_visible_devices.split(","))
        self.cs = [torch.tensor(0., device=f'cuda:{i}') for i in range(self.num_devices)]
        self.algo_list = [
            self.glitch_matmul,
            self.glitch_layernorm,
            self.glitch_big_matmul,
            self.glitch_many_small_matmuls
        ]
    
    def get_cs(self) -> List[torch.Tensor]:
        return self.cs

    def glitch_matmul(self) -> torch.Tensor:
        a = torch.randn((512, 1024))
        b = torch.rand((1024, 512))
        c = a@b
        return c.sum()

    def glitch_layernorm(self) -> torch.Tensor:
        a = torch.randn((1024, 1024))
        b = torch.randn((1024, ))
        c = torch.nn.functional.layer_norm(a, (1024,), b, b)
        return c.sum()

    def glitch_big_matmul(self) -> torch.Tensor:
        a = torch.randn((4096, 8192), device='cuda')
        b = torch.rand((8192, 4096), device='cuda')
        c = (a @ b)*10
        return c.sum()
    
    def glitch_many_small_matmuls(self) -> torch.Tensor:
        c = torch.tensor(0., device='cuda')
        for _ in range(10):
            for i in range(1000):
                a = torch.randn((i, i), device='cuda')
                b = torch.randn((i, i), device='cuda')
                c += (a@b).sum()
        return c
    
    @torch.inference_mode()
    def glitch(self, gpu_idx: int, algo_idx: int):
        device = torch.device(f"cuda:{gpu_idx}")
        torch.set_default_device(device)
        torch.cuda.set_device(device)
        dtype = torch.bfloat16
        torch.set_default_dtype(dtype)
        torch.set_float32_matmul_precision('high')
        self.cs[gpu_idx] += self.algo_list[algo_idx]()

    def run_main_loop(self):
        GLITCH_INTERVAL = 0.2
        while True:
            gpu_idx = random.randint(0, self.num_devices-1)
            algo_idx = random.randint(0, len(self.algo_list)-1)
            # print(f"{lib.colors['GRAY_FG']}Glitching on GPU {gpu_idx} with algo {algo_idx}{lib.colors['CLEAR']}")
            self.glitch(gpu_idx, algo_idx)
            time.sleep(GLITCH_INTERVAL)

class RecordManager:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("{}")
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.loads(f.read().strip())
    
    def have_run_before(self, t: Any) -> bool:
        """
        Return whether test_params `t` has been run before
        """
        return str(t) in self.data

    def add_new_record(self, r: StressResult):
        """
        Add a new record to the record manager
        """
        self.data[str(r.test_params)] = {
            "is_passed": r.is_passed,
            "msg": r.msg,
            "fail_pass_mark": "FAIL" if not r.is_passed else "PASS" # 便于直接使用文本编辑器 / grep 搜索
        }
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.data, indent=4))

class Worker:
    def __init__(self, worker_id: int, run_on_testcase_func: Callable):
        self.worker_id = worker_id
        self.pid = os.getpid()
        self.run_on_testcase_func = run_on_testcase_func
        device = torch.device("cuda:0")
        torch.set_default_device(device)
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('high')
        torch.set_num_threads(16)
        print(f"Worker {worker_id} started on PID {self.pid}")

    def get_pid(self) -> int:
        return self.pid
    
    @torch.inference_mode()
    def run_on_testcase(self, test_params) -> StressResult:
        return self.run_on_testcase_func(self, test_params)

def do_stress_test(
    record_file_path: Optional[str],
    ignore_records: bool,
    num_workers: int,
    enable_glitching: bool,
    run_on_testcase_func: Callable,
    testcases: List[Any],
    use_ray: bool = True
):
    if use_ray:
        assert HAVE_RAY, "ray must be installed to run stress tests"
    else:
        assert num_workers == 1, "num_workers must be 1 when not using ray"
        assert not enable_glitching, "glitching must be disabled when not using ray"

    num_total_testcases = len(testcases)
    print(f"Starting stress test with {record_file_path=}, {ignore_records=}, {num_workers=}, {enable_glitching=}, {run_on_testcase_func=}, {use_ray=}")
    print(f"Number of testcases in total: {num_total_testcases}")

    record_manager = None
    if not ignore_records:
        assert record_file_path is not None, "record_file_path must be specified when not ignoring records"
        record_manager = RecordManager(record_file_path)
        testcases = [
            t
            for t in testcases
            if not record_manager.have_run_before(t)
        ]
    else:
        print('Filtering disabled.')

    num_remaining_testcases = len(testcases)
    num_filtered_testcases = num_total_testcases - num_remaining_testcases
    print(f"Number of testcases after filtering: {num_remaining_testcases} ({num_remaining_testcases/num_total_testcases*100.0:.1f}% remaining)")

    if use_ray:
        ray.init()
        RemoteWorker = ray.remote(num_gpus=1, num_cpus=1)(Worker)
        workers = [
            RemoteWorker.remote(worker_idx, run_on_testcase_func)
            for worker_idx in range(num_workers)
        ]
        worker_pids = [
            ray.get(worker.get_pid.remote())
            for worker in workers
        ]
        worker_cur_jobs = [
            None
            for _ in workers
        ]

        if enable_glitching:
            print("Enabling glitching")
            glitcher = Glitcher.remote(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
            glitcher.run_main_loop.remote()
        else:
            print("Glitching disabled")

        last_print_time = time.time()
        nxt_testcase_idx = 0
        num_finished_testcases = num_filtered_testcases

        while True:
            is_all_worker_idle = worker_cur_jobs.count(None) == len(worker_cur_jobs)
            if is_all_worker_idle and nxt_testcase_idx >= len(testcases):
                break

            # See if there is any idle worker
            for worker_idx, worker_cur_job in enumerate(worker_cur_jobs):
                try:
                    if worker_cur_job is not None:
                        result: StressResult = ray.get(worker_cur_job, timeout=0.01)
                        num_finished_testcases += 1
                        if not ignore_records:
                            record_manager.add_new_record(result)
                        color_code = colors['GREEN_FG'] if result.is_passed else colors['RED_FG']
                        print(f"{color_code}({num_finished_testcases}/{num_total_testcases}, {num_finished_testcases/num_total_testcases*100.0:.1f}%) Testcase {result.test_params} finished: {result.is_passed=}, {result.msg=}{colors['CLEAR']}")
                        worker_cur_jobs[worker_idx] = None

                    if nxt_testcase_idx < len(testcases):
                        # Assign a new job to the worker
                        testcase = testcases[nxt_testcase_idx]
                        print(f"{colors['CYAN_FG']}Assigning testcase #{num_filtered_testcases+nxt_testcase_idx}/{num_total_testcases} to worker {worker_idx} (pid={worker_pids[worker_idx]}) ({testcase}){colors['CLEAR']}")
                        nxt_testcase_idx += 1
                        worker_cur_jobs[worker_idx] = workers[worker_idx].run_on_testcase.remote(testcase)

                except ray.exceptions.GetTimeoutError:
                    continue
            
            if hfai.client.receive_suspend_command():
                print("Received suspend command. Exiting.")
                hfai.client.go_suspend()
            
            if time.time() - last_print_time > 25*60:
                # HFAI Platform kills a task if it doesn't have any output for 30 minutes
                print("Keepalive")
                last_print_time = time.time()
    else:
        worker = Worker(0, run_on_testcase_func)
        for testcase in testcases:
            print(f"Running on {testcase}", flush=True)
            result = worker.run_on_testcase(testcase)
            if not ignore_records:
                record_manager.add_new_record(result)
            print(f"{colors['CYAN_FG']}Running testcase {testcase}{colors['CLEAR']}")
            color_code = colors['GREEN_FG'] if result.is_passed else colors['RED_FG']
            print(f"{color_code}Testcase {result.test_params} finished: {result.is_passed=}, {result.msg=}{colors['CLEAR']}")

def stick_stress_test_args(parser: argparse.ArgumentParser):
    parser.add_argument('-p', '--record-file-path', type=str, default=None)
    parser.add_argument('-n', '--num-workers', required=True, type=int)
    parser.add_argument('-g', '--enable-glitching', action='store_true')
    parser.add_argument('--ignore-records', action='store_true')
    parser.add_argument('--no-ray', action='store_true')
