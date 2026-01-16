# KernelKit

## Overview

KernelKit is a collection of tools and utilities for testing & benchmarking & stress-testing kernels. Currently it includes the following submodules:

- `kernelkit.compare`: Various functions for comparing two tensors.
- `kernelkit.benchmark`: Functions for benchmarking kernel performance.
- `kernelkit.stress`: Functions for stress-testing kernels with various workloads.
- `kernelkit.utils`: Utility functions for kernel testing and benchmarking.

## Usage

首先先明确两个基本概念：

- 下文中的“测试用例配置”一词，指的是一组参数，用来描述一个测试用例的属性，比如 batch size, sequence length 之类的，其大小一般很小。你也可以叫它“训练超参数”。
- 下文中的“测试用例”一词，指的是一个具体的测试用例实例，包含若干个 `torch.Tensor`，其大小可能较大。

本项目**不是**一个 Python distribution package，仅仅是一个 source module。通俗地说，意思就是，这个项目没法通过 `pip install` 或者 `setup.py` 之类的方式安装，而是需要 git clone 成 submodule 之后，直接 import 使用。

下面是每种用况下的使用说明：

对于正确性测试以及性能评测，直接使用 `kernelkit.compare` 和 `kernelkit.benchmark` 中提供的函数即可，可以参考 [FlashMLA 中的脚本](https://gitlab.deepseek.com/deepseek/flash-mla/-/blob/main/tests/test_flash_mla.py?ref_type=heads)（关注那些 `kk.` 开头的函数）。

对于压力测试，情况略有复杂。你需要：

- 准备好测试用例配置、压测配置（比如使用几张 GPU）与函数 `run_on_testcase_func`（见下文），调用 `kernelkit.do_stress_test`。`do_stress_test` 会自动使用 [Ray](https://ray.io) 进行多卡并行的压力测试，且支持 hfai 优雅打断。
- 编写函数 `run_on_testcase_func`。该函数会被 `do_stress_test` 调用，接受测试用例配置 `test_param` 作为输入。在这个函数中，你应该依次完成以下任务：
    - 根据测试用例配置，生成测试用例。
    - 分别调用 reference 实现与待测的 kernel，比较答案的正确性。
    - 连续运行待测的 kernel，并检测输出相较于第一次调用来说是否完全一致。你可以使用 `kernelkit.run_batched_stress_test` 这个强大的函数来辅助完成这件事情。该函数会将 `num_runs` 次运行切分为大小为 `num_runs_per_batch` 的多个 batch。在每个 batch 中，函数会连续调用待测 kernel `num_runs_per_batch` 次，并在调用之间随机地插入比较答案的步骤（这样做是为了制造更多扰动），很方便。

可以参考 [FlashMLA 中的脚本](https://gitlab.deepseek.com/deepseek/flash-mla/-/blob/main/tests/stress_test_flash_mla.py?ref_type=heads)
