# A Deep-Dive Into the New Flash MLA Kernel

In the [previous version](https://github.com/deepseek-ai/FlashMLA/tree/b31bfe72a83ea205467b3271a5845440a03ed7cb) of the Flash MLA kernel, we have achieved impressive performance: 3000 GB/s in memory-intensive settings and 580 TFlops in compute-bound settings. Now, we're pushing these numbers even further, reaching up to 660 TFlops.

In this blog, we present a deep dive into the new kernel, explaining the optimizations and techniques behind this performance boost. We'll first explain why the MLA kernel is compute-bound despite being a decoding-stage attention kernel, then discuss our high-level kernel schedule design, and finally cover the technical details of the new kernel.

## A Theoretical Analysis of the MLA Algorithm

GPU kernels can be classified as either compute-bound (limited by floating-point operations per second, FLOPs) or memory-bound (limited by memory bandwidth). To identify the kernel's bottleneck, we calculate the ratio of FLOPs to memory bandwidth (FLOPs/byte) and compare it with the GPU's capacity.

Assume the number of q heads is $h_q$, the number of q tokens per request is $s_q$ (should be 1 if MTP / speculative decoding is disabled), the number of kv tokens per request is $s_k\ (s_k \gg h_q s_q)$, and the head dimensions of K and V are $d_k$ and $d_v$ respectively. The number of FLOPs is roughly $2 (h_q s_q \cdot d_k \cdot s_k + h_q s_q \cdot s_k \cdot d_v) = 2 h_q s_q s_k (d_k+d_v)$, and the memory access volume (in bytes) is $\mathop{\text{sizeof}}(\text{bfloat16}) \times (h_q s_q d_k + s_k d_k + h_q s_q d_v) \approx 2s_k d_k$. Thus, the compute-memory ratio is $h_q s_q \cdot \frac{d_k+d_v}{d_k} \approx 2 h_q s_q$.

An NVIDIA H800 SXM5 GPU has a peak memory bandwidth of 3.35 TB/s and peak FLOPs of 990 TFlops. However, due to throttling (reducing to ~1600 MHz in our case), the practical peak FLOPs drops to ~865 TFlops. Therefore, when $h_qs_q \ge \frac{1}{2} \cdot \frac{865}{3.35} = 128$, the kernel is compute-bound; otherwise, it's memory-bound.

According to [the overview of DeepSeek's Online Inference System](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), we don't use Tensor Parallel for decoding instances, meaning $h_q$ is 128 and the kernel is compute-bound. Thus, we need to optimize the kernel for compute-bound settings.

## High-Level Design of the New Kernel

To fully utilize GPU compute resources, we need to overlap CUDA Core operations with Tensor Core operations and memory access with computation, keeping the Tensor Core constantly busy. This requires redesigning the kernel's "schedule."

[FlashAttention-3's paper](https://arxiv.org/abs/2407.08608) introduces ping-pong scheduling and intra-warpgroup GEMM-softmax pipelining to overlap block-wise matmul and CUDA Core operations. However, these techniques can't be directly applied here due to resource constraints. The output matrix (scaled and accumulated during each mainloop round, similar to [FlashAttention's algorithm](https://arxiv.org/abs/2205.14135)) must be stored in registers due to [WGMMA instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions) requirements. Each $64 \times 512$ output matrix occupies 32,768 32-bit registers. With only 65,536 32-bit registers per SM, we can store only one output matrix per SM. This eliminates the possibility of having two output matrices and letting them use CUDA Core and Tensor Core in a interleaved manner. We need to find another clever way to overlap CUDA Core and Tensor Core computation.

(You might pause here to ponder - perhaps you can find a better solution than ours!)

Our solution involves an additional mathematical transformation beyond FlashAttention's online softmax and accumulation approach. In each step, we take two KV blocks (called $K_0$, $K_1$, $V_0$, and $V_1$). Since the output matrix occupies 32,768 registers (too many for one warpgroup), we split it vertically into $O_L$ and $O_R$ (each $64 \times 256$). We similarly split $V_0$ and $V_1$ into $V_{0L}$, $V_{0R}$, $V_{1L}$, and $V_{1R}$ (each $64 \times 256$). The output matrix is then computed as follows:

0. Maintain a running max $m$ (initialized to $-\infty$, shared between the two warpgroups) and output matrices $\vec o_L, \vec o_R$ (initialized to 0).
1. [0] Compute $`\vec p_0 = \vec q K_0^\intercal / qk\_scale`$.
2. [1] Compute $`\vec p_1 = \vec q K_1^\intercal / qk\_scale`$.
3. [0] Compute $mp_0 = \max(\vec p_0)$, $`m\_new_0 = \max(m, mp_0)`$, and $`scale_0 = \exp(m\_new_0 - m)`$. Update $`m \gets m\_new_0`$.
4. [0] Perform softmax on $\vec p_0$: $`\vec p_0 \gets \exp(\vec p_0 - m\_new_0)`$.
5. [0] Update $\vec o_L \gets \vec o_L \cdot scale_0 + \vec p_0 V_{0L}$.
6. [1] Compute $mp_1 = \max(\vec p_1)$, $`m\_new_1 = \max(m, mp_1)`$, and $`scale_1 = \exp(m\_new_1 - m)`$. Update $`m \gets m\_new_1`$.
7. [1] Perform softmax on $\vec p_1$: $`\vec p_1 \gets \exp(\vec p_1 - m\_new_1)`$.
8. [1] Update $\vec o_R \gets \vec o_R \cdot (scale_0 \cdot scale_1) + \vec p_1 V_{1R}$.
9. [0] Update $\vec p_0 \gets \vec p_0 \cdot scale_1$.
10. [1] Update $\vec o_R \gets \vec o_R + \vec p_0 V_{0R}$.
11. [0] Update $\vec o_L \gets \vec o_L \cdot scale_1 + \vec p_1 V_{1L}$.

Note: We assume one q head for simplicity, so $\vec q$ and $\vec o$ are vectors. Bracketed numbers indicate the warpgroup performing the operation. Assume $\vec o_L$ resides in warpgroup 0's register and $\vec o_R$ resides in warpgroup 1's register.

This schedule can be viewed as a "ping-pong" variant using one output matrix—we call it "seesaw" scheduling. It's mathematically equivalent to FlashAttention's online softmax algorithm. This schedule allows us to overlap CUDA Core and Tensor Core operations by interleaving the two warpgroups, and also allows us to overlap memory access with computation since we can launch the corresponding Tensor Memory Accelerator (TMA) instructions right after data is no longer needed.

The complete schedule is shown below (remember that in MLA, $K$ and $V$ are the same with different names):

![MLA Kernel Sched](assets/MLA%20Kernel%20Sched.drawio.svg)

## Discussion of Technical Details

This section covers technical details of the new kernel.

First, although the kernel targets compute-bound scenarios (where memory bandwidth isn't the bottleneck), we can't ignore memory latency. If the data is not ready when we want to use it, we have to wait. To solve this problem, we employ the following techniques:

- **Fine-grained TMA copy - GEMM pipelining:** For a $64 \times 576$ K block, we launch 9 TMA copies (each moving a $64 \times 64$ block). GEMM operations begin as soon as each TMA copy completes (When the first TMA copy is done, we can start the first GEMM operation, and so on), improving memory latency tolerance.
- **Cache hints:** Using `cute::TMA::CacheHintSm90::EVICT_FIRST` for TMA copies improves L2 cache hit rates, as shown by experiments.

These optimizations achieve up to 80% Tensor Core utilization (of the throttled theoretical peak) and 3 TB/s memory bandwidth on an H800 SXM5 GPU. While slightly slower (~2%) than the old ping-pong buffer version in memory-bound settings, this is acceptable.

Other performance improvements include:
- **Programmatic Dependent Launch.** We use [programmatic dependent launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization) to overlap `splitkv_mla` and `combine` kernels.
- **Tile Scheduler.** We implement a tile scheduler to allocate jobs (requests and blocks) to SMs. This ensures a balanced load across SMs.

## Acknowledgements

FlashMLA's algorithm and scheduling are inspired by [FlashAttention](https://github.com/dao-AILab/flash-attention/), [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html), and [CUTLASS](https://github.com/nvidia/cutlass), as well as many projects behind them. We thank the authors for their great work.

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernels},
      author={Jiashi Li, Shengyu Liu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
