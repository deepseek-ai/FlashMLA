# A Deep Dive Into the Ascend Sparse Attention Forward Kernel

## Introduction

On September 30, 2026, DeepSeek open-sourced its inference infrastructure and core components for the Huawei Ascend platform, including the Ascend sparse attention prefill and decoding kernels in this repository (deepseek-ai/FlashMLA). Under typical workloads of the DeepSeek V4.1 model, this kernel reaches 410 TFlops during prefill and 360 TFlops during decoding, i.e. 95% and 83% of the hardware's theoretical peak performance, respectively.

In this article, I will systematically describe the design rationale, pipeline schedule, and optimization techniques of this kernel.



## Algorithm Recap

This kernel computes the forward pass of the attention in the DeepSeek Sparse Attention (DSA) architecture used by the DeepSeek V4.1 model, covering both prefill and decoding. It takes a number of q tokens and kv tokens and, following the indices table, lets each q token attend to only the "most important" kv tokens, thereby saving a large amount of compute while preserving model quality.

### Mathematical Formulation

In this kernel, different q tokens are always processed separately. **For simplicity, we consider only a single Q token in the remainder of this section.**

Given:

- Q: `[h_q, d_qk]`

- KV: `[s_kv, d_qk]`

- indices: `[topk]`

- sm_scale: a scalar

the forward pass must compute:

- `gathered_kv = KV.index_select(indices, dim=0)`, `[topk, d_qk]`

- `P = Q @ gathered_kv^T`, `[h_q, topk]`

- `S = softmax(P*sm_scale, dim=-1)`, `[h_q, topk]`

- `out = S @ gathered_kv[:, :d_v]`, `[h_q, d_v]`

### Computation Flow

Because softmax has a global dependency, we compute it with the same online softmax algorithm as Flash Attention, dynamically updating the current per-row maximum of P, the per-row lse (log sum exp), and the output accumulator computed so far. The steps are as follows:

1. Split the topk indices into blocks of size `B_TOPK` (`B_TOPK` is usually 64 or 96)

2. Maintain three quantities: `running_max`, `running_max_for_softmax`, and `running_sumexp`, all of shape `[h_q]` and initialized to $-\infty$, $-\infty$, and 0. Also maintain an output accumulator `out_accum` of shape `[h_q, d_v]`. All of these are float32.

3. For each block:

    1. Read KV sparsely from global memory according to `indices[B_TOPK * k, B_TOPK * k + B_TOPK]`, obtaining this block's `gathered_kv`, `[B_TOPK, d_qk]`

    2. Compute `P = Q @ gathered_kv^T`, `[h_q, B_TOPK]`

    3. Compute the per-row maximum of `P`, `cur_max`, `[h_q]`

    4. Update `running_max`: `running_max = max(running_max, cur_max)` (in other words, `running_max` always reflects the current per-row maximum)

    5. Check whether some row `i` satisfies `running_max[i] - running_max_for_softmax[i] > RESCALE_THRES`, where `RESCALE_THRES` is a hyperparameter, usually set to $6$ in our kernel

    6. If some row `i` triggers this condition, enter the online softmax update

        1. `out_accum <- out_accum * exp(running_max_for_softmax - running_max).unsqueeze(-1)`

        2. `running_sumexp <- running_sumexp * exp(running_max_for_softmax - running_max)`

        3. `running_max_for_softmax <- running_max`

        4. This is equivalent to updating the max used by the online softmax, so that `exp(running_max - running_max_for_softmax)` is always $\le \exp(6)$

        5. If no row `i` triggers this condition, skip this step. This optimization is called "skip scale"

    7. Compute `S = exp(P - running_max_for_softmax)`, cast S to bfloat16, then multiply it with `gathered_kv` and add the result to out_accum (`out_accum += S.to(torch.bfloat16) @ gathered_kv`)

    8. Update `running_sumexp += S.sum(dim=-1)`



## Discussion

Before we formally begin the pipeline design and optimization techniques of the kernel, let us go through a few simple discussions to settle the overall design direction.

### How Many q Heads Should Each AI Core Handle?

The first question is: how many q heads per AI Core is appropriate?

Let us first compute the compute-to-memory ratio, assuming the CUBE Core runs at full throughput (4096 FMA per cycle per AI Core) and the total L2 bandwidth is 5 TB/s:

```Python
d = 512
l2_total_bw = 5e12
num_cores = 32
freq = 1650e6
l2_bw = l2_total_bw / num_cores / freq  # Per AI Core per cycle

for h_q in [16, 32, 64, 128]:
    for topk in [512, 1024]:
        cube_cycles = h_q * topk * (d+d) // 4096
        qo_mem_vol = h_q * d * 2 * 2
        kv_mem_vol = topk * d * 2
        mem_cycles = (qo_mem_vol + kv_mem_vol) / l2_bw
        print(f"{h_q:3d}, {topk=:4d}, {cube_cycles=:6d}, {mem_cycles=:6.0f}")
```

```YAML
16, topk= 512, cube_cycles=  2048, mem_cycles=  5883
 16, topk=1024, cube_cycles=  4096, mem_cycles= 11419
 32, topk= 512, cube_cycles=  4096, mem_cycles=  6229
 32, topk=1024, cube_cycles=  8192, mem_cycles= 11765
 64, topk= 512, cube_cycles=  8192, mem_cycles=  6921
 64, topk=1024, cube_cycles= 16384, mem_cycles= 12457
128, topk= 512, cube_cycles= 16384, mem_cycles=  8305
128, topk=1024, cube_cycles= 32768, mem_cycles= 13841
```

As we can see, the kernel is compute-bound only when head >= 64. At head=64, compute and memory are roughly balanced.

Meanwhile, taking 128 heads per AI Core would cause many problems in terms of SRAM capacity. Specifically, the L0C would be completely filled by O ($128 \times 512$, float32, 256KB), so the accumulator would have to be kept on the VECTOR Core; that in turn requires two AIVs and a fine-grained pipeline (pipelined copy - accumulate), which is very hard to write. In addition, according to micro-benchmark results, an MMAD with M = 64 is already enough to saturate CUBE Core throughput, so 128 heads is unnecessary.

### Where Should the FA Output Accumulator Live?

Whenever the running max differs too much from the scaling max currently in use, Flash Attention has to multiply the output by $\exp(\text{new\_max} - \text{old\_scaling\_max})$; this is called a "rescale". However, data in L0C can only be moved out by the FixPipe or used by the CUBE Core as an accumulator, and cannot be multiplied by a scalar in place. There are two feasible approaches:

1. Maintain an accumulator on the VECTOR Core, and whenever O needs to be rescaled, bring over the accumulation result from the CUBE Core and add it to the current accumulator

2. Keep the accumulator on the CUBE Core at all times, and when a rescale is needed, pull the accumulator up to the VECTOR Core, rescale it, and put it back onto the CUBE Core (by multiplying by an identity matrix)

    1. This consumes extra CUBE throughput and hurts performance. Moving a 64x512 matrix from UB to L0C requires at least one 64x64x512 float matmul

    2. Moreover, if no precision is to be lost, HF32 mode must be turned off; the CUBE Core then delivers only 256 FMA / cycle / AI Core, and the 64x64x512 matmul takes 8192 cycles, which is entirely unacceptable

We therefore chose option 1 in the end.

### Can We Use 1C1V?

The next question is how many VECTOR Cores each CUBE Core should be paired with during computation. Each AI Core of the Huawei Ascend 950 has one CUBE Core and two VECTOR Cores. If the number of VECTOR Cores required per CUBE Core could be kept at one (1C1V), the other VECTOR Core could run some other lightweight work (such as other modules in the model or cross-device communication), improving the overall MFU.

After some analysis, the answer is no, for the following reasons:

1. **MTE2 Issue Queue Entry bound.** On the Huawei Ascend 950 NPU, a single VECTOR Core can have at most 16 "issued but not yet completed" MTE2 copy requests at any moment. Even with the gather2 trick (see below), only 32 tokens can be in flight at a time, which is far too few.

2. **Dequantize bound.** During decoding, the kernel needs to read the KV Cache in FP8 or FP4 format, dequantize it to bfloat16, and then feed it to the CUBE Core. According to micro-benchmarks, with 1C1V, dequantization cannot keep up with the CUBE Core, making dequantization the bottleneck.

3. **FixPipe bound.** When only one CUBE Core and one VECTOR Core are used, the FixPipe moves data out at only 128 Byte/cycle/AI Core. In this case the MMA compute cycles per block are `MMA_M * B_TOPK * (512+512) / 4096 = B_TOPK * 16`, while the cycles needed for the FixPipe to move the data out are `(MMA_M * 512 + MMA_M * B_TOPK) * sizeof(float) / 128 = 1024 + B_TOPK * 2`. Here `MMA_M` is the number of q heads handled by each AI Core, which per the discussion above should be 64, and `B_TOPK` is the tile size when looping along the KV dimension in Flash Attention. For the FixPipe not to become the bandwidth bottleneck, we need `B_TOPK * 16 > 1024 + B_TOPK * 2`, which gives `B_TOPK > 73.14`. To leave enough margin, `B_TOPK` has to be 96 or 128, which on one hand makes UB space even tighter and on the other hand worsens the MTE2 Issue Queue Entry bound mentioned above.

Therefore, we decided in the end to give each CUBE Core two VECTOR Cores, so as to share the overhead of MTE2 issue, dequantization, FixPipe transfers, softmax, and so on.

### When Should the ND2NZ Layout Conversion of KV Happen?

The next question: when should the ND2NZ layout conversion of KV happen?

KV is stored in device memory in ND format, with shape `[s_kv, d_qk]` and stride `[d_qk, 1]`. The CUBE Core, however, requires its input matrices in NZ layout. Normally, we would perform the ND2NZ conversion on the fly while moving data from device memory into the L1 buffer, but our KV tokens are moved "sparsely" from device memory onto the chip by MTE2 one token at a time, so on-the-fly ND2NZ conversion cannot be enabled.

One idea is to treat a single token (512 bfloat16 values) as 32 blocks of 32B when issuing the GM -> L1 copy: these 32 blocks are contiguous in device memory, and are laid out in the L1 buffer with a stride of `B_TOPK * 32B`. Later experiments, however, showed that the BIU inside the AI Core (the unit that issues memory access instructions) triggers access coalescing only when **both the source and the destination are contiguous**. In a case like this, where only the source is contiguous and the destination is not, accesses cannot be coalesced, so L2 bandwidth utilization is very low and the overall bottleneck lies in L2 bandwidth.

Therefore, the only option is to first copy KV from device memory to UB, convert it from ND layout to NZ layout with a SIMD VF, and then copy it to L1.

### How Should the skip-scale Be Synchronized?

The "Algorithm Recap" section above mentioned the "skip-scale" technique, whose idea is that as long as the max currently used for online softmax is not much smaller than the true max, this round's update and accumulation of the output can be skipped.

This operation is even more valuable on Huawei NPUs, because scale-O is awkward there: it requires using the FixPipe to copy O from L0C to UB and then accumulating it into `out_accum` in UB via a VF, wasting the bandwidth of the FixPipe, UB, and the SIMD VF.

However, implementing this operation on the NPU has one difficulty: the signal "should we skip-scale this round" is produced by the VECTOR Core, but its consumers are both the CUBE Core and the VECTOR Core (because the CUBE Core needs it to decide whether to clear O in L0C). In the end, we decided to use an SS buffer and cross-core flags to pass the skip-scale signal.



## Pipeline Schedule

At this point, the kernel's overall design direction is finalized: 64 heads / AI Core, 1C2V, the output accumulator in UB, and KV routed through UB and a SIMD VF for the ND2NZ conversion.

The next step is the pipeline design. Since this kernel should be bound by CUBE Core throughput, our goal is to keep the matmul units on the CUBE Core saturated at all times, and not to let other operations (such as softmax) become the bottleneck.

The most basic pipeline idea is roughly as follows (the bracketed numbers below indicate which KV block an operation belongs to):

1. Compute `P[i] = Q @ K[i]^T` (`@` denotes matrix multiplication)

2. Compute `O += S[i-1] @ V[i-1]`

3. Compute `P[i+1] = Q @ K[i+1]^T`

4. Compute `O += S[i] @ V[i]`

5. ...

In other words, the `O += S @ V` of the previous block is deferred so as to leave time for the softmax operations: `P[i]` is ready as soon as (1) finishes, but it is not used until (4) starts, so there is ample time in between (about 1024 cycles) for the FixPipe to copy `P[i]` from L0C to UB, for softmax, and for copying `S[i]` from UB to L1.

Reading the indices (copying them from device memory to UB), preprocessing the indices (preparing the MTE copy parameters for the gather2 described below), issuing the MTE2 copies, and synchronizing the SS buffer signal in the skip-scale technique make this pipeline somewhat more complex. The CUBE Core side follows the schedule above, while the VECTOR Core side is scheduled as follows:

1. Issue the index read for KV block `i+5`

2. Issue the ND2NZ conversion and the UB -> L1 MTE3 copy for KV block `i+3`

3. Wait for `P[i+1] = Q @ k[i+1]^T` to complete, issue the softmax computation for KV block `i+1`, and copy the result to L1

4. Issue the index preprocessing for KV block `i+5`

5. Issue the O scale for KV block `i`, if this block does not skip-scale O

6. Synchronize with the SIMD VF, get whether KV block `i+1` needs to skip-scale O, and write it into the SS buffer

7. Issue the MTE2 copy for KV block `i+5`, copying KV from device memory to UB

Why is this pipeline so complex?

- First, the kernel relies on the SS buffer to pass the skip-scale signal, but only the scalar core can read and write the SS buffer, while the skip-scale signal is generated by the SIMD VF; this leads to a blocking synchronization between the SIMD VF and the scalar core. If this is not handled well, the scalar core will wait for the SIMD VF to finish and other units will sit idle.

- Second, Huawei's hardware pipes (PIPEs) issue and execute in order. For `PIPE_V`, for example, the order in which SIMD VFs are issued must exactly match the order in which they execute, so if an earlier SIMD VF cannot be issued because some condition is not satisfied (for instance, the `asc_sync_wait` condition is not yet met), the whole hardware pipe is blocked.

We therefore need to design the issue order of the various operations carefully, so that all hardware resources are fully utilized.



## Optimization Techniques

### gather2

This kernel needs to read KV from device memory at token granularity. If we issued one MTE2 copy request for every KV token, the total number of copy requests would be too large and would exceed the MTE2 issue queue depth (16). To avoid this, we use a trick: set the MTE2 request's `burst_count=2` and `src_stride` to the difference between the start addresses of the two tokens, so that two KV tokens can be copied at once, saving MTE2 copy requests and reducing issue overhead. The parameters of the MTE2 copy are computed with a SIMD VF, so the scalar core is not blocked.

### Decoding KV Cache Layout Optimization

In [the previous Flash MLA](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7), each block in the KV Cache (KV Block) stores the data of all tokens in the block first (i.e. the quantized FP8 or FP4 values), followed by the scales of all tokens. This design is not friendly to the Huawei Ascend: each token needs two MTE2 copies to fetch its data and scale. So in this kernel, we place a token's data and scale next to each other, so that data and scale no longer have to be copied separately.

### Fast ND2NZ

As mentioned above, we need a SIMD VF to perform the ND2NZ layout conversion of KV. The efficiency of this SIMD VF directly affects the efficiency of the whole kernel, especially the length of the "idle window" before the first QK matmul is launched. We therefore made the following optimizations to this SIMD VF:

- In the SIMD VF, use `asc_loadalign` to read the ND-format KV and `vsstb` to write each 32B data block back to UB at a certain interval, which completes the ND2NZ conversion.

- Pad one extra row in the output NZ buffer, to ensure that the write-back has no UB bank conflict.

- Use `asc_loadalign_postupdate` and the "auto-increment pointer" feature of `vsstb` to eliminate the pointer-update overhead, so that the SIMD VF's small scalar core does not become a new bottleneck.

- Process a pair of adjacent 256B elements at a time, so as to use the bandwidth of all UB banks to the full.

In the end, this VF achieves peak UB read and write throughput.

### The icache Miss Problem

Huawei Ascend NPUs have a relatively small instruction cache (icache): only 32K (CUBE Core), 16K (VECTOR Core), or 8K (SIMD VF). We therefore need to keep the total code size under control, to prevent code that is too large from causing frequent instruction cache thrashing. Useful tricks include using the hardware loop instruction (`VLOOP`) in the SIMD VF instead of unrolling loops with `#pragma unroll`, and being careful that function inlining can multiply code size (which is why the systolic array's main loop and drain loop were fused together at the end of the VECTOR Core).

### L1 Bank Balancing

The Ascend 950 NPU's L1 is split into two banks, which respectively control the 0 - 256K and 256K - 512K address ranges. To make full use of the bandwidth of these two banks, I split every tensor (such as Q and the quantized K) into two halves and place them in the two banks respectively, so that the bandwidth of both banks is saturated at all times.

### skip-scale

The skip-scale technique has many benefits, including:

- Reducing the volume of FixPipe copies and the amount of VF computation

- Reducing the pressure on UB read and write bandwidth

- Reducing the computation and data movement, and thus the power drawn by the kernel, which in turn allows a higher frequency and improves performance
