# Ascend Sparse Attention Forward 算法与优化技术简析

## 前言

DeepSeek 于 2026 年 9 月 30 日开源了华为昇腾平台的推理基础设施与核心组件，其中就包括本仓库（deepseek-ai/FlashMLA）中的昇腾平台的稀疏 Attention 的 prefill 与 decoding 算子。在 DeepSeek v4.1 模型的典型工况下，该算子可以在 prefill 时达到 410 TFlops，decoding 时达到 360 TFlops，分别达到了硬件理论性能上限的 95% 与 83%。

在本文中，我将系统地介绍该算子的设计思路、流水排布与优化技巧。



## 算法回顾

本算子用于计算 DeepSeek V4.1 模型中使用的 DeepSeek Sparse Attention (DSA) 架构中的 Attention 的前向传播，包括 Prefill 和 Decoding。其接受若干 q token 与 kv token，按照 indices 表的指示，让每个 q token 只关注部分“最重要的” kv token，从而在保证模型效果的同时大幅节约算力。

### 数学建模

在本算子中，不同的 q token 总是分开处理的。**为了简便，我们后面只考虑单个 Q token。**

给定：

- Q: `[h_q, d_qk]`

- KV: `[s_kv, d_qk]`

- indices: `[topk]`

- sm_scale: 一个标量

前向传播需要计算的是：

- `gathered_kv = KV.index_select(indices, dim=0)`, `[topk, d_qk]`

- `P = Q @ gathered_kv^T`, `[h_q, topk]`

- `S = softmax(P*sm_scale, dim=-1)`, `[h_q, topk]`

- `out = S @ gathered_kv[:, :d_v]`, `[h_q, d_v]`

### 计算流程

由于 softmax 具有全局依赖，我们使用 Flash Attention 同款的 online softmax 算法进行计算，动态更新当前的每行的 P 的最大值、每行的 lse (log sum exp)、以及迄今为止的输出累加器。具体步骤如下：

1. 将 topk 个 indices 切成若干块，每块大小为 `B_TOPK`（`B_TOPK` 一般取 64 或 96）

2. 维护三个东西：`running_max`, `running_max_for_softmax` 和 `running_sumexp`，shape 均为 `[h_q]`，初始值为 $-\infty$, $-\infty$ 和 0。另维护一个 output accumulator `out_accum`, shape 为 `[h_q, d_v]`。这几个东西都是 float32 的。

3. 对于每个块：

    1. 根据 `indices[B_TOPK * k, B_TOPK * k + B_TOPK]` 稀疏地从 global memory 中读取 KV，得到这个块的 `gathered_kv`, `[B_TOPK, d_qk]`

    2. 计算 `P = Q @ gathered_kv^T`, `[h_q, B_TOPK]`

    3. 计算 `P` 的每行的最大值 `cur_max`, `[h_q]`

    4. 更新 `running_max`：`running_max = max(running_max, cur_max)`（也就是说 `running_max` 总是反映当前每行的最大值）

    5. 检查是否有某一行 `i` 满足 `running_max[i] - running_max_for_softmax[i] > RESCALE_THRES`，其中 `RESCALE_THRES` 为一超参数，在我们的 kernel 中一般取 $6$

    6. 如果有某一行 `i` 触发了这个条件，进入 online softmax 更新流程

        1. `out_accum <- out_accum * exp(running_max_for_softmax - running_max).unsqueeze(-1)`

        2. `running_sumexp <- running_sumexp * exp(running_max_for_softmax - running_max)`

        3. `running_max_for_softmax <- running_max`

        4. 这相当于更新了 online softmax 所使用的 max，因此 `exp(running_max - running_max_for_softmax)` 总是 $\le \exp(6)$ 的

        5. 如果没有任何一行 `i` 触发了这个条件，就跳过这一步。该优化技术被称为“skip scale”

    7. 计算 `S = exp(P - running_max_for_softmax)`，并将 S cast 成 bfloat16，然后与 `gathered_kv` 相乘，加到 out_accum 上（`out_accum += S.to(torch.bfloat16) @ gathered_kv`）

    8. 更新 `running_sumexp += S.sum(dim=-1)`



## 讨论

在正式开始介绍算子的流水设计、优化技巧之前，我们先进行几个简单的讨论，确定算子的总体设计方向：

### 每个 AI Core 负责多少个 q head

第一个问题是：每个 AI Core 负责多少个 q head 合适呢？

先来算一下算存比：按照 CUBE Core 打满（4096 FMA per cycle per AICore）、L2 总带宽 5 TB/s 计算：

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

可以看出 head >= 64 才是 Compute bound。对于 head=64 的情况，compute 和 memory 大致均衡。

同时，如果 per AI Core 的 head 取 128 的话，SRAM 容量方面会有诸多问题。具体来说，L0C 的容量会被 O（128 x 512, float32, 256KB）完全占满，因此 accumulator 必须放在 VECTOR Core 上面，因此必须用两个 AIV，以及必须要做细粒度 Pipeline（Pipeline 拷贝 - 累加），很难写。以及，根据 micro-benchmark 的结果，MMAD 的 M 取 64 就足以打满 CUBE Core 的吞吐，因此 head 取 128 是不必要的。

### FA 的 output accumulator 在哪里做

Flash Attention 需要在 running max 和当前缩放使用的 scaling max 差异过大的时候，需要把 output 乘上 $\exp(\text{new\_max} - \text{old\_scaling\_max})$，称为“rescale”。但是，L0C 上的数据只能被 FixPipe 搬出或被 CUBE Core 用作累加器，无法原地乘以标量。可行的做法有两种：

1. 在 VECTOR Core 上维护一个 accumulator，每次需要 rescale-O 的时候，把 CUBE 上的 accumulation 结果拿过来，并加在当前的 accumulator 上面

2. accumulator 始终在 CUBE Core 上面，当需要 rescale 的时候，把 accumulator 拉到 vector 上面，rescale，并放回到 CUBE 上面（通过乘一个单位矩阵）

    1. 这种做法需要消耗额外的 CUBE 算力，有损性能。把一个 64x512 的矩阵从 UB 搬到 L0C 上，至少需要做一个 64x64x512 的 float 矩阵乘法

    2. 并且，如果希望不损失精度，需要关闭 HF32 mode，此时 CUBE 算力只有 256 FMA / cycle / AI Core，64x64x512 的矩阵乘法需要 8192 周期，根本无法接受

因此，最终选择了方案一。

### 能不能用 1C1V

下一个问题是，每个 CUBE Core 在计算时配合上多少个 VECTOR Core 比较合理呢？华为昇腾 950 的每个 AI Core 具有一个 CUBE Core 和两个 VECTOR Core，倘若能把每个 CUBE Core 所需的 VECTOR Core 数量控制在一个（1C1V），那么另一个 VECTOR Core 便可以跑一些其他的轻量操作（比如模型中的其它模块或跨卡通信），提高整体的 MFU。

经过一些分析，答案是不能。原因如下：

1. **MTE2 Issue Queue Entry bound**。在华为昇腾 950 NPU 上，单个 VECTOR Core 在同一时刻最多只能有 16 个“已发出但还未完成”的 MTE2 拷贝请求。哪怕用上了 gather2 技巧（见下文），一次也只能拷贝 32 个 token，太少了。

2. **Dequantize bound**。在 decoding 的时候，算子需要读取 FP8 或 FP4 格式的 KV Cache，将其反量化为 bfloat16 格式，再送入 CUBE Core 计算。根据微基准测试（micro-benchmark），使用 1C1V 时，反量化的速度赶不上 CUBE Core 计算的速度，导致反量化成为瓶颈。

3. **FixPipe bound**。当只使用一个 CUBE Core 和一个 VECTOR Core 的时候，FixPipe 向外搬运的速度只有 128 Byte/cycle/AI Core。此时每个块的 MMA 计算周期为 `MMA_M * B_TOPK * (512+512) / 4096 = B_TOPK * 16`，FixPipe 搬运所需的周期为 `(MMA_M * 512 + MMA_M * B_TOPK) * sizeof(float) / 128 = 1024 + B_TOPK * 2`。其中 `MMA_M` 代表每个 AI Core 处理的 q head 数量，根据上文的讨论，应该是 64；`B_TOPK` 为 Flash Attention 中，沿着 KV 方向循环时的 Tile size。如果希望 FixPipe 不要成为带宽瓶颈的话，需要 `B_TOPK * 16 > 1024 + B_TOPK * 2`，得到 `B_TOPK > 73.14`。为了留出足够的余量，`B_TOPK` 需要选取 96 或 128，这一方面使得 UB 空间更加紧张，另一方面也加重了上面提到的 MTE2 Issue Queue Entry Bound 的问题。

因此，最终我们还是决定为每个 CUBE Core 配备两个 VECTOR Core，以分摊 MTE2 发射、反量化、FixPipe 搬运、softmax 等操作的开销。

### KV 的 ND2NZ Layout 转换何时进行

下一个问题：KV 的 ND2NZ Layout 转换何时进行呢？

KV 在显存中是以 ND 格式存储的，shape 为 `[s_kv, d_qk]`，stride 为 `[d_qk, 1]`。但是，CUBE Core 要求输入的矩阵使用 NZ layout。正常来说，我们会在把数据从显存搬运到 L1 buffer 中时，随路完成 ND2NZ 转换，但我们的 KV 是使用 MTE2 一个一个 token 地“稀疏”的从显存上搬到片上存储上的，无法启用随路的 ND2NZ 转换。

有一个思路是，在发起 GM -> L1 拷贝时，将一个 token（512 个 bfloat16）视为 32 个 32B 的数据块，这 32 个数据块在显存上是连续的，在 L1 buffer 中则以 `B_TOPK * 32B` 为 stride 排布。但后续实验显示，AI Core 内部的 BIU（负责发送访存指令的单元）只会在**源和目标都连续**的时候才会触发访存合并，而像这里这种“只有源连续，而目的不连续”的情况是无法合并访存的，导致 L2 带宽利用率很低，整体瓶颈位于 L2 带宽。

所以，唯一的做法就是将 KV 先从显存拷贝到 UB，使用 SIMD VF 将其从 ND Layout 转为 NZ Layout，然后再拷贝到 L1。

### 如何进行 skip-scale 的同步

上文“算法回顾”一章中提到过“skip-scale”这一技术，其思想为，只要现在用来做 online softmax 的 max 没有比真正的 max 小太多，就可以跳过本轮的 output 的更新与累加。

这个操作在华为 NPU 上的意义更大 —— 因为 NPU 上 scale-O 很麻烦，需要使用 FixPipe 将 O 从 L0C 拷贝到 UB，然后通过 VF 累加到 UB 的 `out_accum` 上，浪费 FixPipe、UB 与 SIMD VF 的带宽。

但是，想在 NPU 上实现这个操作有一个难点：“这轮要不要 skip-scale”的信号是 VECTOR Core 发起的，消费者却是 CUBE Core 与 VECTOR Core 二者（因为 CUBE Core 需要决定是否将 L0C 上的 O 清空）。最终，我们决定使用 SS buffer 与 cross core flag 完成这里的 skip-scale 信号传递。



## 流水线

至此，Kernel 的总体设计方向已经定稿：64 个 head / AI Core，1C2V，output accumulator 放在 UB 上，KV 经由 UB 和 SIMD VF 完成 ND2NZ 的中转。

下一步便是流水线的设计了。鉴于这个 kernel 应该是 CUBE Core 的算力 bound，我们的目标就是：让 CUBE Core 上的矩阵乘法计算单元时时刻刻都能打满，并且不要让其他操作（比如 softmax）成为瓶颈。

最基本的流水思路大致是这样（下面使用中括号中的数字来指代这是第几个 KV block 对应的操作）：

1. 计算 `P[i] = Q @ K[i]^T`（`@` 代表矩阵乘法）

2. 计算 `O += S[i-1] @ V[i-1]`

3. 计算 `P[i+1] = Q @ K[i+1]^T`

4. 计算 `O += S[i] @ V[i]`

5. ...

也即，相当于将前一个块的 `O += S @ V` 延后计算，以给 softmax 操作留出时间：`P[i]` 在 (1) 结束后就已就绪，直到 (4) 开始时才会被用到，期间有充足的时间（大约 1024 周期）供 FixPipe 将 `P[i]` 从 L0C 拷贝至 UB、softmax、以及将 `S[i]` 从 UB 拷贝至 L1 使用。

index 的读取（从显存拷贝到 UB）、index 的预处理（预处理下文所说的 gather2 的 MTE 拷贝参数）、MTE2 拷贝的发起、以及 skip-scale 技术中的 SS buffer 信号同步则会让这个流水线变得复杂一些。CUBE Core 一侧的流水如上文所示，VECTOR Core 一侧的流水如下排布：

1. 发起第 `i+5` 个 KV 块的 index 读取

2. 发起第 `i+3` 个 KV 块的 ND2NZ 转换与 UB -> L1 的 MTE3 拷贝

3. 等待 `P[i+1] = Q @ k[i+1]^T` 完成，发起第 `i+1` 个 KV 块的 softmax 计算，并将结果

4. 发起第 `i+5` 个 KV 块的 index 预处理

5. 发起第 `i` 个 KV 块的 O scale，如果这一块没有 skip-scale O 的话

6. 和 SIMD VF 同步，获取第 `i+1` 个块是否需要 skip-scale O，将其写入 SS buffer

7. 发起第 `i+5` 个 KV 块的 MTE2 拷贝，将 KV 从显存拷贝至 UB

为何这个流水如此复杂？

- 一是因为，算子依靠 SS buffer 传递 skip-scale 的信号，但能读写 SS buffer 的只有 scalar core，而 skip-scale 的信号由 SIMD VF 生成，因此这会导致 SIMD VF -> Scalar Core 的阻塞式同步。倘若此处没有处理好，便会导致 scalar core 等待 SIMD VF 完成、其他单元出现空闲的情况。

- 二是因为，华为的各个硬件管线（PIPE）都是顺序发射、顺序执行的，比如对于 `PIPE_V` 来说，发射 SIMD VF 的顺序与执行的顺序必须完全一致，如果前面的 SIMD VF 由于某些条件不满足（比如 `asc_sync_wait` 条件尚不满足）而不能发射的话，就会阻塞整条硬件管线。

因此，我们需要谨慎设计各个操作的发起顺序，以让各种硬件资源得到充分的利用。



## 优化技巧

### gather2

本算子需要从显存中以 token 为粒度读取 KV。倘若我们为每个 KV 都发起一次 MTE2 拷贝请求，那么总拷贝请求数就会过多，超过 MTE2 的 Issue Queue 队列深度（16）。为此，我们使用了一种技巧：设置 MTE2 请求的 `burst_count=2`，`src_stride` 为两个 token 开始地址之差，这样就可以一次性拷贝两个 KV token 了，节约 MTE2 拷贝请求数量并降低发射 overhead。MTE2 拷贝的各个参数使用 SIMD VF 计算得到，避免阻塞 scalar 核心。

### Decoding KV Cache Layout 优化

在 [先前的 Flash MLA](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7) 中，KV Cache 中的每个块（KV Block）的前面是该块中所有 token 的 data（也即量化得到的 FP8 或 FP4），然后才是所有 token 的 scale。但是这种设计对于华为昇腾来说并不友好：每个 token 需要两次 MTE2 拷贝才能拿到 data 和 scale。所以，在本算子中，我们将一个 token 的 data 和 scale 相邻放置，这样就不必分别拷贝 data 和 scale 了。

### 高速 ND2NZ

前文提到，我们需要一个 SIMD VF 来完成 KV 的 ND2NZ layout 转换。这个 SIMD VF 的效率直接关系到整个算子的效率，特别是启动第一个 QK 矩阵乘法之前的“空窗期”的长度。为此，我们对该 SIMD VF 做出了如下优化：

- 在 SIMD VF 中，使用 `asc_loadalign` 读取 ND 格式的 KV，并使用 `vsstb` 将其中的每个 32B 数据块以一定间隔写回 UB，如此便完成了 ND2NZ 的转换。

- 在输出的 NZ buffer 中，额外 pad 一行，以保证写回时没有 UB 的 bank conflict。

- 使用 `asc_loadalign_postupdate` 与 `vsstb` 自带的“自增 pointer”的功能，免除更新指针的开销，防止 SIMD VF 的小 Scalar Core 成为新的瓶颈。

- 一次性处理一对相邻的 256B 元素，以利用满 UB 的所有 bank 的带宽。

最终该 VF 可以打满 UB 的读、写吞吐。

### icache miss 问题

华为昇腾 NPU 上的 Instruction Cache (icache) 大小相对有限，只有 32K (CUBE Core)、16K (VECTOR Core) 或 8K (SIMD VF)，因此需要注意控制代码的总体积，防止因为代码过大而导致指令缓存出现频繁的换入换出。值得一提的技巧包括在 SIMD VF 中使用硬件循环指令（`VLOOP`）而不是 `#pragma unroll` 展开循环、以及需要小心函数的内联会导致代码体积成倍数地增长（因此在 VECTOR Core 的结尾把 Systolic array 的 main loop 和 drain loop 给 fuse 在了一起）。

### L1 bank 配平

Ascend 950 NPU 的 L1 分为两个 bank，分别掌控着 0 - 256K 以及 256K - 512K 两段地址空间。为了充分利用这两个 bank 的带宽，我把每个 Tensor（比如 Q、量化好的 K 等）都切成两份，分别放在两个 bank 中，以随时随地地打满两个 bank 的带宽。

### skip-scale

Skip scale 技术具有许多好处，包括：

- 减少 FixPipe 的拷贝量与 VF 计算量

- 降低对 UB 读、写带宽的压力

- 通过减少计算与数据搬运量，降低算子运行时所需的功率，进而获得更高的频率并提高性能



