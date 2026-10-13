# FlashMLA on Ampere（RTX 3090, sm_86）MLA decode 优化报告

分支 `opt/sm86`（pzhao-eng/FlashMLA `a100_replace` 的 Ampere 移植之上），数据对应提交 `04b3338` / `1714846`。测量日期 2026-10-12 ~ 10-13。
所有数字都能在 `/data2/LiuJun/tmp/flashmla/{logs,ncu,nsys}` 的 CSV / 日志 / 报告里找到；完整矩阵见 [bench_matrix.md](bench_matrix.md)，
kernel 侧过程记录见 [perf_log_kernel.md](perf_log_kernel.md)，系统侧见 [perf_log_sys.md](perf_log_sys.md)。

## 0. 摘要

* 原 fork 的 sm80 kernel 在 3090 上**根本无法启动**（需要 113,280 B 共享内存，sm_86 上限 101,376 B），且只接受 32-token 页。
* 新写的 sm86 kernel（`impl="v2"`，默认）在整个矩阵上**最快或持平**：CUDA Graph 下几何均值，legacy / v2 = 1.07（h16）、1.17（h128）；
  Triton / v2 = 1.47、2.23；FlashInfer 0.6.6 / v2 = 3.43、1.77（page 32，bf16）。
* 带宽受限区（h_q=16，b≥64，s_k≥4096）达到只读峰值的 **94–99%**（graph 中位 847 GB/s）；计算受限区（h_q=128，b≥16，s_k≥4096）达到
  bf16 峰值的 **73–75%**（eager，中位 51.9 TFLOPS）/ 66–72%（graph）。FP16 实例 h_q=128 达 FP16 峰值的 77%。
* 可选 8-bit KV cache（FP8 e4m3 或 INT8，每 token 一个 fp32 scale，rope 维保留 bf16，0.559× 字节）在 h_q=16、b≥32 上
  eager **1.56–1.71×**、graph **1.47–1.58×**；可选的「每 SM 2 CTA」配置在长上下文下 graph 也达 **1.58–1.68×**。INT8 精度比 FP8 高 2.4–13 倍。
* DeepSeek-V2-Lite 单层 decode：融合 prologue + 整层 CUDA Graph 后，相对未融合 eager 快 **2.0×（b=64）~ 14×（b=1, s_k=1024）**，每层 kernel 数 40–43 → 8。
* 未达标：b≤4 且 KV < ~35 MB 时的延迟目标（graph 下超 2.8–4.0 µs，根因与已试方案见 §7），以及默认配置下 8-bit KV 在 graph 口径的 1.6×（可选配置只在长上下文达标）。
* 正确性：合并后 508 个测试全部通过；compute-sanitizer memcheck / racecheck / synccheck / initcheck 0 错误。

## 1. 背景与硬件上限

| 项 | 值 | 来源 |
|---|---|---|
| GPU | RTX 3090 ×4，82 SM，sm_86，每 block 共享内存上限 101,376 B | — |
| 只读带宽（decode 的上限） | **873.8 GB/s**（D2D copy 841.6 GB/s） | GPU1 实测，`benchmarks/phase0_peaks.py` |
| bf16 张量算力（FP32 累加） | **70.0 TFLOPS**（fp16 66.4）；GeForce 上 FP32 累加只有半速 | 8192³ matmul 实测 |
| 工具链 | nvcc 12.4.131 + torch 2.7.1+cu126，CUTLASS `afa1772`，只编 sm_86 | `/usr/local/cuda-11.8` 没有 nvcc，未安装新工具链 |
| 时钟 | 不能锁频（无 sudo），负载下 1.7–1.9 GHz；基准用 `--cool` 控温、记录 `sm_mhz` / 功耗 | — |

**MLA decode 的算术强度**：KV 每 token 576 维（512 latent + 64 rope），bf16 为 1152 B；每个 query 行每 token 做 QK 2·576 + PV 2·512 = 2176 FLOP。
强度 ≈ **1.89·M FLOP/B**（M = h_q·s_q），而 3090 的拐点是 70.0e12 / 873.8e9 ≈ 80 FLOP/B（M≈42）。所以 h_q=16（30 FLOP/B）是纯带宽受限，
h_q≥64 是算力受限；两个区的优化目标完全不同。

**原 fork 的问题**（`csrc/flash_fwd_mla_kernel_sm80.h`）：
1. Q(32×576) + 双缓冲 K + P + reduce ≈ 111 KB > 99 KB，`cudaFuncSetAttribute` 返回 invalid argument 后 `exit(1)`。
2. `page_block_size` 必须等于 32，接不了 vLLM（page 64）。
3. 内循环：warp 按 2M×4N 排，每个 warp 只算 16×8 的 S，Q 的 A 片段每次 MMA 都从 smem 重读；P 走 smem；每次迭代约 4 个 `__syncthreads`；只有 2 级流水。
   作者在 A100 上的数字是 464 GB/s（23%）/ 59 TFLOPS（19%）。
4. combine 是独立 kernel，metadata kernel 单线程串行，只支持 bf16。

**测量口径**（`benchmarks/bench_decode.py`）：eager = 单次调用，每次前刷 256 MB L2，CUDA event 计时（含 launch 开销），取中位；
graph = 一个 CUDA Graph 捕获 N 层（默认 27 = DeepSeek-V2-Lite），**每层用自己的 KV cache**（避免 L2 命中虚高），报每层时间。计时都在空闲的 GPU1 上。

## 2. 优化清单与实测效果

| # | 优化 | 位置 | 效果（实测） |
|---|---|---|---|
| 1 | legacy 回退：BlockM=16 / 4 warp，93.5 KB smem，0 spill | `csrc/flash_fwd_mla_kernel_sm80.h` | 能跑了；h16 b≥64 s_k≥4096 807–857 GB/s（92–98%），h128 b≥64 41–44 TFLOPS（60–63%）——作为基线 |
| 2 | **新 sm86 kernel v2**（设计见 §3） | `csrc/sm86/` | h128 比 legacy 快 1.15–1.24×；h16 小 batch 快 1.09–1.32×；h16 大 batch 持平（两者都在 DRAM 峰值） |
| 3 | 两套默认 tile：M≤16 → BM16 / Nt16 / 3 级；M>16 → BM32 / Nt32 / 2 级 | `csrc/sm86/mla_sm86.h` | BM32+Nt32 在 h_q≥32 比 Nt16 快 12–13%（h128 b64 s_k4096 1602 → 1394 µs，44 → 52 TFLOPS） |
| 4 | page 16 / 32 / **64**（vLLM 页大小），尾部 `cp.async` src-size=0 补零 | `csrc/sm86/` | page 64 与 page 32 持平（h16 b64 s_k4096 graph 370.6 vs 370.9 µs）；page 16 时 BM32 的 tile 被页大小限制为 16 token，h128 b64 s_k4096 比 page 64 慢 16–18% |
| 5 | FP16 实例 | `csrc/sm86/mla_decode_sm86_fp16.cu` | h128 b64 s_k4096 eager 1435 µs = 50.9 TFLOPS（FP16 峰值的 77%） |
| 6 | 并行 device 端 metadata（平衡切分 `floor(p·T/P)`，记录每段首个 seqlen）；graph-safe API（`out` / `lse` / `workspace` 预分配，无 host 同步） | `csrc/`、`flash_mla/flash_mla_interface.py` | 修掉 `ceil` 切分导致的空闲 SM（b1 s_k8192 原来只有 ~65/82 个 part 有活）；整个 decode 可被 CUDA Graph 捕获 |
| 7 | combine kernel 按行数选择变体（wide16 / wide8 / wide4 / warp-per-row） | `csrc/sm86/` | b1 9.2 → 5.5 µs；b32 s_k1024 11.7 → 5.2 µs；h16 b64 s_k4096 整体 387 → 374 µs |
| 8 | **8-bit KV cache**：FP8 e4m3 或 INT8 + fp32 per-token scale，rope 64 维保留 bf16；kernel 内用 LOP3/PRMT 位运算反量化（sm_86 没有 `cvt.e4m3`），MMA 仍是 bf16 | `csrc/sm86/mla_decode_sm86_q*.cu`、`docs/kv_quant_format.md` | 每 token 644 B（bf16 1152 B）；h16 b≥32 eager 1.56–1.71×、graph 1.47–1.58×（§4.3） |
| 8′ | 8-bit kernel「每 SM 2 CTA」可选配置：新增实例 `CfgQ<16,32,2,2>`，运行时 `FLASH_MLA_SM86_QCFG=16,2,2,32` | `csrc/sm86/mla_decode_sm86_q.cuh` | 长上下文 graph FP8 1.58–1.68×、INT8 1.57–1.62×；s_k≤2048 变慢，默认不开（§4.3） |
| 9 | **融合 decode prologue**：RoPE + RMSNorm + q 拼接 + 分页 KV 写入（可选写入时量化）一个 kernel | `csrc/fused/` | graph 内 1.8–14 µs，对比同样操作的 PyTorch 版 59–90 µs（graph）/ 552–562 µs（eager）；量化写入与 torch 参考逐位一致 |
| 10 | **整层 CUDA Graph**：DeepSeek-V2-Lite 单层 decode | `flash_mla/mla_decode_layer.py` | 对未融合 eager 2.0–14×；每层 kernel 40–43 → 8（§5） |
| 11 | 基准与 profiling 工具：eager/graph 计时、四方对比、矩阵汇总、ncu/nsys 脚本 | `benchmarks/`、`tools/` | — |

## 3. v2 kernel 设计要点

* **Q 放寄存器，不进 smem**：CTA = BM 行（head × s_q）× 整段 KV。BM16 是 4 个 warp；BM32 是 8 个 warp = 2 个 M 组 × 4。
  组内 warp j 负责 QK 的维度片 [144j, 144j+144)（9 个 k16 步），Q 的这一片作为 A 片段常驻寄存器，只加载一次；PV 负责 V 的列 [128j, 128j+128)，
  O 累加器 16×128 fp32 = 64 个寄存器。省下的 smem 全部给 KV 多级环形缓冲。
* **CTA 内 QK split-K**：每个 warp 把自己的部分 S 写到双缓冲 smem，组内 named barrier（`bar.sync id,128`）后按固定顺序求和，结果在各 warp 间逐位一致；
  softmax 在组内冗余计算（quad shuffle），**P 直接在寄存器里转成 bf16 A 片段**，不经过 smem。
* **流水**：所有 warp 发 `cp.async`，多级 `wait_group`，每级一个 CTA barrier（FA2 风格）；smem 行做 XOR swizzle，`ldmatrix` / `ldmatrix.trans` 无 bank conflict；
  页号提前一个 tile 预取（b128 s_k4096：832 → 734 µs）。
* **资源**：BM16 187 寄存器 / 59.4 KB smem；BM32 212 寄存器 / 90 KB；8-bit kernel 218 寄存器 / 49.4 KB；所有实例 0 spill。
* **split-KV**：沿用 FlashMLA 的持久 CTA 调度（每个 part 一段 (batch, token) 区间），82 个 part（BM16）或 4 个 m-block × 20 个 part（h128，同一 KV 被 4 个 m-block 经 L2 共享）。
  partial O / LSE 写入预分配 workspace，再由 combine kernel 合并。

## 4. Kernel 级前后对比

### 4.1 主形状四方对比（page 32，bf16，CUDA Graph，每层 µs；GPU1）

| 形状 | v2 | legacy | Triton | FlashInfer | legacy/v2 | Triton/v2 | FlashInfer/v2 |
|---|---|---|---|---|---|---|---|
| h16 b1 s_k1024 | 9.4 | 10.3 | 15.4 | 25.5 | 1.09 | 1.63 | 2.70 |
| h16 b1 s_k4096 | 15.9 | 21.0 | 27.0 | 62.8 | 1.32 | 1.70 | 3.95 |
| h16 b1 s_k8192 | 22.9 | 27.9 | 37.0 | 83.8 | 1.22 | 1.62 | 3.66 |
| h16 b64 s_k4096 | 370.9 | 371.3 | 529.6 | 1405.2 | 1.00 | 1.43 | 3.79 |
| h16 b128 s_k8192 | 1404.4 | 1410.8 | 1928.5 | 5144.6 | 1.00 | 1.37 | 3.66 |
| h128 b1 s_k8192 | 58.8 | 69.5 | 128.8 | 90.2 | 1.18 | 2.19 | 1.54 |
| h128 b64 s_k4096 | 1499.5 | 1779.9 | 3742.6 | 2631.0 | 1.19 | 2.50 | 1.75 |
| h128 b64 s_k8192 | 2885.2 | 3509.2 | 7733.4 | 5134.6 | 1.22 | 2.68 | 1.78 |
| h128 b128 s_k8192 | 5820.5 | 6991.5 | 15777.2 | 10254.9 | 1.20 | 2.71 | 1.76 |

几何均值（最小–最大）`impl / v2`：

| 口径 | h_q | legacy | Triton | FlashInfer |
|---|---|---|---|---|
| graph，page 32 | 16 | 1.07（1.00–1.32） | 1.47（1.34–1.70） | 3.43（2.70–4.18） |
| graph，page 32 | 128 | 1.17（1.02–1.23） | 2.23（1.17–2.71） | 1.77（1.54–2.18） |
| eager，page 32 | 16 | 1.03（0.93–1.25） | 1.38（1.20–1.52） | 3.11（2.07–3.98） |
| eager，page 32 | 128 | 1.16（1.01–1.24） | 2.22（1.13–2.64） | 1.77（1.39–2.13） |
| graph，page 64 全矩阵 | 8 / 16 / 32 / 64 / 128 | — | 1.41 / 1.63 / 2.07 / 2.29 / 2.46 | 3.17 / 2.87 / 2.20 / 1.80 / 1.73 |

* 唯一慢于 legacy 的点是 eager h16 b1 s_k1024（14.3 vs 15.4 µs，0.93×）：KV 只有 1.2 MB，时间几乎全是 launch 与固定开销；graph 下 v2 反超（9.4 vs 10.3）。
* h16 b≥64 两者持平是预期的：legacy 修好以后已经在 DRAM 峰值的 92–98%，没有余量（见 §6 ncu：两者读同样的 304 MB，时间差 0.2%）。
* FP16 与 bf16 基本相同（h128 b64 s_k4096 eager 1435 vs 1421 µs）；Triton / FlashInfer 对 v2 的比值在 fp16 下同样成立（graph 几何均值 h16 1.64 / 2.86，h128 2.42 / 1.71）。

### 4.2 记分卡输入（v2，bf16，page 64）

| 区间 | 实测 | 占实测峰值 |
|---|---|---|
| h16 b≥64 s_k≥4096，eager | 814–862 GB/s（中位 843） | 93–99% |
| h16 b≥64 s_k≥4096，graph | 821–864 GB/s（中位 847） | 94–99% |
| h128 b≥16 s_k≥4096，eager | 50.8–52.6 TFLOPS（中位 51.9） | 73–75% |
| h128 b≥16 s_k≥4096，graph | 46.5–50.6 TFLOPS（中位 49.9） | 66–72% |

h128 的 graph 口径比 eager 慢约 4%：两者 SM 时钟采样中位相近（1800 / 1793 MHz），但 graph 连续运行时功耗 318 W vs 269 W，
推测是算力受限 kernel 在持续负载下的功耗/时钟波动（未定论）。

### 4.3 8-bit KV（h_q=16，s_q=1，page 64，GPU1，`--cool`，8 层 graph；同一 harness 同一 GPU 下 bf16 vs 8-bit）

| b | s_k | bf16 eager / graph µs | FP8 eager / graph µs（加速比） | INT8 eager / graph µs（加速比） |
|---|---|---|---|---|
| 32 | 4096 | 196.6 / 190.9 | 126.0 / 121.6（1.56× / 1.57×） | 121.8 / 125.1（1.61× / 1.53×） |
| 32 | 8192 | 368.6 / 364.4 | 216.1 / 230.8（1.71× / 1.58×） | 223.2 / 237.7（1.65× / 1.53×） |
| 64 | 4096 | 374.8 / 370.4 | 222.2 / 234.9（1.69× / 1.58×） | 225.3 / 242.4（1.66× / 1.53×） |
| 64 | 8192 | 714.8 / 712.2 | 431.1 / 458.9（1.66× / 1.55×） | 447.5 / 475.6（1.60× / 1.50×） |
| 128 | 4096 | 722.9 / 719.8 | 432.9 / 462.3（1.67× / 1.56×） | 450.6 / 479.5（1.60× / 1.50×） |
| 128 | 8192 | 1403.9 / 1402.4 | 865.3 / 911.5（1.62× / 1.54×） | 897.0 / 943.7（1.57× / 1.49×） |

* 小 batch：graph 几何均值 b≤4 1.24×（1.08–1.39），b=16 1.45×——固定开销占比大，字节减半帮助有限。
* **h_q≥64 不要用 8-bit**：0.93–0.97×。那里本来就是算力受限，反量化只会再加 ALU 负担。
* 8-bit kernel 不是纯带宽受限：ncu 显示 DRAM 只到 86.6%，最大的 stall 是 math-pipe throttle（32%），即反量化指令在抢 ALU（§6）。
  graph 口径比 eager 慢 ~6.6%（时钟低约 3%），这也是 graph 下没到 1.6× 的直接原因。
* 口径说明：开发阶段曾记录 1.73–1.75×，那组数据的 bf16 基准是在被其他用户任务占用的 GPU0 上测的（b128 s_k4096 bf16 772.6 µs，干净 GPU1 上 719.8 µs），
  分母偏大；以本表为准。

**可选：每 SM 2 个 CTA**（`FLASH_MLA_SM86_QCFG=16,2,2,32`）。tile 与默认相同（BM16 / Nt32 / 2 级，218 寄存器、49.4 KB smem），
只是 split part 数从 82 翻倍到 164，让每个 SM 同时驻留 2 个 CTA（8 个 warp）去掩盖反量化的延迟。graph，h_q=16，对 bf16 的加速比：

| b | s_k | FP8 默认 | FP8 2 CTA/SM | INT8 默认 | INT8 2 CTA/SM |
|---|---|---|---|---|---|
| 4 | 4096 | 1.38× | 1.20× | 1.36× | 1.20× |
| 16 | 4096 | 1.53× | 1.44× | 1.50× | 1.41× |
| 32 | 8192 | 1.58× | **1.65×** | 1.53× | **1.61×** |
| 64 | 4096 | 1.58× | **1.61×** | 1.53× | **1.60×** |
| 64 | 8192 | 1.55× | **1.68×** | 1.50× | **1.62×** |
| 128 | 4096 | 1.56× | 1.58× | 1.51× | 1.57× |
| 128 | 8192 | 1.54× | **1.61×** | — | 1.58× |

适用区间（FP8，graph，默认时间 / 2 CTA/SM 时间，>1 表示 2 CTA/SM 更快）：

| b（行）× s_k（列） | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 0.87 | 0.94 | 0.94 | 1.02 | 1.05 |
| 32 | 0.93 | 0.94 | 0.98 | 1.04 | 1.09 |
| 48 | 0.91 | 0.94 | 1.01 | 1.07 | 1.08 |
| 64 | 0.94 | 0.97 | 1.03 | 1.09 | 1.06 |
| 96 | 0.94 | 0.98 | 1.05 | 1.11 | — |
| 128 | 0.91 | 1.00 | 1.05 | 1.05 | — |

收益由**每条序列的长度**决定：part 翻倍后短序列被切得更碎，每段的固定开销和 partial / combine 流量抵消了延迟掩盖的收益。
eager 的门槛更高（s_k=4096 时 0.94–1.01，s_k≥8192 且 b≥64 时 1.02–1.08）。自动选择需要实际 s_k，而它只在 GPU 上（API 不做 host 同步，才能被 graph 捕获），
所以默认配置不变；以长上下文为主的服务（多数请求 s_k≥4096、b≥48）建议打开。正确性由 `test_quant_two_ctas_per_sm` 覆盖（FP8/INT8 × bf16/fp16 × page 16/32/64）。

**精度**（单层，反量化参考 vs bf16 cache；合成的 MLA 形状 cache，3 种分布 × ctx 1k/4k/16k × q 锐度 ×1/×4 × 3 个种子；服务器上没有真实权重）：

| 格式 | 每 token 字节 | 写入路径 vs torch 参考 | cos_diff（attention 输出） | cos_diff（整层输出）最坏 | 门限 |
|---|---|---|---|---|---|
| bf16 | 1152 | — | 0 | 0 | 8e-5（无损门限） |
| FP8 e4m3 + fp32 scale，rope bf16 | 644（0.559×） | 逐位一致 | 5e-5 ~ 2.4e-3 | 2.4e-3 | < 5e-3 |
| INT8 + fp32 scale，rope bf16 | 644（0.559×） | 逐位一致 | 6e-6 ~ 2.5e-4 | 2.6e-4 | < 5e-4 |

同样字节数下 INT8 比 FP8 准 2.4–13 倍，速度几乎一样，建议优先 INT8；两者都是可选项。

## 5. 整层：DeepSeek-V2-Lite decode layer

结构：q_proj / kv_a_proj → W_UK 吸收 bmm → **融合 prologue** → attention → W_UV bmm → o_proj。h_q=16，每层自己的 KV cache，27 层 graph，GPU1。

### 5.1 每层时间（µs，s_k=4096）

| b | 未融合 eager | 未融合 graph | 融合 graph：legacy (p32) | v2 bf16 | v2 FP8 | v2 INT8 | Triton (p64) | 融合 graph v2 对未融合 eager |
|---|---|---|---|---|---|---|---|---|
| 1 | 820.1 | 114.7 | 71.1 | **65.8** | 66.1 | 65.4 | 76.4 | 12.5× |
| 8 | 895.1 | 184.9 | 115.7 | 117.9 | 102.3 | **101.5** | 133.8 | 7.6× |
| 32 | 903.1 | 330.0 | 258.2 | 259.3 | **194.3** | 194.8 | 355.7 | 3.5× |
| 64 | 899.2 | 515.0 | 440.9 | 440.0 | **310.5** | 322.6 | 599.6 | 2.0× |

s_k=1024：b1 未融合 eager 825.8 → 融合 graph v2 58.2 µs（**14.2×**）；b64 890.0 → 177.4 µs（5.0×），FP8 142.7 µs。（未融合列 = v2 attention + PyTorch 实现的 prologue；
换成 legacy attention 时这一列在 ±2% 内相同，因为 eager 下时间几乎全是 launch 开销。）

* **eager 是 launch 受限**：未融合层约 40 次 launch，不管 batch 多大都在 820–900 µs。整层 graph 去掉 launch 开销（b1 820 → 115 µs），
  融合再去掉零碎的 elementwise kernel 和它们之间的空隙（115 → 66 µs，1.74×）。
* batch 变大后 attention 主导，融合的边际收益降到 1.17×（b64），这时 8-bit KV 才是主要手段：b64 s_k4096 再快 1.42×（440 → 310.5 µs）。
  对 8-bit 自己的未融合 eager 层（多了量化 kernel，每层 56 个）是 1255.8 → 310.5 µs，4.0×。
* 融合 graph 下 v2 对 Triton 快 1.13–1.37×；对 legacy 在 b1 快 8%，b≥8 持平（±2%，page 32 vs 64）。

### 5.2 nsys：每层 kernel 构成（b=8，s_k=4096，8 层 graph，`--cuda-graph-trace=node`）

| 版本 | kernel/层 | attention | combine | prologue | 4 个 GEMM | 其它 | busy/层 |
|---|---|---|---|---|---|---|---|
| 未融合（legacy attention，page 32） | ~44 | 51.5 | 4.2 | （拆成 PyTorch 小 kernel，计入「其它」） | 67.8（5 个 GEMM） | ~80（~36 个 elementwise / copy / index） | ~203 µs |
| 融合 v2 bf16 | 8 | 51.6 | 6.7 | 2.5 | 60.4 | 2.3 | ~123.5 µs |
| 融合 v2 FP8 | 8 | 33.8 | 6.4 | 4.0（含量化写入） | 58.7 | 2.3 | ~105 µs |

另外每个 decode step（所有层共用）跑 1 次 metadata kernel（4.2 µs）。融合后 GEMM 占一半左右，attention 约 42%；再往下的整层收益要靠 GEMM 侧
（例如把 W_UK 吸收 bmm 与 q_proj 合并、或 o_proj 的 split-K），不在本项目范围。

## 6. ncu / nsys 证据

ncu `--set full --clock-control base`（基础时钟，便于横向比较；绝对时间比 boost 下长），GPU1，page 64，bf16 除非注明。

| 形状 / kernel | 时间 | 关键指标 | 结论 |
|---|---|---|---|
| h16 b64 s_k4096，v2（BM16/Nt16/3 级，82 CTA × 128 线程） | 365.2 µs | 读 304.4 MB（= 833 GB/s，95%）；tensor 16%；warps active 8.4%；smem ld bank conflict 0 | 纯带宽受限，已在峰值 |
| 同上，legacy | 365.8 µs | 读 304.4 MB；93.5 KB smem；smem ld 波前 9.74M（v2 6.55M） | 与 v2 持平，legacy 的 smem 流量多 49% 但被带宽掩盖 |
| h16 b64 s_k4096，v2 FP8 | 222.3 µs（1.64×） | 读 171.2 MB；DRAM 86.6%；stall：math-pipe throttle 32.1%、wait 13.5%、long scoreboard 5.0% | 反量化 ALU 受限，DRAM 没吃满 |
| h128 b64 s_k4096，v2（BM32/Nt32/2 级，4×20 CTA × 256 线程） | 1.389 ms | tensor 38.2%（该指标以 FP16 累加速率为 100%，FP32 累加上限约 50%，即 ≈76% 的可达上限）；L2 命中 74.7%（4 个 m-block 共享 KV） | 算力受限，接近可达上限 |
| 同上，legacy | 1.523 ms | tensor 31.6%；L2 命中 86.5% | v2 快 1.10×（基础时钟） |
| h16 b1 s_k8192，v2 主 kernel + combine | 21.9 + 7.6 µs | 主 kernel 读 9.5 MB；combine 64 CTA × 512 线程 | combine 约占 26%，是小 batch 的主要固定开销之一 |

**小 batch 时间分解**（globaltimer trace，`-DMLA_SM86_TRACE`，b1 s_k8192，82 CTA）：CTA 中位在 **4.1 µs** 时才拿到第一个 tile
（metadata → block table → `cp.async` → DRAM，三次串行访存 + launch），之后以 DRAM 峰值流式读取约 100 token，16–18 µs 结束；尾部（最后 tile + partial O 写回 + 不均衡）约 3 µs；
combine 5.3–5.9 µs（一个空 kernel 就要 1.5 µs）。也就是说 kernel 一旦跑起来已经在 DRAM 极限，b≤4 的差距全是固定开销。

**BM16 的计算侧**（L2 常驻数据，排除 DRAM）：36 TFLOPS 等效，tensor 管线活跃 26%（≈ FP32 累加上限的 52%）；CPI 4.4 = 选中 1.0 + math-pipe throttle 1.48 +
固定延迟等待 0.95 + short scoreboard 0.34 + long scoreboard 0.20 + barrier 0.16。剩下的是依赖等待，只有更多 warp 才能藏住（但 2 CTA/SM 会让 split 数翻倍，见 §7）。

报告文件：`/data2/LiuJun/tmp/flashmla/ncu/si_*.ncu-rep`（含 `.details.txt`、`.raw.csv`），`/data2/LiuJun/tmp/flashmla/nsys/si_layer_*.nsys-rep`。

## 7. 没有成功 / 放弃的方案

| 方案 | 结果 | 原因 | 何时值得重试 |
|---|---|---|---|
| 原 fork kernel 不改直接跑 | 启动失败 | smem 113,280 B > 101,376 B | — |
| legacy warp-specialized 版（`legacy_ws`）最小修改 | 放弃，sm_86 上抛 RuntimeError | 8-warp 2M×4N 的 PV tiling 写死了 BlockM=32 | — |
| **v2b**：producer warp + mbarrier warp specialization | 正确，但不更快 | 带宽受限区已在峰值；计算区瓶颈是依赖等待，不是 barrier | 有 TMA/wgmma 的硬件（sm_90） |
| **v2c**：软件流水 | 163 → 157 µs（L2 常驻），真实形状无收益 | ptxas 已经在循环内交错 HMMA / MUFU / LDSM | — |
| **融合 last-arriver combine**（计数器自复位，确定性） | b128 750.6 vs 743.4 µs（噪声）；b32 s_k1024 73.7 → **81.9 µs** | fence + atomic + 单 SM 串行合并 96–640 KB partial 落在每个 split 的关键路径上；超过上限的行仍要 combine launch，省不掉 launch | 保留为 `FLASH_MLA_SM86_FUSE`，默认关 |
| cooperative in-kernel combine（grid barrier + 分布式合并） | 只做了估算，未实现 | b≤2 最多省 ~3 µs，需要 cooperative launch，按行数扩展差 | sm_90 cluster / DSMEM |
| 2 CTA/SM（bf16 kernel，BM16） | L2 常驻 133 vs 163 µs，真实形状不划算 | split 数翻倍，partial 流量和 combine 变大 | — |
| 调度参数 OV / MINBLK 扫描 | 影响 < 2% | 固定开销不在切分粒度上 | — |
| h16 的 tile 配置扫描（16/3、16/4、2 CTA×16/2、32/2…） | 差异 < 1.5% | DRAM 受限后 tile 配置无关紧要 | — |
| 8-bit KV 用于 h_q≥64 | 0.93–0.97× | 算力受限，反量化增加 ALU | 有 FP8 张量核的硬件（sm_89+） |
| 只给 decode kernel 做 CUDA Graph，b≥64 | eager ≈ graph | launch 开销被几百 µs 的 kernel 摊薄；graph 的价值在小 batch 和整层 | — |
| 在 h16 大 batch bf16 上追求超过 legacy | 无余量 | 修好的 legacy 已在峰值的 92–98% | — |

## 8. 记分卡（对照计划 §5 目标）

| 目标 | 实测 | 判定 |
|---|---|---|
| h16 s_q1 b64–128 s_k4k–8k ≥ 80% 实测带宽（≈700 GB/s） | graph 821–864 GB/s（中位 847，94–99%）；eager 93–99% | ✅ |
| h128 ≥ 60% bf16 峰值（≥42 TFLOPS） | eager 50.8–52.6 TFLOPS（73–75%）；graph 46.5–50.6（66–72%）；fp16 77% | ✅ |
| b≤4 延迟 ≤ 1.3 ×（KV 字节 / 带宽）+ 5 µs | graph：KV ≥ 37.7 MB 达标（b1 s_k32768、b4 s_k8192）；KV ≤ 18.9 MB 超 2.8–4.0 µs；eager 全部超（2.3–8.6 µs） | ❌（部分） |
| FP8 KV 带宽受限区 ≥ 1.6× | 默认：eager 1.56–1.71×（6 个形状中 5 个 ≥1.6）、graph 1.47–1.58×；可选 2 CTA/SM：长上下文 graph 1.58–1.68× | ◐ |
| 全矩阵胜过 Triton / legacy / FlashInfer | 最快或持平；唯一例外 eager h16 b1 s_k1024 对 legacy 0.93× | ✅ |
| 正确性 | 合并后 508 测试通过（sm86 140、quant 102、fused prologue 24、fused layer 8、kv_quant 18、整层 e2e 152、fork sm80 64）；sanitizer 0 错误；graph 回放改 seqlen 正确 | ✅ |

b≤4 的超出部分 = 启动 ~4 µs + 尾部 ~3 µs + combine ~5 µs 的固定成本（§6），在 Ampere 上没有 PDL / cluster 可以重叠，剩余可削空间估计 1–3 µs。

## 9. 构建与使用

```bash
# 环境：nvcc >= 12.x（测试用 nvcc 12.4 + torch 2.7.1+cu126），只编 sm_86，原地构建
FLASH_MLA_ARCH=86 python setup.py build_ext --inplace        # 或 tools/build.sh
#   FLASH_MLA_LINEINFO=1 加 -lineinfo（ncu 源码视图）；FLASH_MLA_DEFINES="FLASH_MLA_SM86_NT_SMALL=32 ..." 设置 kernel 可调参数
python -m pytest tests/test_flash_mla_sm86.py tests/test_flash_mla_sm86_quant.py tests/test_fused_prologue.py \
    tests/test_fused_layer.py tests/test_kv_quant.py tests/test_decode_layer_e2e.py tests/test_flash_mla_sm80.py -q
```

```python
import torch, flash_mla
md, nsplit = flash_mla.get_mla_metadata(cache_seqlens, s_q * h_q, 1)          # device 端，无 host 同步
ws = flash_mla.alloc_mla_workspace(b, s_q * h_q, 1, md.shape[0])               # 一次分配，各层各步复用
o, lse = flash_mla.flash_mla_with_kvcache(
    q, kv_cache, block_table, cache_seqlens, 512, md, nsplit,
    softmax_scale=(128 + 64) ** -0.5,      # DeepSeek：吸收前的 head dim，不是 576 ** -0.5
    causal=s_q > 1, out=out, lse=lse, workspace=ws,   # 给定 out/lse/workspace 即可被 CUDA Graph 捕获
    impl="v2")                             # 默认；"legacy" = fork kernel 的 BlockM=16 回退（仅 page 32、bf16）

# 8-bit KV：uint8 [num_blocks, page, 640] + fp32 scale [num_blocks, page]，q 仍为 bf16/fp16
kvq, scale = flash_mla.quantize_kv(kv_native.view(-1, page, 576), "int8")      # 或 "fp8"
md, nsplit = flash_mla.get_mla_metadata(cache_seqlens, s_q * h_q, 1, kv_dtype="int8")
o, lse = flash_mla.flash_mla_with_kvcache(q, kvq, block_table, cache_seqlens, 512, md, nsplit,
                                           softmax_scale=(128 + 64) ** -0.5, kv_scale=scale, kv_dtype="int8")

# 长上下文的 8-bit 服务可选：进程启动前 export FLASH_MLA_SM86_QCFG=16,2,2,32（每 SM 2 CTA，适用区间见 §4.3）

# 整层：from flash_mla import MLAConfig, MLADecodeLayer, alloc_cache, make_state, capture_layer_graph
```

约束：`impl="v2"` 要求 d=576、d_v=512、K/V 共享 cache、cache 行 16 字节对齐；page ∈ {16, 32, 64}；8-bit 只走 v2 且 h_q 建议 ≤ 32。

复现基准：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD \
  python benchmarks/bench_decode.py --preset core --impls v2,legacy,triton --modes eager,graph --cool
tools/run_flashinfer.sh ...                       # FlashInfer 在独立的 vllm 环境里跑，同样的 CSV 格式
python benchmarks/bench_layer.py --mode graph     # 整层
python benchmarks/summarize_matrix.py             # 生成 docs/bench_matrix.md
tools/prof_ncu.sh <tag> -- python benchmarks/bench_one.py --impl v2 --hq 16 --b 64 --sk 4096 --page 64
tools/prof_nsys.sh <tag> -- python benchmarks/bench_layer.py --mode graph --iters 20
```

## 10. 剩余想法（按预期收益排序）

1. **8-bit 自动选择 part 数**：2 CTA/SM 的收益只取决于实际 KV 长度，而它只在 GPU 上。可以让 metadata kernel 按实际总 token 数与平均 s_k
   在 82 / 164 个 part 之间选择，decode 的 grid 固定为 164、多余的 CTA 直接退出。这样默认就能拿到长上下文的 +3–11%，短上下文不受影响。
2. **8-bit 反量化减负**：per-token scale 已经在 MMA 外乘到 S / P 上；剩下的转换开销来自每个 latent 元素被转换两次（QK 用 `ldmatrix`、PV 用
   `ldmatrix.trans` 各从 8-bit smem 读一次再转换）。可以试每个 tile 只转换一次、写成 bf16 smem 共享，用 smem 带宽换 ALU。INT8 现在经 fp32 魔数
   （PRMT + FADD 再打包），比 FP8 的位运算 + `fma.bf16x2` 略贵，与 INT8 在 graph 下略慢一致。
3. **b≤4 固定开销**：把 metadata 计算并进主 kernel 或提前预取，省一次串行访存（估计 0.5–1 µs）；b≤2 专用的小 combine 路径。
4. **h_q=8**：swap-AB，让 m16 不浪费一半的 MMA 行。
5. **整层 GEMM 侧**：融合后 GEMM 占层时间的一半，可以考虑 W_UK 吸收与 q_proj 合并、小 batch 的 split-K GEMM。
6. **真实权重精度评估**：目前 8-bit 精度用的是合成统计数据（刻意对 per-token scale 不友好），有 DeepSeek 权重后应复测。
7. **接入 vLLM**：v2 支持 page 64 与 graph-safe workspace，可以作为 vLLM 的 MLA decode 后端在 3090 上做端到端吞吐评估。

## 附：数据来源

| 内容 | 文件 |
|---|---|
| 峰值、基线、融合层、量化精度 | `docs/perf_log_sys.md`，`logs/phase0_*`、`logs/bench_*` |
| kernel 设计过程、ncu、trace | `docs/perf_log_kernel.md`，`ncu/sk_*`、`nsys/sk_*` |
| 四方矩阵、记分卡、fp16、page 16、8-bit | `docs/bench_matrix.md`（`benchmarks/summarize_matrix.py` 由 `logs/bench_si_*.csv` 生成） |
| 8-bit 同条件复测 | `logs/bench_audit8_L{8,27}_{native,fp8,int8}_*.csv`，`logs/mc_audit8_summary.csv` |
| 8-bit 2 CTA/SM | `logs/bench_qm{,2,3}_*.csv`，`logs/mc_qminb{,2,3}_summary.csv` |
| 整层各后端 | `logs/si_mx_layerB_*.log` |
| ncu / nsys | `ncu/si_*`，`nsys/si_layer_*` |

（路径相对 `/data2/LiuJun/tmp/flashmla/`。）
