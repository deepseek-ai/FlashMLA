# A Deep Dive Into The Flash MLA FP8 Decoding Kernel on Hopper

With the release of DeepSeek-V3.2, we have doubled the context length of our models from 64K tokens to 128K tokens. This puts significant pressure on GPU memory (a single request with 128K tokens requires a KVCache of size $576 \times 2 \times 62 \times 128 \times 1024 = 8.72\ \mathrm{GiB}$), which can lead to out-of-memory (OOM) errors or under-utilized GPUs due to small batch sizes. To address this, we introduced FP8 KVCache for DeepSeek-V3.2.

However, writing a high-performance decoding kernel is challenging due to the need for dequantization and its sparse memory access patterns. In this blog, we share the story behind our new FP8 sparse decoding kernel for Hopper GPUs. We will first explain our FP8 KVCache format, then provide a theoretical analysis of clock cycles, and finally detail the techniques used in our new kernel.

## The FP8 KVCache Format

Recall that the decoding phase of the Multi-head Latent Attention (MLA) algorithm operates similarly to Multi-Query Attention (MQA), with 128 query heads and 1 key head, where `head_dim_k = 576` and `head_dim_v = 512` respectively. To reduce the size of the KVCache while maintaining accuracy, we use a fine-grained quantization method. Specifically, we apply tile-level quantization (with a tile size of $1 \times 128$) to the first 512 elements in each token's KV Cache. This results in 512 `float8_e4m3` values and 4 `float32` scale factors. For the remaining 64 elements (the RoPE part), we do not apply quantization as they are sensitive to precision loss. Therefore, in GPU memory, each token's KVCache occupies 656 bytes, consisting of 512 `float8_e4m3`s, 4 `float32`s, and 64 `bfloat16`s.

Inside the kernel, we first dequantize the 512 `float8_e4m3` values into 512 `bfloat16`s. We then concatenate them with the 64 original `bfloat16` values from the RoPE part. Finally, we perform the MQA calculation using matrix multiplication-add (MMA) operations in `bfloat16` precision (i.e., the inputs to the MMAs are in `bfloat16` and the outputs are in `float32`. This applies to both the QK gemm and the attention-score-V gemm).

## Theoretical Analysis of Clock Cycles

The main challenge is that Tensor Cores (which handle MMA calculations) are extremely fast, while the dequantization process, performed on CUDA Cores, struggles to keep up.

The basic unit on an NVIDIA GPU is the Stream Multiprocessor (SM). You can think of each SM as an independent core on the GPU. For simplicity, let's focus on a single SM. Each SM can process 4096 MMA Flops per clock cycle (calculated as `989 TFlops / 1830 MHz / 132 SMs` on H800). In our kernel, each CTA runs on one SM, and each SM is only mapped to one CTA. If we assign each CTA (CUDA Thread Block) to process 64 query heads, it only requires $64 \times (576+512) \times 2 / 4096 \approx 34$ cycles for MMA operations per K/V token.

However, because the H800 cannot directly cast `float8_e4m3` to `bfloat16`, dequantizing the KVCache for one token requires the following steps:
1.  Convert `float8_e4m3` to `half`
2.  Convert `half` to `float32`
3.  Convert `float32` to `bfloat16`
4.  Multiply the converted `bfloat16` value by the `float32` scale factor

According to [NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions), we need at least $(\frac{1}{64} + \frac{1}{64} + \frac{1}{16} + \frac{1}{256}) \times 512 \approx 50$ cycles for dequantizing each token! This is significantly more than the 34 cycles required for the MMA operations, meaning the kernel is **dequantization-bound**. If left unaddressed, dequantization would become the performance bottleneck, leaving the powerful Tensor Cores underutilized.

## Crossover

Before we continue, it's important to note a key fact: every query head within the same query token attends to the same key heads, because this is Multi-Query Attention (MQA).

Recall that each CTA processes 64 query heads, while DeepSeek-V3.2 has a total of 128 query heads. If we can find a way to "share" the dequantized K/V values between two CTAs that are processing different sets of query heads, then each CTA would only need to dequantize **half** of the KV cache – which is fantastic! We call this method "crossover", since the idea was actually inspired by [Chromosomal crossover](https://en.wikipedia.org/wiki/Chromosomal_crossover) during [Meiosis](https://en.wikipedia.org/wiki/Meiosis).

The next question is, how do we implement this in CUDA? Before NVIDIA's Hopper architecture, the only options for data exchange between CTAs were global memory or the L2 cache, which are slow. However, the powerful Distributed Shared Memory gave us a new solution.

## Distributed Shared Memory to the Rescue

Distributed Shared Memory (DSM) is a new feature introduced with the Hopper architecture, alongside the CTA Cluster (thread block cluster). CTAs within the same cluster can directly access each other's shared memory. For more details, you can refer to [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

Here is how we use it: We launch CTAs in clusters of size 2. Each CTA within a cluster is responsible for 64 query heads from the same query token. Each CTA performs the following steps:
1.  Loads *half* of the quantized K/V from global memory. We use a wide `__ldg` load with a width of 128 bits to improve performance.
2.  Dequantizes its assigned half on the CUDA Cores.
3.  Stores the dequantized K/V into its own shared memory.
4.  Simultaneously uses `st.async` to write the dequantized K/V into the shared memory of the other CTA in the cluster.

For synchronization between these operations, we rely on the [cluster transaction barrier](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), another powerful programming primitive available in CTA Clusters. After the data exchange is complete, each CTA has the *full* set of dequantized K and V values available in its own shared memory, which it can then use to perform the MMA operations.

## Performance
Using these techniques, we achieved 410 TFLOPS in a compute-bound configuration (batch_size=128, num_heads=128, s_q=2, topk=2048) on H800 SXM5 GPUs. This is a significant improvement over the 250 TFLOPS achieved by our previous FP8 sparse decoding kernel without the crossover technique. 

Although this number is still below the 640 TFLOPS peak of our previous bfloat16 dense decoding kernel, one reason is that it's a **sparse** kernel, and its topk is only 2048. With a smaller topk, the relative overhead of the kernel's prologue and epilogue becomes larger compared with dense decoding with long context length. If we set topk to a larger value, such as 32768, this kernel can achieve up to 460 TFLOPS.

From another perspective, the execution time of this kernel in the configuration mentioned above is comparable to that of the dense decoding kernel when the sequence length is around 3000. When the sequence length exceeds 3000, the performance advantage of our new kernel becomes even more significant. This also highlights the effectiveness of our DeepSeek Sparse Attention algorithm.
