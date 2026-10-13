#!/bin/bash
# Run benchmarks/bench_flashinfer.py (FlashInfer fa2 MLA baseline) in the read-only vllm conda env; nothing is installed.
#   tools/run_flashinfer.sh --preset main --page 32 --modes eager,graph --check --cool --tag mytag
# Env: CUDA_VISIBLE_DEVICES (default 1).  JIT cache: FLASHINFER_WORKSPACE_BASE (default /data2/LiuJun/tmp/flashmla/flashinfer_cache).
HERE=$(cd "$(dirname "$0")" && pwd)
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data2/LiuJun/conda/envs/vllm
export CUDA_HOME=/data2/LiuJun/cuda/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.6
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/data2/LiuJun/tmp/flashmla/flashinfer_cache}
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} MAX_JOBS=8
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
cd "$HERE/.." && python benchmarks/bench_flashinfer.py "$@"
