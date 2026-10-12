#!/bin/bash
# Build the extension(s) in place for sm_86 with the flashmla conda env (nvcc 12.4 + torch 2.7.1+cu126).
# Usage: tools/build.sh [repo_or_worktree_dir]      Env: MAX_JOBS (default 12; keep <= nproc/2), NVCC_THREADS (default 2),
#        FLASH_MLA_LINEINFO=1 for ncu source correlation.   Use afterwards with PYTHONPATH=<dir>.
set -e
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data2/LiuJun/conda/envs/flashmla
cd "${1:-$(cd "$(dirname "$0")/.." && pwd)}"
export FLASH_MLA_ARCH=86 TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=${MAX_JOBS:-12} NVCC_THREADS=${NVCC_THREADS:-2}
echo "building in $(pwd): $(nvcc --version | tail -2 | head -1), torch $(python -c 'import torch;print(torch.__version__)')"
python setup.py build_ext --inplace
