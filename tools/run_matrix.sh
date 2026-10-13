#!/bin/bash
# The integration matrix of docs/ampere_opt_report.md (about one hour on one RTX 3090).  Needs the extensions built in place (tools/build.sh).
#   CUDA_VISIBLE_DEVICES=1 tools/run_matrix.sh [logdir]      then:  python benchmarks/summarize_matrix.py --out docs/bench_matrix.md
# CSVs are written to $FLASHMLA_LOG_DIR (default /data2/LiuJun/tmp/flashmla/logs); --cool keeps the GPU below its 83 C throttle target.
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data2/LiuJun/conda/envs/flashmla
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export FLASHMLA_COOL_HI=${FLASHMLA_COOL_HI:-78} FLASHMLA_COOL_LO=${FLASHMLA_COOL_LO:-66} FLASHMLA_COOL_MAX_S=${FLASHMLA_COOL_MAX_S:-30}
L=${1:-${FLASHMLA_LOG_DIR:-/data2/LiuJun/tmp/flashmla/logs}}
mkdir -p "$L"
BD="python benchmarks/bench_decode.py --modes eager,graph --check --cool"
FI="$HERE/run_flashinfer.sh --modes eager,graph --check --cool"
step() { name=$1; shift; echo "=== $name $(date +%T)"; "$@" > "$L/mx_$name.log" 2>&1; echo "exit=$? $(date +%T)"; }
# headline: the 18 main shapes at page 32, all four implementations back to back (legacy needs page 32)
step head_p32_bf16     $BD --preset main --page 32 --impls v2,legacy,triton --tag si_head_p32_bf16
step head_fi_p32_bf16  $FI --preset main --page 32 --tag si_head_fi_p32_bf16
# full matrix: h_q {8,16,32,64,128} x s_q {1,2} x b {1,4,16,64,128} x s_k {1024,4096,8192} (+ b=1 at 32768)
step p64_bf16          $BD --preset full --page 64 --impls v2,triton --tag si_p64_bf16
step p32_bf16          $BD --preset full --page 32 --impls legacy,v2 --tag si_p32_bf16
# 8-bit KV, fp16: core matrix (h_q {16,64,128}); the 8-bit runs keep a bf16 reference copy of layer 0, so give them a smaller layer budget
step p64_fp8           $BD --mem-gb 6 --preset core --page 64 --kv fp8 --impls v2 --tag si_p64_fp8
step p64_int8          $BD --mem-gb 6 --preset core --page 64 --kv int8 --impls v2 --tag si_p64_int8
step p64_fp16          $BD --preset core --page 64 --dtype float16 --impls v2,triton --tag si_p64_fp16
step p16_spot          $BD --preset spot --page 16 --impls v2,triton --tag si_p16_spot
# FlashInfer (vllm env)
step fi_p64_bf16       $FI --preset full --page 64 --tag si_fi_p64_bf16
step fi_p16_spot       $FI --preset spot --page 16 --tag si_fi_p16_spot
step fi_p64_fp16_core  $FI --preset core --page 64 --dtype float16 --tag si_fi_p64_fp16_core
echo MATRIXDONE
