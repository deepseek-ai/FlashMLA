#!/bin/bash
# Nsight Systems driver (plan section 4): timeline with CUDA + NVTX, CUDA-graph kernel nodes traced individually.
#
#   tools/prof_nsys.sh <tag> -- <command...>
#
# env:  PROF_GPU physical GPU index (default 1); PYTHONPATH selects the flash_mla package;
#       NSYS_BREAKDOWN_ARGS extra args for tools/nsys_breakdown.py (e.g. "--nvtx decode_step").
# Output (/data2/LiuJun/tmp/flashmla/nsys/): <tag>.nsys-rep, <tag>.sqlite, <tag>_cuda_gpu_kern_sum.csv, <tag>_nvtx_sum.csv,
#       and the breakdown printed by tools/nsys_breakdown.py (per-kernel time, idle gaps between kernels).
# Examples:
#   tools/prof_nsys.sh layer_graph -- python benchmarks/bench_layer.py --mode graph --iters 20
#   PYTHONPATH=$WT tools/prof_nsys.sh v2_h16 -- python benchmarks/bench_one.py --impl v2 --hq 16 --b 64 --sk 4096 --graph
set -e
TAG=$1; shift
if [ "$1" == "--" ]; then shift; fi
if [ $# -eq 0 ]; then sed -n 2,13p "$0"; exit 1; fi
HERE=$(cd "$(dirname "$0")" && pwd)
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data2/LiuJun/conda/envs/flashmla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${PROF_GPU:-1}
NSYS=/usr/local/bin/nsys
D=/data2/LiuJun/tmp/flashmla/nsys
mkdir -p $D
set -x
$NSYS profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite=true -o "$D/$TAG" "$@"
set +x
$NSYS export --type sqlite --force-overwrite=true -o "$D/$TAG.sqlite" "$D/$TAG.nsys-rep" > /dev/null
$NSYS stats --force-overwrite=true --report cuda_gpu_kern_sum --format csv -o "$D/$TAG" "$D/$TAG.nsys-rep" > /dev/null 2>&1 || true
$NSYS stats --force-overwrite=true --report nvtx_sum --format csv -o "$D/$TAG" "$D/$TAG.nsys-rep" > /dev/null 2>&1 || true
python "$HERE/nsys_breakdown.py" "$D/$TAG.sqlite" ${NSYS_BREAKDOWN_ARGS:-} || true
echo "nsys report: $D/$TAG.nsys-rep"
