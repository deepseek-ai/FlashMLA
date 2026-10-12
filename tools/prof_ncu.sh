#!/bin/bash
# Nsight Compute driver (plan section 4).
#
#   tools/prof_ncu.sh <tag> [extra ncu options] -- <command...>
#
# env:  NCU_SET     full (default: --set full --import-source yes) | key (tools/ncu_key_metrics.txt, few passes) | roofline
#       NCU_KERNEL  kernel-name regex (function name, partial match). Default "(flash_fwd|mla_|sm86)(?!.*combine)" = main kernels,
#                   not the split-KV combine kernel; use "combine" for that one; triton baseline: "_fwd_grouped|_fwd_kernel_stage2"
#       NCU_SKIP    launches to skip (default 5), NCU_COUNT launches to profile (default 1)
#       NCU_CLOCK   base (default, stable comparisons) | none (absolute numbers)
#       PROF_GPU    physical GPU index (default 1); PYTHONPATH selects the flash_mla package.
# Source correlation needs a -lineinfo build:  FLASH_MLA_LINEINFO=1 tools/build.sh <worktree>
# Example:
#   PYTHONPATH=$WT tools/prof_ncu.sh v2_h16_b64 -- python benchmarks/bench_one.py --impl v2 --hq 16 --b 64 --sk 4096 --page 64
# Output: /data2/LiuJun/tmp/flashmla/ncu/<tag>.ncu-rep + .details.txt (full) + .raw.csv, key metrics printed at the end.
set -e
TAG=$1; shift
EXTRA=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do EXTRA+=("$1"); shift; done
if [ "$1" == "--" ]; then shift; fi
if [ $# -eq 0 ]; then sed -n 2,15p "$0"; exit 1; fi
HERE=$(cd "$(dirname "$0")" && pwd)
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data2/LiuJun/conda/envs/flashmla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${PROF_GPU:-1}
NCU=/opt/nvidia/nsight-compute/2024.1.1/ncu
OUT=/data2/LiuJun/tmp/flashmla/ncu/$TAG
mkdir -p /data2/LiuJun/tmp/flashmla/ncu
case ${NCU_SET:-full} in
  full)     SETARGS=(--set full --import-source yes) ;;
  roofline) SETARGS=(--section SpeedOfLight --section SpeedOfLight_RooflineChart --section MemoryWorkloadAnalysis) ;;
  key)      SETARGS=(--metrics "$(grep -v '^#' "$HERE/ncu_key_metrics.txt" | paste -sd, -)") ;;
  *) echo "bad NCU_SET"; exit 1 ;;
esac
set -x
$NCU "${SETARGS[@]}" --clock-control ${NCU_CLOCK:-base} -k "regex:${NCU_KERNEL:-(flash_fwd|mla_|sm86)(?!.*combine)}" \
     --launch-skip ${NCU_SKIP:-5} --launch-count ${NCU_COUNT:-1} -f -o "$OUT" "${EXTRA[@]}" "$@"
set +x
$NCU --import "$OUT.ncu-rep" --page raw --csv > "$OUT.raw.csv" 2>/dev/null || true
if [ "${NCU_SET:-full}" != "key" ]; then
  $NCU --import "$OUT.ncu-rep" --page details > "$OUT.details.txt" 2>/dev/null || true
fi
echo "ncu report: $OUT.ncu-rep"
python "$HERE/ncu_summary.py" "$OUT.raw.csv" || true
