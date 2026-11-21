#!/bin/bash
# Validation step A3: robustness of the measurement choices,
# public data only. Driver for exp/d3robust.py; another driver (exp/run_review.sh) runs A1/A2/A4/A6.
# Each GPU job starts when >= 6000 MB of GPU memory has stayed free for a minute and is retried
# after a CUDA out-of-memory. A job whose done-file exists is skipped. Two lanes run side by side:
#   eval   std (threshold sweep, disc, sanity vs d3zoo) -> large (16-25 mm positive control)
#          -> low-data control seeds 1-2
#   train  low-data control seed 0 -> mismatched-pairs control seeds 0-2
# then, once both lanes finish: controls through the retention protocol and on acquired input,
# and the summary (out_public/d3robust_T2.json, fig_public/robust*.pdf).
# The expected control outcomes are written by `d3robust.py prereg` before anything here runs.
#   usage: bash exp/run_review_A3.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
cd $ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
LOG=$ROOT/out_public/logs
O=out_public
st() { echo "$(date +%F_%T) $*" >> $LOG/review_A3_status.log; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { while true; do
    [ "$(gpu_free)" -ge 6000 ] && { sleep 60; [ "$(gpu_free)" -ge 6000 ] && return; }
    sleep 120; done; }
job() {   # job <name> <log> <done-file> <script and args...>
  local nm=$1 log=$2 done=$3; shift 3
  [ -e "$done" ] && { st "skip  $nm"; return 0; }
  for t in 1 2 3 4 5 6 7 8; do
    wait_gpu; st "start $nm"
    $PY "$@" > $log 2>&1 && { st "done  $nm"; return 0; }
    grep -q -E "OutOfMemoryError|CUDA out of memory" $log || break
    st "oom   $nm, retry $t"
  done
  st "FAIL  $nm (see $log)"; return 1
}

[ -f $O/d3robust_T2.json ] || { st "FAIL no preregistration; run exp/d3robust.py prereg first"; exit 1; }

lane_eval() {
  job std   $LOG/A3_std.log   $O/d3robust_rows_std_pub3t_hip_T2.npz   exp/d3robust.py std --models zoo || return 1
  job large $LOG/A3_large.log $O/d3robust_rows_large_pub3t_hip_T2.npz exp/d3robust.py large || return 1
  for s in 1 2; do
    job n3_s$s $LOG/base_unet_n3_seed${s}_T2.log $O/base_unet_n3_seed${s}_T2.pt \
        exp/d3robust.py train_lowdata --n_subj 3 --seed $s --steps 12000 || return 1
  done
}
lane_train() {
  job n3_s0 $LOG/base_unet_n3_seed0_T2.log $O/base_unet_n3_seed0_T2.pt \
      exp/d3robust.py train_lowdata --n_subj 3 --seed 0 --steps 12000 || return 1
  job mm_s0 $LOG/base_unet_mismatched_T2.log $O/base_unet_mismatched_T2.pt \
      exp/baselines.py --contrast T2 --method unet_mismatched --steps 12000 --seed 0 --tag unet_mismatched || return 1
  for s in 1 2; do
    job mm_s$s $LOG/base_unet_mismatched_seed${s}_T2.log $O/base_unet_mismatched_seed${s}_T2.pt \
        exp/baselines.py --contrast T2 --method unet_mismatched --steps 12000 --seed $s \
        --tag unet_mismatched_seed$s || return 1
  done
}

lane_eval > /dev/null 2>&1 & P1=$!
sleep 90                       # stagger, so the two lanes do not pass wait_gpu together
lane_train > /dev/null 2>&1 & P2=$!
wait $P1; r1=$?
wait $P2; r2=$?
[ $r1 -eq 0 ] || st "eval lane failed"
[ $r2 -eq 0 ] || st "train lane failed"
job std_ctl  $LOG/A3_std_ctl.log  $O/d3robust_rows_stdctl_pub3t_hip_T2.npz exp/d3robust.py std --models ctl || exit 1
job ssim_ctl $LOG/A3_ssim_ctl.log $O/d3robust_ssim_ctl_T2.json             exp/d3robust.py ssim_ctl || exit 1
st "start summary"
$PY exp/d3robust.py summary > $LOG/A3_summary.log 2>&1 || { st "FAIL summary (see $LOG/A3_summary.log)"; exit 1; }
st "done  summary"
[ $r1 -eq 0 ] && [ $r2 -eq 0 ] || exit 1
exit 0
