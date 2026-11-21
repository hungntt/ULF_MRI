#!/bin/bash
# Validation experiments (phase A: A1, A2, A4, A6), public data only.
# Jobs run one at a time on the shared RTX 4090: each starts when >= 9 GB has stayed free for a
# minute and is retried after a CUDA out-of-memory. A job whose done-file exists is skipped.
#   stage_sanity  d3stage pooled sample vs out_public/d3dfactor_T2.json   -> d3stage_sanity_T2.json
#   stage_run     A1: stage-wise retention, 12 enhancers / 36 models, 2 operators, 6 cohorts
#   stage_sum     A1 summary JSON + fig_public/stage_retention.pdf (CPU)
#   spec          A2: false-lesion rate, test-retest, spill-over            -> d3spec_T2.json
#   agree         A4: subject-level CCC and Bland-Altman                    -> stats_agree.json (CPU)
#   h2lo_*        A6: H2LO as a third-party operator                        -> d3h2lo_T2.json
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
cd $ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
LOG=$ROOT/out_public/logs
st() { echo "$(date +%F_%T) $* (review)" >> $LOG/status.log; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { while true; do
    [ "$(gpu_free)" -ge 9000 ] && { sleep 60; [ "$(gpu_free)" -ge 9000 ] && return; }
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
cpu() {   # cpu <name> <log> <done-file> <script and args...>   (no GPU wait)
  local nm=$1 log=$2 done=$3; shift 3
  [ -e "$done" ] && { st "skip  $nm"; return 0; }
  st "start $nm"; $PY "$@" > $log 2>&1 && { st "done  $nm"; return 0; }
  st "FAIL  $nm (see $log)"; return 1
}

stage() {   # A1
  job stage_sanity $LOG/stage_sanity.log out_public/d3stage_rows_pooled_pub3t_hip_T2.npz \
      exp/d3stage.py run --mode pooled --splits test ext_v5 pub3t_unc pub3t_penn pub3t_hip || return 1
  $PY exp/d3stage.py sanity > $LOG/stage_sanity_check.log 2>&1 || return 1
  grep -q PASS $LOG/stage_sanity_check.log || { st "FAIL stage sanity check did not pass"; return 1; }
  job stage_run $LOG/stage_run.log out_public/d3stage_rows_pub3t_hip_T2.npz exp/d3stage.py run --mode dense || return 1
  cpu stage_sum $LOG/stage_sum.log out_public/d3stage_T2.json exp/d3stage.py summary || return 1
}
spec() { job spec $LOG/spec.log out_public/d3spec_T2.json exp/d3spec.py; }                 # A2
agree() { cpu agree $LOG/agree.log out_public/stats_agree_T2.json exp/stats_agree.py; }     # A4
h2lo() { job h2lo $LOG/h2lo.log out_public/d3h2lo_T2.json exp/d3h2lo.py run; }               # A6

# usage: bash exp/run_review.sh [stage spec agree h2lo]   (default: all, in this order)
for step in "${@:-stage spec agree h2lo}"; do
  for s in $step; do $s || exit 1; done
done
exit 0
