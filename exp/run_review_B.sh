#!/bin/bash
# Validation phase B: physical check of the learned
# degradation operator D against acquired 64 mT scans, public data only (exp/d1_physcheck.py).
# GPU stages start when >= 6 GB has stayed free for a minute and are retried after a CUDA
# out-of-memory; CPU stages cap threads at 4 (SynthSeg and structures use 3 workers).
# A stage whose done-file exists is skipped.
#   prep     3T T1 stacks as NIfTI                     -> datasets/scratch_public/physcheck/in
#   seg      SynthSeg on CPU                            -> physcheck/seg
#   struct   structure masks and local thickness        -> physcheck/struct, qc_struct.png
#   measure  D(y) and instance contrasts (GPU)          -> physcheck/rows.npz
#   erase    k-grid retention, seed-0 enhancers (GPU)   -> physcheck/erase_{test,ext_v5}.npz
#   summary  statistics and figures (CPU)               -> out_public/d1_physcheck*.json, fig_public/physcheck*.pdf
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
cd $ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer}
W=$ULF_SCRATCH/physcheck
LOG=$ROOT/out_public/logs
mkdir -p $W $LOG $ULF_FIG
st() { echo "$(date +%F_%T) $* (review_B)" >> $LOG/status.log; }
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
cpu() {   # cpu <name> <log> <done-file> <script and args...>   (no GPU wait)
  local nm=$1 log=$2 done=$3; shift 3
  [ -e "$done" ] && { st "skip  $nm"; return 0; }
  st "start $nm"; $PY "$@" > $log 2>&1 && { st "done  $nm"; return 0; }
  st "FAIL  $nm (see $log)"; return 1
}

prep()    { cpu physcheck_prep $LOG/physcheck_prep.log $W/in/V5-0066.nii.gz exp/d1_physcheck.py prep; }
seg()     { cpu physcheck_seg $LOG/physcheck_seg.log $W/seg/V5-0066.nii.gz exp/d1_physcheck.py seg; }
struct()  { cpu physcheck_struct $LOG/physcheck_struct.log $W/qc_struct.png exp/d1_physcheck.py struct; }
measure() { job physcheck_measure $LOG/physcheck_measure.log $W/rows.npz exp/d1_physcheck.py measure; }
erase()   { job physcheck_erase $LOG/physcheck_erase.log $W/erase_ext_v5.npz exp/d1_physcheck.py erase; }
summary() { cpu physcheck_summary $LOG/physcheck_summary.log $ROOT/out_public/d1_physcheck.json exp/d1_physcheck.py summary; }

# usage: bash exp/run_review_B.sh [prep seg struct measure erase summary]   (default: all, in order)
for step in "${@:-prep seg struct measure erase summary}"; do
  for s in $step; do $s || exit 1; done
done
exit 0
