#!/bin/bash
# Extra public-data experiments that run beside exp/run_public.sh. Each job starts only when
# >= 9 GB of GPU memory has stayed free for a minute (the main driver keeps two jobs on the
# GPU), and is retried if it still dies of CUDA out-of-memory.
#   1. d3realret   retention of screened real focal T2 features (exp/d3real.py, REAL_TAG=_pub_loose)
#   2. d1 forward  a second degradation operator D, trained on OpenNeuro's second Swoop unit (HFE)
#                  for the same 15 dev-train subjects (datasets/npy_public_hfe)
#   3. d3dfactor   retention = scanner term x enhancer term, under the two operators
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
cd $ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
LOG=$ROOT/out_public/logs
st() { echo "$(date +%F_%T) $* (extra)" >> $LOG/status.log; }
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

pub() { export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public; }
pub
REAL_Z=3.5 REAL_ZA=2.5 REAL_RUN=2 REAL_TAG=_pub_loose \
  job realret $LOG/realret.log out_public/d3realret_pub_loose.json exp/d3realret.py
( export ULF_DATA=$ROOT/datasets/npy_public_hfe ULF_OUT=$ROOT/out_public_hfe ULF_SCRATCH=$ROOT/datasets/scratch_public_hfe
  mkdir -p $ULF_OUT
  job d1_forward_T2_hfe $ROOT/out_public_hfe/d1_forward_T2.log out_public_hfe/d1_forward_T2.pt \
      exp/d1.py forward --contrast T2 --steps 6000 )
pub
until [ -f out_public/d1_learned_cal_T2.pt ]; do sleep 300; done     # all seed-0 corpus models
job dfactor $LOG/dfactor.log out_public/d3dfactor_T2.json exp/d3dfactor.py
