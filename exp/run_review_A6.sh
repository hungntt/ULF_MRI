#!/bin/bash
# Validation step A6: H2LO as a third-party degradation operator.
# One GPU job (exp/d3h2lo.py run), started when >= 5000 MB stay free for a minute, retried on CUDA OOM.
#   usage: bash exp/run_review_A6.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
cd $ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
LOG=$ROOT/out_public/logs
st() { echo "$(date +%F_%T) $* (review_A6)" >> $LOG/status.log; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { while true; do
    [ "$(gpu_free)" -ge 5000 ] && { sleep 60; [ "$(gpu_free)" -ge 5000 ] && return; }
    sleep 120; done; }
[ -e out_public/d3h2lo_T2.json ] && { st "skip  h2lo"; exit 0; }
for t in 1 2 3 4 5 6 7 8; do
  wait_gpu; st "start h2lo"
  $PY exp/d3h2lo.py run > $LOG/h2lo.log 2>&1 && { st "done  h2lo"; exit 0; }
  grep -q -E "OutOfMemoryError|CUDA out of memory" $LOG/h2lo.log || break
  st "oom   h2lo, retry $t"
done
st "FAIL  h2lo (see $LOG/h2lo.log)"; exit 1
