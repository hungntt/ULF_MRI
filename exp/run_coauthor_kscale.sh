#!/bin/bash
# Validation check: k-scaled erasure, all 36 models, both arms.
# One GPU job, started when >= 6000 MB stay free for a minute, retried on CUDA OOM.
#   usage: bash exp/run_coauthor_kscale.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
cd $ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
LOG=$ROOT/out_public/logs
st() { echo "$(date +%F_%T) $* (coauthor_kscale)" >> $LOG/status.log; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { while true; do
    [ "$(gpu_free)" -ge 6000 ] && { sleep 60; [ "$(gpu_free)" -ge 6000 ] && return; }
    sleep 120; done; }
for t in 1 2 3 4 5 6 7 8; do
  wait_gpu; st "start kscale"
  $PY exp/coauthor_kscale.py run >> $LOG/coauthor_kscale.log 2>&1 && { st "done  kscale"; exit 0; }
  grep -q -E "OutOfMemoryError|CUDA out of memory" $LOG/coauthor_kscale.log || break
  st "oom   kscale, retry $t"
done
st "FAIL  kscale (see $LOG/coauthor_kscale.log)"; exit 1
