#!/bin/bash
# Validation check: inpainting-error propagation, resumes per draw.
# One GPU job, started when >= 6000 MB stay free for a minute, retried on CUDA OOM.
#   usage: bash exp/run_coauthor_inpaint.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
cd $ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
LOG=$ROOT/out_public/logs
st() { echo "$(date +%F_%T) $* (coauthor_inpaint)" >> $LOG/status.log; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { while true; do
    [ "$(gpu_free)" -ge 6000 ] && { sleep 60; [ "$(gpu_free)" -ge 6000 ] && return; }
    sleep 120; done; }
for t in 1 2 3 4 5 6 7 8; do
  wait_gpu; st "start inpaint"
  $PY exp/coauthor_inpaint.py run >> $LOG/coauthor_inpaint.log 2>&1 && { st "done  inpaint"; exit 0; }
  grep -q -E "OutOfMemoryError|CUDA out of memory" $LOG/coauthor_inpaint.log || break
  st "oom   inpaint, retry $t"
done
st "FAIL  inpaint (see $LOG/coauthor_inpaint.log)"; exit 1
