#!/bin/bash
# ATLAS R3.0 transplant (the ATLAS analysis plan S3-S6): register -> place -> prereg -> run (GPU) ->
# released (GPU + CPU) -> summary. Finished stages are skipped by their own caching.
#   usage: bash exp/run_atlas.sh [stages...]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd $ROOT
PY="${PYTHON:-python} -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
LOG=$ROOT/out_public/logs
st() { echo "$(date +%F_%T) $* (atlas)" >> $LOG/status.log; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { while true; do
    [ "$(gpu_free)" -ge 8000 ] && { sleep 30; [ "$(gpu_free)" -ge 8000 ] && return; }
    sleep 120; done; }
stage() {   # stage <name> <gpu 0/1>
  local nm=$1 g=$2
  [ "$g" = 1 ] && wait_gpu
  st "start atlas_$nm"
  $PY exp/d3atlas.py $nm > $LOG/atlas_$nm.log 2>&1 || { st "FAIL  atlas_$nm (see $LOG/atlas_$nm.log)"; exit 1; }
  st "done  atlas_$nm"
}
for s in ${@:-register place prereg run released summary}; do
  case $s in run|released) stage $s 1;; *) stage $s 0;; esac
done
exit 0
