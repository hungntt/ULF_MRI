#!/bin/bash
# Off-lesion false-structure rate (exp/d3offlesion.py) over the six cohorts. The 4090 is shared:
# before each attempt wait until at least NEED_MB is free, and retry on CUDA OOM without a cap
# (the caller's timeout bounds the run). Each cohort is skipped once its rows exist.
cd $ROOT
export ULF_DATA=datasets/npy_public ULF_OUT=out_public ULF_FIG=fig_public
export ULF_SCRATCH=${ULF_SCRATCH:-/tmp/ulf_scratch}
PY="${PYTHON:-python} -I -u"
NEED_MB=${NEED_MB:-6000}
LOG=out_public/logs/offlesion.log
mkdir -p out_public/logs
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for split in test ext_on_hfe ext_v5 pub3t_unc pub3t_penn pub3t_hip; do
  attempt=0
  until [ -f out_public/d3offlesion_rows_${split}_T2.npz ]; do
    while [ "$(free_mb)" -lt "$NEED_MB" ]; do sleep 60; done
    attempt=$((attempt + 1))
    $PY exp/d3offlesion.py run --splits $split >> $LOG 2>&1 \
      || { echo "[$split] attempt $attempt failed, waiting for GPU memory" >> $LOG; sleep 60; }
  done
done
$PY exp/d3offlesion.py summary >> $LOG 2>&1
echo DONE >> $LOG
