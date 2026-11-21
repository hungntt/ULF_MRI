#!/bin/bash
# Validation phase C (MS only; ATLAS deferred):
# real MS lesions (Muslim et al. 2022, 1.5 T) transplanted into healthy public 3T targets.
# CPU stages use at most 3 workers x 4 threads (shared host); GPU jobs start when >= 6 GB has
# stayed free for a minute and are retried after a CUDA out-of-memory. A job whose done-file
# exists is skipped.
#   prep      exp/prep_lesion.py run/targets/qc/pack/synthseg            (CPU, FreeSurfer)
#   lesions   d3reallesion.py lesions: inpaint, fractional contrast m     (CPU)
#   register  d3reallesion.py register: SynthMorph joint patient->target  (CPU)
#   place     d3reallesion.py place: WM / brain placement checks          (CPU)
#   sanity    d3reallesion.py sanity: Gaussian through the transplant path (GPU)
#   transplant d3reallesion.py run: 12 seed-0 enhancers, routes A-C, map  (GPU)
#   inplace   d3reallesion.py inplace: patients' own images, pseudo-lesion inpainting bias (GPU)
#   seeds     d3reallesion.py run --seeds: seeds 1-4 of the 5 seeded architectures (GPU)
#   summary   d3reallesion.py summary -> out_public/d3reallesion.json, fig_public/reallesion.pdf
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
cd $ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer}
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
RLD=$ROOT/datasets/scratch_public/reallesion
LOG=$ROOT/out_public/logs
mkdir -p $LOG $RLD
st() { echo "$(date +%F_%T) $* (review_C)" >> $LOG/status.log; }
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

prep() {
  cpu C_prep_run $LOG/C_prep_run.log $RLD/prep_run.done exp/prep_lesion.py run --workers 3 || exit 1
  touch $RLD/prep_run.done
  cpu C_prep_targets $LOG/C_prep_targets.log $RLD/targets.json exp/prep_lesion.py targets --workers 3 || exit 1
  cpu C_prep_qc $LOG/C_prep_qc.log $RLD/exclusions.json exp/prep_lesion.py qc || exit 1
  cpu C_prep_pack $LOG/C_prep_pack.log $ROOT/datasets/npy_public_ms/manifest_ms_T2.json exp/prep_lesion.py pack || exit 1
  cpu C_synthseg $LOG/C_synthseg.log $RLD/synthseg.done exp/prep_lesion.py synthseg || exit 1
  touch $RLD/synthseg.done
}
lesions() { cpu C_lesions $LOG/C_lesions.log $RLD/lesions.done exp/d3reallesion.py lesions || exit 1; touch $RLD/lesions.done; }
register() { cpu C_register $LOG/C_register.log $RLD/register.done exp/d3reallesion.py register --workers 3 || exit 1; touch $RLD/register.done; }
place() { cpu C_place $LOG/C_place.log $RLD/place.done exp/d3reallesion.py place || exit 1; touch $RLD/place.done; }
sanity() { job C_sanity $LOG/C_sanity.log $ROOT/out_public/d3reallesion_sanity.json exp/d3reallesion.py sanity || exit 1; }
transplant() { job C_run $LOG/C_run.log $ROOT/out_public/d3reallesion_rows_s0.json exp/d3reallesion.py run || exit 1; }
inplace() { job C_inplace $LOG/C_inplace.log $ROOT/out_public/d3reallesion_inplace.json exp/d3reallesion.py inplace || exit 1; }
seeds() { job C_seeds $LOG/C_seeds.log $ROOT/out_public/d3reallesion_rows_seeds.json exp/d3reallesion.py run --seeds || exit 1; }
summary() { cpu C_summary $LOG/C_summary.log $ROOT/out_public/d3reallesion.json.never exp/d3reallesion.py summary || exit 1; }

# usage: bash exp/run_review_C.sh [prep lesions register place sanity transplant inplace seeds summary]
for step in "${@:-prep lesions register place sanity transplant inplace summary}"; do
  for s in $step; do $s; done
done
exit 0
