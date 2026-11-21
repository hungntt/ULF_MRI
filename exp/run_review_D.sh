#!/bin/bash
# Validation phase D: public pretrained models in the
# retention benchmark, public data only. GPU jobs run one at a time on the shared RTX 4090: each
# starts when the job's memory need has stayed free for a minute (6 GB by default; SuperSynth
# peaks at 8.4 GB, so 10 GB) and is retried after a CUDA out-of-memory. FreeSurfer tools run on
# the CPU with 4 threads each and at most 3 workers. A step whose done-file exists is skipped.
#   make      simulated + acquired 64 mT volumes            datasets/scratch_public/pretrained/in
#   ssynth    SuperSynth (float wrapper, GPU)                 .../pretrained/out/*/ss_*
#   sscheck   official mri_super_synth CLI on one volume      out_public/d3pretrained_sscheck.json
#   synthsr   SynthSR rerun (sanity) + acquired inputs (CPU)  .../pretrained/{sr_rerun,out}
#   zoo       36 zoo models + routes A-C on the same volumes  out_public/d3pretrained_{zoo,route}_rows.json
#   gambas    GAMBAS (own venv, GPU)                          .../pretrained/gambas
#   synthseg  SynthSeg on outputs from the acquired inputs    .../pretrained/synthseg
#   score     summary + figure                                out_public/d3pretrained.json, fig_public/pretrained*.pdf
# usage: bash exp/run_review_D.sh [steps...]   (default: all, in this order); ULF_SPLIT=ext_v5 for Zenodo
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
cd $ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
export ULF_SPLIT=${ULF_SPLIT:-test}
SFX=""; [ "$ULF_SPLIT" != "test" ] && SFX="_$ULF_SPLIT"
W=$ROOT/datasets/scratch_public/pretrained$SFX
LOG=$ROOT/out_public/logs
mkdir -p $LOG $W
st() { echo "$(date +%F_%T) $* (review_D$SFX)" >> $LOG/status.log; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { local need=${1:-6000}; while true; do
    [ "$(gpu_free)" -ge $need ] && { sleep 60; [ "$(gpu_free)" -ge $need ] && return; }
    sleep 120; done; }
job() {   # job <MB needed> <name> <log> <done-file> <command...>
  local need=$1 nm=$2 log=$3 done=$4; shift 4
  [ -e "$done" ] && { st "skip  $nm"; return 0; }
  for t in 1 2 3 4 5 6 7 8; do
    wait_gpu $need; st "start $nm"
    "$@" > $log 2>&1 && { st "done  $nm"; return 0; }
    grep -q -E "OutOfMemoryError|CUDA out of memory" $log || break
    st "oom   $nm, retry $t"
  done
  st "FAIL  $nm (see $log)"; return 1
}
cpu() {   # cpu <name> <log> <done-file> <command...>   (no GPU wait)
  local nm=$1 log=$2 done=$3; shift 3
  [ -e "$done" ] && { st "skip  $nm"; return 0; }
  st "start $nm"; "$@" > $log 2>&1 && { st "done  $nm"; return 0; }
  st "FAIL  $nm (see $log)"; return 1
}
D="exp/d3pretrained.py"
GPY="$ROOT/datasets/raw/gambas/.venv/bin/python -I -B -u"

make()     { job 6000 pre_make$SFX $LOG/pre_make$SFX.log $W/meta.json $PY $D make; }
ssynth()   { job 10000 pre_ssynth$SFX $LOG/pre_ssynth$SFX.log $W/ss_done $PY $D ssynth && touch $W/ss_done; }
sscheck()  { cpu pre_sscheck$SFX $LOG/pre_sscheck$SFX.log out_public/d3pretrained_sscheck$SFX.json $PY $D ss_check; }
synthsr()  { cpu pre_synthsr$SFX $LOG/pre_synthsr$SFX.log $W/sr_done $PY $D synthsr && touch $W/sr_done; }
ZOO_DONE=out_public/d3pretrained_zoo_rows$SFX.json; [ "$ULF_SPLIT" = test ] && ZOO_DONE=out_public/d3pretrained_route_rows.json
zoo()      { job 6000 pre_zoo$SFX $LOG/pre_zoo$SFX.log $ZOO_DONE $PY $D zoo; }
gambas()   { cpu pre_gprep$SFX $LOG/pre_gprep$SFX.log $W/gambas/prep_done $PY $D gambas_prep || return 1
             job 12000 pre_gambas$SFX $LOG/pre_gambas$SFX.log $W/gambas/run_done $GPY $D gambas_run || return 1
             cpu pre_gback$SFX $LOG/pre_gback$SFX.log $W/gambas/back_done $PY $D gambas_back; }
synthseg() { cpu pre_synthseg$SFX $LOG/pre_synthseg$SFX.log $W/synthseg/seg_done $PY $D synthseg; }
score()    { job 6000 pre_score$SFX $LOG/pre_score$SFX.log /nonexistent $PY $D score || return 1
             cpu pre_fig$SFX $LOG/pre_fig$SFX.log /nonexistent $PY $D fig; }

for step in "${@:-make ssynth synthsr zoo sscheck gambas synthseg score}"; do
  for s in $step; do $s || exit 1; done
done
exit 0
