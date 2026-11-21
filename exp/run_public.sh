#!/bin/bash
# Full retrain on public data only (no unpaired 64 mT pool, no registration-gated
# data). Arrays come from exp/prep_public.py; every output goes to out_public/ and
# datasets/scratch_public/, so out/ and datasets/npy are never written.
#
#   bash exp/run_public.sh 1    # models that need nothing else               (2 GPU lanes)
#   bash exp/run_public.sh 2    # models that need D (and the IXI corpus)     (2 GPU lanes)
#   bash exp/run_public.sh 4a   # evaluations of the seed-0 models            (2 GPU lanes + CPU)
#   bash exp/run_public.sh 3    # seeds 1-4: five baseline architectures and the corpus arm
#   bash exp/run_public.sh 4b   # add the seeds to the zoo, SynthSeg and statistics
#   bash exp/run_public.sh all  # 1, 2, 4a, 3, 4b, stopping at the first stage with a failed lane
#
# The evaluations run before the seeds so that every result but the seed-averaged zoo
# statistics is available early; d3zoo and SynthSeg skip finished work, so 4b only adds the
# seeds. Commands, steps and tags follow the drivers that produced the current results
# (run_all.sh, run_baselines.sh, run_final.sh, run_d3v2.sh, run_qa_all.sh, run_seeds12k*.sh,
# run_night1/3/4.sh, run_lw.sh, run_detect*.sh, run_external*.sh, run_zoo_rolling.sh).
# A finished checkpoint is never retrained, so a stage can be
# re-run after an interruption.
#
# Note: errexit (set -e) is ignored inside functions called from `a || b`, which is how `all`
# calls the stages, so a failed training run ends its lane with an explicit exit instead.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python} -I -u"
export ULF_DATA=$ROOT/datasets/npy_public ULF_OUT=$ROOT/out_public
export ULF_SCRATCH=$ROOT/datasets/scratch_public ULF_FIG=$ROOT/fig_public
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4  # shared host
LOG=$ULF_OUT/logs
mkdir -p $ULF_OUT $ULF_SCRATCH $ULF_FIG $LOG
cd $ROOT
st() { echo "$(date +%F_%T) $*" >> $LOG/status.log; }

# run_retry <log> <script and args...>: run; if it died of CUDA out-of-memory (the GPU is
# shared with other users' jobs), wait 10 min and try again, up to 8 times.
run_retry() {
  local log=$1; shift
  for t in 1 2 3 4 5 6 7 8; do
    $PY "$@" > $log 2>&1 && return 0
    grep -q -E "OutOfMemoryError|CUDA out of memory" $log || return 1
    st "oom   $(basename $log .log), retry $t in 10 min"; sleep 600
  done
  return 1
}
# train <checkpoint name> <script and args...>: skip if out_public/<name>.pt exists; otherwise
# run, and end the calling lane if the checkpoint did not appear.
train() {
  local ck=$1; shift
  if [ -f "$ULF_OUT/$ck.pt" ]; then st "skip  $ck"; return 0; fi
  st "start $ck"
  run_retry $LOG/$ck.log "$@"
  if [ -f "$ULF_OUT/$ck.pt" ]; then st "done  $ck"
  else st "FAIL  $ck (see logs/$ck.log)"; exit 1; fi
}
# ev <log name> <script and args...>: an evaluation; failures are logged, not fatal.
ev() {
  local nm=$1; shift
  st "start $nm"
  if run_retry $LOG/$nm.log "$@"; then st "done  $nm"; else st "FAIL  $nm (see logs/$nm.log)"; fi
}
need() { for f in "$@"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done; }
lanes() { local rc=0; for p in "$@"; do wait $p || rc=1; done; return $rc; }   # status of all lanes

stage1() {
  need $ULF_DATA/paired_train_hf_T2.npy $ULF_DATA/dev_val_ids.json
  (  # lane A: supervised enhancers and the cross-contrast routes
    [ -f $ULF_OUT/base_identity_T2.json ] || run_retry $LOG/base_identity_T2.log exp/baselines.py --contrast T2 --method identity
    for m in unet unet_ssim pix2pix lohiresgan unet_noresid; do
      train base_${m}_T2 exp/baselines.py --contrast T2 --method $m --steps 12000; done
    train base_unet_noresid_T1 exp/baselines.py --contrast T1 --method unet_noresid --steps 12000
    train d2_T2toT1  exp/d2.py --use T2 --out T1 --steps 10000 --tag T2toT1
    train d2_bothtoT1 exp/d2.py --use T1,T2 --out T1 --steps 10000 --tag bothtoT1
  ) & A=$!
  (  # lane B: degradation operator D, regression heads, diffusion, QA map + ensemble
    train d1_forward_T2 exp/d1.py forward --contrast T2 --steps 6000
    train d1_forward_T1 exp/d1.py forward --contrast T1 --steps 6000
    # d3.py's calibration-size sweep can fail after the checkpoint is saved (3 dev-val
    # subjects); only the checkpoint is used, so train() checks for the .pt alone.
    train d3_cqr_T2   exp/d3.py --contrast T2 --mode cqr   --steps 10000 --alpha 0.1
    train d3_gauss_T2 exp/d3.py --contrast T2 --mode gauss --steps 10000 --alpha 0.1
    train d3v2_diff_T2 exp/d3v2.py --contrast T2 --steps 14000 --gamma 1.0
    train d3qa_fold3_T2 exp/d3qa.py crossfit --contrast T2 --steps 9000
    train d3qa_T2 exp/d3qa.py qa --contrast T2 --steps 9000
  ) & B=$!
  lanes $A $B
}

stage2() {
  need $ULF_OUT/d1_forward_T2.pt $ULF_OUT/d1_forward_T1.pt $ULF_OUT/base_unet_noresid_T2.pt \
       $ULF_OUT/d3qa_T2.pt $ULF_DATA/ixi_hf_T2.npy $ULF_DATA/manifest_ixi_T2.json
  (  # lane A: corpus arm (IXI Hammersmith, D-rendered pseudo-pairs); its seeds are in stage 3
    for mode in handcrafted learned learned_cal; do
      [ -f $ULF_DATA/d1_pseudo_ulf_T2_$mode.npy ] || run_retry $LOG/d1_pairs_$mode.log exp/d1.py pairs --contrast T2 --mode $mode \
        || { st "FAIL  d1_pairs_$mode"; exit 1; }
    done
    for mode in none handcrafted learned learned_cal; do
      train d1_${mode}_T2 exp/d1.py train --contrast T2 --mode $mode --pre_steps 8000 --ft_steps 6000; done
  ) & A=$!
  (  # lane B: retention maps and the extra route-C models
    train d3detect_T2 exp/d3detect.py --contrast T2 --steps 8000 --cs bright
    for s in 1 2 3; do
      train d3detect_T2_s$s exp/d3detect.py --contrast T2 --steps 8000 --cs bright --seed $s \
            --tag _s$s --labtag "" --resume; done
    train d3detect_T2_signed exp/d3detect.py --contrast T2 --steps 8000 --cs signed --tag _signed
    train d3detect_T2_realin exp/d3detreal.py --input real --tag _realin --steps 8000
    train d2_bothtoT1_ctl  exp/d3aug.py --aug 0.0 --tag _ctl
    train d2_bothtoT1_aug  exp/d3aug.py --aug 0.5 --tag _aug
    train d2_bothtoT1_augw exp/d3aug.py --aug 0.5 --lw 1.0 --tag _augw
  ) & B=$!
  lanes $A $B
}

seg() {   # SynthSeg on one split's exported volumes, then the zoo statistics (CPU)
  export ULF_SPLIT=$1
  ev synthseg_prepare_$1 exp/d3synthseg.py prepare
  ev synthseg_run_$1 -c "import sys; sys.path.insert(0,'$ROOT'); import exp.d3synthseg as s; s.run(threads=12, workers=2)"
  ev synthseg_score_$1 exp/d3synthseg.py score
  ev zoo_stats_$1 exp/d3zoo_stats.py
  unset ULF_SPLIT
}
zoo_all() {   # enhancer zoo per split; ext_on_hfe shares the test split's 3T arrays
  ev zoo_test$1   exp/d3zoo.py --chunk 4
  ev zoo_ext_v5$1 exp/d3zoo.py --chunk 4 --split ext_v5
  ev zoo_ext_on_hfe$1 exp/d3zoo.py --chunk 8 --split ext_on_hfe --reuse_from test
  for s in pub3t_unc pub3t_penn pub3t_hip; do ev zoo_$s$1 exp/d3zoo.py --chunk 4 --split $s; done
}

stage4a() {
  need $ULF_OUT/d3detect_T2_s3.pt $ULF_OUT/d2_bothtoT1_augw.pt $ULF_OUT/d1_learned_cal_T2.pt \
       $ULF_OUT/base_unet_noresid_T1.pt $ULF_OUT/d2_bothtoT1.pt
  (  # lane E1: enhancer zoo (seed-0 models), then SynthSeg and preliminary statistics on CPU
    zoo_all ""
    seg test; seg ext_v5; seg ext_on_hfe
  ) & A=$!
  (  # lane E2: polarity, map and closed loop per cohort, then the map's own analyses
    for s in test ext_v5 pub3t_unc pub3t_penn pub3t_hip; do ev lesiontype_$s exp/d3lesiontype.py $s; done
    ev external exp/d3external.py test ext_on_hfe ext_v5 pub3t_unc pub3t_penn pub3t_hip
    ev external_summary exp/d3external_summary.py
    ev detect_eval exp/d3detect_eval.py --contrast T2
    for t in _s1 _s2 _s3 _signed; do ev detect_eval$t exp/d3detect_eval.py --contrast T2 --tag $t; done
    ev finding  exp/d3finding.py --contrast T2 --cal pub3t   # 27 calibration / 26 test subjects
    ev brainfrac exp/d3brainfrac.py T2
    ev ood      exp/d3ood.py --contrast T2
    ev xarch    exp/d3xarch.py
    ev mech     exp/d3mech.py T2
    ev pipe      exp/d3pipe.py
    ev pipe_same exp/d3pipe.py --c1_frac 1.0 --tag _same
    ev polinfo   exp/d3polinfo.py
    ev synthsr_make exp/d3synthsr.py make
    ev synthsr_run_1mm exp/d3synthsr.py run 1mm
    ev synthsr_score_1mm exp/d3synthsr.py score 1mm
    ev synthsr_run exp/d3synthsr.py run
    ev synthsr_score exp/d3synthsr.py score
  ) & B=$!
  wait $A $B
}

stage3() {
  need $ULF_DATA/d1_pseudo_ulf_T2_learned_cal.npy
  (  # lane A: 12 runs
    for s in 1 2 3 4; do
      train d1_learned_cal_seed${s}_T2 exp/d1.py train --contrast T2 --mode learned_cal \
            --pre_steps 8000 --ft_steps 6000 --seed $s --tag learned_cal_seed$s; done
    for m in unet unet_ssim; do for s in 1 2 3 4; do
      train base_${m}_seed${s}_T2 exp/baselines.py --contrast T2 --method $m --steps 12000 --seed $s --tag ${m}_seed$s
    done; done
  ) & A=$!
  (  # lane B: 12 runs
    for m in pix2pix unet_noresid lohiresgan; do for s in 1 2 3 4; do
      train base_${m}_seed${s}_T2 exp/baselines.py --contrast T2 --method $m --steps 12000 --seed $s --tag ${m}_seed$s
    done; done
  ) & B=$!
  lanes $A $B
}

stage4b() {
  need $ULF_OUT/base_lohiresgan_seed4_T2.pt $ULF_OUT/base_unet_ssim_seed4_T2.pt
  zoo_all "_seeds"
  seg test; seg ext_v5; seg ext_on_hfe
}

case "${1:-}" in
  1|2|3|4a|4b) stage${1} ;;
  all) for k in 1 2 4a 3 4b; do
         stage$k || { st "STOPPED after stage $k (a lane failed)"; exit 1; }
         st "stage $k finished"; done; exit 0 ;;
  *) echo "usage: bash exp/run_public.sh 1|2|4a|3|4b|all"; exit 2 ;;
esac || { st "stage ${1} FAILED"; exit 1; }
st "stage ${1} finished"
