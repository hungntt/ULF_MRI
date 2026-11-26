"""Prep for the handcrafted map baseline: SynthSeg on the 3T T1 of the 15 OpenNeuro training subjects.

The retention map's 5754 training labels lie in these subjects; the handcrafted baseline needs a tissue class at
each label. Same input and orientation as exp/d1_physcheck.py prep/seg (which covers dev-val, test and Zenodo),
written to a separate folder so the physical check's own inputs are untouched.

    ULF_DATA=datasets/npy_public ULF_SCRATCH=datasets/scratch_public python exp/coauthor_seg_train.py
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "."
sys.path.insert(0, ROOT)
DATA = os.environ.get("ULF_DATA", f"{ROOT}/datasets/npy_public")
SCR = os.environ.get("ULF_SCRATCH", f"{ROOT}/datasets/scratch_public")
WORK = f"{SCR}/coauthor_seg"
FS = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")


def train_subjects():
    return [s["subject_id"] for s in json.load(open(f"{DATA}/manifest_train_T2.json"))["subjects"]]


def prep():
    import nibabel as nib
    from exp.d1_physcheck import stack
    os.makedirs(f"{WORK}/in", exist_ok=True)
    for sid in train_subjects():
        f = f"{WORK}/in/{sid}.nii.gz"
        if not os.path.exists(f):
            y1 = stack(sid, "train", "T1", "hf", DATA)
            nib.save(nib.Nifti1Image(np.transpose(y1, (1, 2, 0)), np.eye(4)), f)


def seg(workers=3, threads=4):
    env = dict(os.environ, FREESURFER_HOME=FS, FS_LICENSE=f"{FS}/license.txt", OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(threads))
    os.makedirs(f"{WORK}/seg", exist_ok=True)
    todo = [s for s in train_subjects() if not os.path.exists(f"{WORK}/seg/{s}.nii.gz")]

    def go(sid):
        cmd = [f"{FS}/bin/mri_synthseg", "--i", f"{WORK}/in/{sid}.nii.gz", "--o", f"{WORK}/seg/{sid}.nii.gz",
               "--robust", "--cpu", "--threads", str(threads)]
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"  {sid}: exit {r.returncode}", flush=True)
        return r.returncode
    with ThreadPoolExecutor(workers) as ex:
        rc = list(ex.map(go, todo))
    assert not any(rc), "SynthSeg failed on some subjects"


if __name__ == "__main__":
    prep(); seg()
