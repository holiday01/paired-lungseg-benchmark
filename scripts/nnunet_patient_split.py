#!/usr/bin/env python3
"""Write a PATIENT-GROUPED 5-fold splits_final.json for nnU-Net Dataset001_LungTumor.

Why: convert_to_nnunet.py names cases LT_000, LT_001, ... in the sorted order of the label
files /data3/ct/lph_ct/img_cancer_label/*.nii.gz, whose stems are CTLC_<patient>_B<nn>. With
36/102 patients carrying >1 tumour, nnU-Net's default RANDOM 5-fold can place two tumours from
the same patient in different folds, leaking patient identity and inflating the Dice ceiling.
This reconstructs the LT_nnn -> patient map from the same sorted glob and writes a
GroupKFold-by-patient split so the ceiling is honest.

Run AFTER nnUNetv2_plan_and_preprocess (which creates the preprocessed dataset dir).
"""
import glob
import json
import os
import re
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold

LAB = "/data3/ct/lph_ct/img_cancer_label"
IMG = "/data3/ct/lph_ct/img_cancer"
PREP = Path("/home/holiday/lung_ct/data/nnUNet_preprocessed/Dataset001_LungTumor")

# reproduce the exact case naming from convert_to_nnunet.py
labs = sorted(glob.glob(f"{LAB}/*.nii.gz"))
case_ids, patients = [], []
n = 0
for lp in labs:
    stem = os.path.basename(lp).replace(".nii.gz", "")
    if not os.path.exists(f"{IMG}/{stem}.nii.gz"):
        continue
    cid = f"LT_{n:03d}"
    pat = re.sub(r"_B\d+$", "", stem)   # strip lesion suffix -> patient id
    case_ids.append(cid)
    patients.append(pat)
    n += 1

case_ids = np.array(case_ids)
patients = np.array(patients)
n_pat = len(set(patients))
print(f"{len(case_ids)} cases, {n_pat} patients "
      f"({len(case_ids) - n_pat} multi-tumour duplicates)")

gkf = GroupKFold(n_splits=5)
splits = []
for tr, va in gkf.split(case_ids, groups=patients):
    # assert no patient leakage across this fold
    assert not (set(patients[tr]) & set(patients[va])), "patient leak in fold!"
    splits.append({"train": sorted(case_ids[tr].tolist()),
                   "val": sorted(case_ids[va].tolist())})
    print(f"  fold: train {len(tr)} cases / val {len(va)} cases "
          f"({len(set(patients[va]))} val patients)")

if PREP.exists():
    out = PREP / "splits_final.json"
    out.write_text(json.dumps(splits, indent=2))
    print(f"wrote patient-grouped {out}")
else:
    print(f"NOTE: {PREP} does not exist yet -- run nnUNetv2_plan_and_preprocess first, "
          f"then re-run this script.")
