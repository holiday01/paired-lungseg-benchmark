#!/usr/bin/env python3
"""Convert the 102 cancer cases (original-res CT + tumor mask) to nnU-Net v2 raw format.

Gives nnU-Net the ORIGINAL volumes so its auto-config picks the optimal spacing/patch
(its main strength) — a proper gold-standard baseline to compare our 3D U-Net against.
Each label file (CTLC_xxx_Bnn) -> one training case. Binary tumor label {0,1}.

Output: $nnUNet_raw/Dataset001_LungTumor/{imagesTr,labelsTr}/ + dataset.json
Then (separately): nnUNetv2_plan_and_preprocess -d 1 ; nnUNetv2_train 1 3d_fullres 0
"""
import os, glob, re, json, shutil, sys, numpy as np, nibabel as nib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "nnUNet_raw" / "Dataset001_LungTumor"
LAB = "/data3/ct/lph_ct/img_cancer_label"
IMG = "/data3/ct/lph_ct/img_cancer"


def main():
    (RAW / "imagesTr").mkdir(parents=True, exist_ok=True)
    (RAW / "labelsTr").mkdir(parents=True, exist_ok=True)
    labs = sorted(glob.glob(f"{LAB}/*.nii.gz"))
    n = 0
    for lp in labs:
        stem = os.path.basename(lp).replace(".nii.gz", "")
        ip = f"{IMG}/{stem}.nii.gz"
        if not os.path.exists(ip):
            continue
        cid = f"LT_{n:03d}"
        shutil.copy(ip, RAW / "imagesTr" / f"{cid}_0000.nii.gz")
        # binarize label (some masks may carry label value !=1)
        m = nib.load(lp); a = (np.asanyarray(m.dataobj) > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(a, m.affine, m.header), RAW / "labelsTr" / f"{cid}.nii.gz")
        n += 1
        if n % 30 == 0:
            print(f"  {n} cases converted", flush=True)
    ds = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": n,
        "file_ending": ".nii.gz",
    }
    json.dump(ds, open(RAW / "dataset.json", "w"), indent=2)
    print(f"\nwrote {n} cases -> {RAW}")
    print("next:")
    print(f'  export nnUNet_raw="{RAW.parent}"; export nnUNet_preprocessed="{ROOT}/data/nnUNet_preprocessed"; '
          f'export nnUNet_results="{ROOT}/data/nnUNet_results"')
    print("  nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")
    print("  nnUNetv2_train 1 3d_fullres 0")


if __name__ == "__main__":
    main()
