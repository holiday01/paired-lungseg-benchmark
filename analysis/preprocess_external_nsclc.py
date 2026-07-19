#!/usr/bin/env python3
"""Preprocess TCIA NSCLC-Radiomics (external cohort) into the SAME format/pipeline as the
in-site tumour cohort, for external validation of the headline segmentation model.

Identical steps to scripts/preprocess_seg.py: reorient to LPS, resample image (linear,
HU_AIR fill) + mask (nearest) to 1mm isotropic, clip to [-1000,400] HU, crop to tumour
bbox + 80mm margin, save as image:int16/mask:uint8/case:str npz -- so the exact same
inference code (norm(), PATCH, DynUNet architecture) applies unchanged.

Source: data/external/nsclc_radiomics/NSCLC-Radiomics-NIFTI/LUNG1-XXX/{image,seg-GTV-1}.nii.gz
(HuggingFace mirror farrell236/NSCLC-Radiomics-NIFTI of the public TCIA collection).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "scripts"))
import preprocess as P  # noqa: E402

SRC = ROOT / "data" / "external" / "nsclc_radiomics" / "NSCLC-Radiomics-NIFTI"
OUT = ROOT / "data" / "external_nsclc_processed_1mm"
OUT.mkdir(parents=True, exist_ok=True)
SPACING = (1.0, 1.0, 1.0)
HU_MIN, HU_MAX = -1000, 400
HU_AIR = getattr(P, "HU_AIR", -1024)
MARGIN_MM = 80.0


def bbox_with_margin(mask, margin_vox):
    nz = np.argwhere(mask > 0)
    if not len(nz):
        return tuple(slice(0, s) for s in mask.shape)
    lo = np.maximum(nz.min(0) - margin_vox, 0)
    hi = np.minimum(nz.max(0) + margin_vox + 1, mask.shape)
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def main():
    margin_vox = int(round(MARGIN_MM / SPACING[0]))
    case_dirs = sorted(d for d in SRC.iterdir() if d.is_dir())
    print(f"found {len(case_dirs)} case directories in {SRC}")
    n_ok, n_fail, n_empty = 0, 0, 0
    sizes_cm3 = []
    for i, cdir in enumerate(case_dirs):
        case = cdir.name
        ip, lp = cdir / "image.nii.gz", cdir / "seg-GTV-1.nii.gz"
        if not (ip.exists() and lp.exists()):
            print(f"[{i}] {case}: missing image or GTV-1 mask, skip")
            n_fail += 1
            continue
        try:
            img = P.reorient_lps(sitk.ReadImage(str(ip)))
            msk = P.reorient_lps(sitk.ReadImage(str(lp)))
            img_rs = P.resample_to_spacing(img, SPACING, sitk.sitkLinear, HU_AIR)
            msk_rs = P.resample_to_spacing(msk, SPACING, sitk.sitkNearestNeighbor, 0)
            ia = np.clip(sitk.GetArrayFromImage(img_rs), HU_MIN, HU_MAX).astype(np.int16)
            ma = (sitk.GetArrayFromImage(msk_rs) > 0).astype(np.uint8)
            if ia.shape != ma.shape:
                ma = ma[tuple(slice(0, s) for s in ia.shape)]
            if ma.sum() == 0:
                print(f"[{i}] {case}: empty GTV-1 mask after resample, skip")
                n_empty += 1
                continue
            sl = bbox_with_margin(ma, margin_vox)
            ia, ma = ia[sl], ma[sl]
        except Exception as e:
            print(f"[{i}] {case}: FAIL {type(e).__name__}: {e}")
            n_fail += 1
            continue
        np.savez_compressed(OUT / f"ext_{case}.npz", image=ia, mask=ma, case=case)
        tum_cm3 = float(ma.sum()) * np.prod(SPACING) / 1000.0
        sizes_cm3.append(tum_cm3)
        n_ok += 1
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(case_dirs)}] ok={n_ok} fail={n_fail} empty={n_empty}")

    print(f"\nDONE: {n_ok} ok, {n_fail} failed, {n_empty} empty-mask, out of {len(case_dirs)} -> {OUT}")
    if sizes_cm3:
        arr = np.array(sizes_cm3)
        print(f"tumor size cm3: median={np.median(arr):.1f} min={arr.min():.2f} max={arr.max():.1f}")


if __name__ == "__main__":
    main()
