#!/usr/bin/env python3
"""Bring-your-own-data preprocessing for the segmentation benchmark.

Turns a directory of CT images and matching tumour masks into the compact
``.npz`` volumes that ``train_seg.py`` consumes. You provide:

    data/img/   your CT images   (NIfTI .nii/.nii.gz or NRRD)
    data/msk/   the tumour masks  (same file names as the images)
    data/manifest.csv   a table listing which image goes with which mask

Manifest columns (header required):
    image   file name inside --img-dir
    mask    file name inside --msk-dir
    case    (optional) patient id used to keep all of one patient's tumours
            on the same side of a train/val/test split. If omitted, the case
            id is the image stem with a trailing ``_B<number>`` stripped, so
            ``patientA_B00`` and ``patientA_B01`` are grouped as ``patientA``.

Each pair is reoriented to LPS, resampled to an isotropic grid (linear for the
image, nearest-neighbour for the mask), HU-windowed, cropped to the tumour
bounding box plus a margin, and saved as ``<case-hash>.npz`` holding
``{image, mask, case}`` -- exactly the schema ``train_seg.py`` expects.

Example
-------
    python scripts/prepare_data.py \
        --manifest data/manifest.csv --img-dir data/img --msk-dir data/msk \
        --out data/processed_seg --spacing 1.0

Then train a strategy on the result:
    python scripts/train_seg.py --seg-dir data/processed_seg --tag S2_dynunet_plain --seed 7
"""
from __future__ import annotations
import argparse, csv, hashlib, os, re, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

HU_MIN, HU_MAX = -1000, 400   # lung-CT window used throughout the benchmark
HU_AIR = -1000
MARGIN_MM = 80.0


def case_hash(stem: str) -> str:
    """Stable, non-reversible id so output names carry no patient information."""
    return "seg_" + hashlib.sha1(stem.encode()).hexdigest()[:12]


def default_case(stem: str) -> str:
    return re.sub(r"_B\d+$", "", stem)


def reorient_lps(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, "LPS")


def resample(img: sitk.Image, spacing, interp, pad_value) -> sitk.Image:
    in_spacing = np.array(img.GetSpacing(), float)
    in_size = np.array(img.GetSize(), int)
    out_spacing = np.array(spacing, float)
    out_size = np.ceil(in_size * in_spacing / out_spacing).astype(int).tolist()
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(out_spacing.tolist())
    r.SetSize(out_size)
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputDirection(img.GetDirection())
    r.SetInterpolator(interp)
    r.SetDefaultPixelValue(float(pad_value))
    return r.Execute(img)


def bbox_with_margin(mask: np.ndarray, margin_vox: int):
    nz = np.argwhere(mask > 0)
    if not len(nz):
        return tuple(slice(0, s) for s in mask.shape)
    lo = np.maximum(nz.min(0) - margin_vox, 0)
    hi = np.minimum(nz.max(0) + margin_vox + 1, mask.shape)
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV with columns image,mask[,case]")
    ap.add_argument("--img-dir", default="data/img")
    ap.add_argument("--msk-dir", default="data/msk")
    ap.add_argument("--out", default="data/processed_seg")
    ap.add_argument("--spacing", type=float, default=1.0, help="isotropic mm")
    a = ap.parse_args()

    img_dir, msk_dir, out = Path(a.img_dir), Path(a.msk_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spacing = (a.spacing,) * 3
    margin_vox = int(round(MARGIN_MM / a.spacing))

    with open(a.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "image" not in rows[0] or "mask" not in rows[0]:
        sys.exit("manifest must have at least 'image' and 'mask' columns")

    written = []
    for i, r in enumerate(rows):
        ip, mp = img_dir / r["image"], msk_dir / r["mask"]
        stem = re.sub(r"\.nii(\.gz)?$|\.nrrd$", "", r["image"])
        case = (r.get("case") or "").strip() or default_case(stem)
        if not ip.exists() or not mp.exists():
            print(f"[{i}] {r['image']}: missing image or mask, skip"); continue
        try:
            img = resample(reorient_lps(sitk.ReadImage(str(ip))), spacing, sitk.sitkLinear, HU_AIR)
            msk = resample(reorient_lps(sitk.ReadImage(str(mp))), spacing, sitk.sitkNearestNeighbor, 0)
            ia = np.clip(sitk.GetArrayFromImage(img), HU_MIN, HU_MAX).astype(np.int16)
            ma = (sitk.GetArrayFromImage(msk) > 0).astype(np.uint8)
            if ia.shape != ma.shape:
                ma = ma[tuple(slice(0, s) for s in ia.shape)]
            sl = bbox_with_margin(ma, margin_vox)
            ia, ma = ia[sl], ma[sl]
        except Exception as e:
            print(f"[{i}] {r['image']}: FAIL {type(e).__name__}: {e}"); continue
        aid = case_hash(stem)
        np.savez_compressed(out / f"{aid}.npz", image=ia, mask=ma, case=case)
        tum_cm3 = float(ma.sum()) * (a.spacing ** 3) / 1000.0
        written.append({"file": f"{aid}.npz", "case": case,
                        "shape": "x".join(map(str, ia.shape)), "tumor_cm3": round(tum_cm3, 2)})
        print(f"[{i+1}/{len(rows)}] case={case} shape={ia.shape} tumor={tum_cm3:.2f}cm3")

    with open(out / "manifest_seg.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "case", "shape", "tumor_cm3"])
        w.writeheader(); w.writerows(written)
    n_cases = len({r["case"] for r in written})
    print(f"\nwrote {len(written)} volumes over {n_cases} cases -> {out}")


if __name__ == "__main__":
    main()
