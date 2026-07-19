#!/usr/bin/env python3
"""External validation of the headline DynUNet (5 seeds) on TCIA NSCLC-Radiomics.

For each seed's checkpoint: per-case Dice (matching train_seg.py's evaluate() formula
exactly) and lesion-level FROC (matching froc_analysis.py's connected-component IoU>=0.1
matching, MIN_LESION_VOX=10 fragment filter, threshold sweep) over the WHOLE external
cohort (no train/val/test split needed -- it's held out by construction, never seen in
training). Aggregates across the 5 seeds in the SAME format as the in-site
froc_5seed_summary.json (mean/SD/per-seed at each threshold) for a direct comparison.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as cc_label

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "scripts"))
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference

EXT_DIR = ROOT / "data" / "external_nsclc_processed_1mm"
SEG_DIR = ROOT / "data" / "processed_seg_1mm"
OUT = ROOT / "ssl_study" / "results_external_nsclc.json"
PATCH = (96, 96, 96)
HU_MIN, HU_MAX = -1000.0, 400.0
SEEDS = [7, 42, 101, 1337, 2024]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_LESION_VOX = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm(img):
    return ((np.clip(img, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def build_model():
    net = DynUNet(spatial_dims=3, in_channels=1, out_channels=2,
                  kernel_size=[3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 2],
                  upsample_kernel_size=[2, 2, 2, 2], filters=[16, 32, 64, 128, 256], dropout=0.0)
    return net.to(device)


def _label_filtered(mask):
    lab, n = cc_label(mask.astype(np.uint8))
    if n == 0:
        return lab, 0, np.zeros(1, int)
    sizes = np.bincount(lab.ravel())
    keep = [k for k in range(1, n + 1) if sizes[k] >= MIN_LESION_VOX]
    out = np.zeros_like(lab)
    for new_id, k in enumerate(keep, start=1):
        out[lab == k] = new_id
    return out, len(keep), np.bincount(out.ravel())


def lesion_match(gt, pred, iou_thr=0.1):
    gl, ng, gt_size = _label_filtered(gt)
    pl, npd, pr_size = _label_filtered(pred)
    if ng == 0:
        return 0, 0, npd
    mask = (gl > 0) & (pl > 0)
    inter = {}
    if mask.any():
        pairs = pl[mask].astype(np.int64) * (ng + 1) + gl[mask].astype(np.int64)
        for code, c in zip(*np.unique(pairs, return_counts=True)):
            pk, gj = divmod(int(code), ng + 1)
            inter[(pk, gj)] = c
    detected, used_pred = set(), set()
    by_gt = {}
    for (pk, gj), c in inter.items():
        by_gt.setdefault(gj, []).append((c, pk))
    for gj in range(1, ng + 1):
        for c, pk in sorted(by_gt.get(gj, []), reverse=True):
            if pk in used_pred:
                continue
            union = gt_size[gj] + pr_size[pk] - c
            if union > 0 and c / union >= iou_thr:
                detected.add(gj)
                used_pred.add(pk)
                break
    return ng, len(detected), npd - len(used_pred)


files = sorted(glob.glob(str(EXT_DIR / "ext_*.npz")))
print(f"external cohort: {len(files)} cases")

per_seed = {}
for seed in SEEDS:
    ckpt = SEG_DIR / f"best_best_dynunet_s{seed}.pt"
    if not ckpt.exists():
        print(f"MISSING checkpoint seed {seed}")
        continue
    model = build_model()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Per-threshold running totals, computed per case then discarded (never hold all 421
    # cases' probability volumes at once -- external tumors are much larger than in-site
    # ones (median 39cm3 vs 4.3cm3), so their crops are correspondingly bigger).
    dices = []
    n_scan = 0
    totals = {t: {"gt": 0, "det": 0, "fp": 0} for t in THRESHOLDS}
    with torch.no_grad():
        for i, f in enumerate(files):
            d = np.load(f, allow_pickle=True)
            img_raw, msk = d["image"], d["mask"].astype(np.uint8)
            if msk.sum() == 0:
                continue
            x = torch.from_numpy(norm(img_raw))[None, None].to(device)
            logits = sliding_window_inference(x, PATCH, 2, model, overlap=0.5, mode="gaussian")
            p = torch.softmax(logits, 1)[0, 1].cpu().numpy().astype(np.float32)
            pred05 = (p >= 0.5).astype(np.uint8)
            inter = float((pred05.astype(bool) & msk.astype(bool)).sum())
            denom = float(pred05.sum() + msk.sum())
            dices.append(2 * inter / denom if denom > 0 else 0.0)
            n_scan += 1
            for t in THRESHOLDS:
                ng, det, fp = lesion_match(msk, (p >= t).astype(np.uint8))
                totals[t]["gt"] += ng
                totals[t]["det"] += det
                totals[t]["fp"] += fp
            del p, pred05, x, logits
            if (i + 1) % 50 == 0:
                print(f"  seed {seed}: {i+1}/{len(files)} inferred", flush=True)
        torch.cuda.empty_cache()

    curve = [{"thr": float(t), "sensitivity": totals[t]["det"] / max(totals[t]["gt"], 1),
              "fp_per_scan": totals[t]["fp"] / max(n_scan, 1), "n_gt_lesions": totals[t]["gt"]}
             for t in THRESHOLDS]

    per_seed[seed] = {"n_scan": n_scan, "dice_mean": float(np.mean(dices)),
                       "dice_sd": float(np.std(dices, ddof=1)), "froc": curve}
    print(f"seed {seed}: n={n_scan} dice={np.mean(dices):.3f}+/-{np.std(dices, ddof=1):.3f}")
    del model
    torch.cuda.empty_cache()

# aggregate across seeds, same shape as the in-site froc_5seed_summary.json
seeds_done = sorted(per_seed)
dice_means = [per_seed[s]["dice_mean"] for s in seeds_done]
froc_aggregate = {}
for i, t in enumerate(THRESHOLDS):
    sens = [per_seed[s]["froc"][i]["sensitivity"] for s in seeds_done]
    fps = [per_seed[s]["froc"][i]["fp_per_scan"] for s in seeds_done]
    froc_aggregate[str(t)] = {
        "sens_mean": float(np.mean(sens)), "sens_sd": float(np.std(sens, ddof=1)) if len(sens) > 1 else None,
        "fp_mean": float(np.mean(fps)), "per_seed_sens": sens,
    }

out = {
    "n_external_cases": len(files),
    "seeds": seeds_done,
    "dice_mean_across_seeds": float(np.mean(dice_means)),
    "dice_sd_across_seeds": float(np.std(dice_means, ddof=1)) if len(dice_means) > 1 else None,
    "per_seed": per_seed,
    "froc_aggregate": froc_aggregate,
    "note": ("External cohort = TCIA NSCLC-Radiomics (421 cases w/ GTV-1 masks), never seen in "
             "training. Same DynUNet checkpoints as the in-site headline result. Compare "
             "dice_mean_across_seeds to in-site 0.659+/-0.028, and froc_aggregate['0.5'] "
             "sens_mean to in-site 0.385 @ ~4.3 FP/scan."),
}
OUT.write_text(json.dumps(out, indent=2))
print(f"\nwrote {OUT}")
print(f"External Dice: {out['dice_mean_across_seeds']:.3f} +/- {out['dice_sd_across_seeds']:.3f}")
print(f"External FROC @0.5: sens={froc_aggregate['0.5']['sens_mean']:.3f} FP/scan={froc_aggregate['0.5']['fp_mean']:.2f}")
