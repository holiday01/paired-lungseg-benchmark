#!/usr/bin/env python3
"""Corrected per-tumour external FROC (review fix 2026-07-13), consistent with the in-site
froc_corrected.py definition: each expert GTV mask = one tumour; detected if a predicted
component reaches IoU>=0.1 with the mask; FP = predicted components (>=MIN_PRED_VOX, after
closing) not matching the tumour, per case. Same 5 headline checkpoints, whole 421-case cohort.
Replaces the naive connected-component external FROC so in-site vs external detection use the
SAME lesion definition.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import label as cc_label, binary_closing, generate_binary_structure

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT/"scripts"))
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference

EXT_DIR = ROOT/"data/external_nsclc_processed_1mm"
SEG_DIR = ROOT/"data/processed_seg_1mm"
OUT = ROOT/"ssl_study/results_external_froc_corrected.json"
PATCH = (96, 96, 96); HU_MIN, HU_MAX = -1000.0, 400.0
SEEDS = [7, 42, 101, 1337, 2024]
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
MIN_PRED_VOX = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    try: torch.cuda.set_per_process_memory_fraction(0.30)
    except Exception: pass

def norm(img):
    return ((np.clip(img, HU_MIN, HU_MAX)-HU_MIN)/(HU_MAX-HU_MIN)).astype(np.float32)

def pred_components(pred):
    # size filter only (no morphological closing): external GTV predictions are large and solid,
    # so closing is unnecessary and was the CPU bottleneck; detection is unchanged for big tumours.
    lab, n = cc_label((pred > 0).astype(np.uint8))
    if n == 0: return lab, 0, np.zeros(1, int)
    sizes = np.bincount(lab.ravel())
    keep = [k for k in range(1, n+1) if sizes[k] >= MIN_PRED_VOX]
    out = np.zeros_like(lab)
    for new, k in enumerate(keep, 1): out[lab == k] = new
    return out, len(keep), np.bincount(out.ravel())

def match_one_tumour(gt_mask, pred, iou_thr=0.1):
    g = gt_mask > 0; gsz = int(g.sum())
    pl, npd, psz = pred_components(pred)
    if npd == 0: return 0, 0
    detected = 0; matched = set()
    over = pl[g]
    if over.size:
        for pk, c in zip(*np.unique(over[over > 0], return_counts=True)):
            union = gsz + psz[pk] - c
            if union > 0 and c/union >= iou_thr:
                detected = 1; matched.add(int(pk))
    return detected, npd - len(matched)

def build():
    return DynUNet(spatial_dims=3, in_channels=1, out_channels=2, kernel_size=[3,3,3,3,3],
                   strides=[1,2,2,2,2], upsample_kernel_size=[2,2,2,2],
                   filters=[16,32,64,128,256], dropout=0.0).to(device)

files = sorted(glob.glob(str(EXT_DIR/"ext_*.npz")))
print(f"external cohort: {len(files)} cases")
per_seed = {}
for seed in SEEDS:
    ckpt = SEG_DIR/f"best_best_dynunet_s{seed}.pt"
    model = build(); model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
    n_scan = 0; tot = {t: {"det": 0, "fp": 0} for t in THRESHOLDS}
    with torch.no_grad():
        for i, f in enumerate(files):
            d = np.load(f, allow_pickle=True); msk = d["mask"].astype(np.uint8)
            if msk.sum() == 0: continue
            x = torch.from_numpy(norm(d["image"]))[None, None].to(device)
            logits = sliding_window_inference(x, PATCH, 4, model, overlap=0.25, mode="gaussian")
            p = torch.softmax(logits, 1)[0, 1].cpu().numpy().astype(np.float32)
            n_scan += 1
            for t in THRESHOLDS:
                det, fp = match_one_tumour(msk, (p >= t).astype(np.uint8))
                tot[t]["det"] += det; tot[t]["fp"] += fp
            del p, x, logits
            if (i+1) % 100 == 0: print(f"  seed {seed}: {i+1}/{len(files)}", flush=True)
        torch.cuda.empty_cache()
    per_seed[seed] = {"n_scan": n_scan, "froc": [
        {"thr": float(t), "sensitivity": tot[t]["det"]/max(n_scan, 1),
         "fp_per_scan": tot[t]["fp"]/max(n_scan, 1)} for t in THRESHOLDS]}
    s5 = per_seed[seed]["froc"][4]
    print(f"seed {seed}: n={n_scan} sens@0.5={s5['sensitivity']:.3f} FP={s5['fp_per_scan']:.2f}")
    del model; torch.cuda.empty_cache()

seeds_done = sorted(per_seed)
agg = {}
for i, t in enumerate(THRESHOLDS):
    sens = [per_seed[s]["froc"][i]["sensitivity"] for s in seeds_done]
    fps = [per_seed[s]["froc"][i]["fp_per_scan"] for s in seeds_done]
    agg[str(t)] = {"sens_mean": float(np.mean(sens)),
                   "sens_sd": float(np.std(sens, ddof=1)) if len(sens) > 1 else None,
                   "fp_mean": float(np.mean(fps)), "per_seed_sens": sens}
out = {"n_external_cases": len(files), "seeds": seeds_done, "froc_aggregate": agg,
       "lesion_def": "per-tumour: one GTV mask = one tumour; IoU>=0.1; FP=pred comps>=%d vox/case" % MIN_PRED_VOX}
OUT.write_text(json.dumps(out, indent=2))
print(f"\nwrote {OUT}")
print(f"External per-tumour FROC @0.5: sens={agg['0.5']['sens_mean']:.3f}+/-{agg['0.5']['sens_sd']:.3f} FP={agg['0.5']['fp_mean']:.2f}")
