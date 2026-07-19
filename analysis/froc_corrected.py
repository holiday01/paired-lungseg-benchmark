#!/usr/bin/env python3
"""CORRECTED lesion-level FROC (review fix, 2026-07-13).

The original froc_analysis.py defined a GT "lesion" as a connected component of the
1 mm-resampled mask (>=10 vox). Resampling fragments each expert tumour into ~4.5 components
(145 masks -> 648 components), so the old per-lesion sensitivity (0.385) counted the model as
missing tiny same-tumour fragments, not distinct tumours. This version defines the GT lesion as
the EXPERT ANNOTATION UNIT: each .npz mask = one tumour. Detection = some predicted component
overlaps that tumour at IoU>=thr. FP = predicted components (>=MIN_PRED_VOX) that overlap no
tumour, per cropped volume. Reports per-tumour detection sensitivity vs FP per volume.

Same per-seed patient split as train_seg. Runs inference from the saved headline checkpoints.
"""
import sys, json, glob, argparse, numpy as np, torch
from pathlib import Path
from scipy.ndimage import label as cc_label, binary_closing, generate_binary_structure
sys.path.insert(0, "/home/holiday/lung_ct/scripts")
import train_seg as TS
from monai.inferers import sliding_window_inference

ROOT = Path("/home/holiday/lung_ct"); OUT = ROOT/"ssl_study"
if torch.cuda.is_available():
    try: torch.cuda.set_per_process_memory_fraction(0.30)
    except Exception: pass
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
PATCH = TS.PATCH
STRUCT = generate_binary_structure(3, 3)
MIN_PRED_VOX = 50   # a predicted blob must exceed this to count as a (TP or FP) detection

def pred_components(pred):
    """closed connected components of the prediction, fragments < MIN_PRED_VOX removed."""
    m = binary_closing(pred > 0, structure=STRUCT, iterations=1)
    lab, n = cc_label(m.astype(np.uint8))
    if n == 0:
        return lab, 0, np.zeros(1, int)
    sizes = np.bincount(lab.ravel())
    keep = [k for k in range(1, n+1) if sizes[k] >= MIN_PRED_VOX]
    out = np.zeros_like(lab)
    for new, k in enumerate(keep, 1):
        out[lab == k] = new
    return out, len(keep), np.bincount(out.ravel())

def match_one_tumour(gt_mask, pred, iou_thr=0.1):
    """GT = one tumour (whole mask). Returns (detected: 0/1, n_fp)."""
    g = gt_mask > 0
    gsz = int(g.sum())
    pl, npd, psz = pred_components(pred)
    if npd == 0:
        return 0, 0
    detected = 0; matched = set()
    over = pl[g]                                   # pred labels inside the tumour
    if over.size:
        for pk, c in zip(*np.unique(over[over > 0], return_counts=True)):
            union = gsz + psz[pk] - c
            if union > 0 and c/union >= iou_thr:
                detected = 1; matched.add(int(pk))
    n_fp = npd - len(matched)                      # pred blobs not matching the tumour
    return detected, n_fp

def froc(ckpt, model_name, test_cases, thresholds):
    from monai.networks.nets import DynUNet
    net = DynUNet(spatial_dims=3, in_channels=1, out_channels=2,
                  kernel_size=[3,3,3,3,3], strides=[1,2,2,2,2],
                  upsample_kernel_size=[2,2,2,2], filters=[16,32,64,128,256]).to(DEVICE).eval()
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    probs, gts = [], []
    with torch.no_grad():
        for img, msk, _ in test_cases:
            if msk.sum() == 0: continue            # skip empty/contaminated masks
            x = torch.from_numpy(img)[None, None].to(DEVICE)
            logit = sliding_window_inference(x, PATCH, 4, net, overlap=0.25, mode="gaussian")
            probs.append(torch.softmax(logit, 1)[0, 1].cpu().numpy().astype(np.float32))
            gts.append(msk.astype(np.uint8))
    n_scan = len(probs)
    curve = []
    for t in thresholds:
        det = fp = 0
        for p, g in zip(probs, gts):
            d, f = match_one_tumour(g, (p >= t).astype(np.uint8))
            det += d; fp += f
        curve.append({"thr": float(t), "sensitivity": det/max(n_scan, 1),
                      "fp_per_scan": fp/max(n_scan, 1), "n_gt_tumours": n_scan})
        print(f"  thr={t:.1f} sens={det/max(n_scan,1):.3f} FP/vol={fp/max(n_scan,1):.2f} (n_tum={n_scan})", flush=True)
    return curve, n_scan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="froc_corr")
    a = ap.parse_args()
    files = sorted(glob.glob(str(ROOT/"data/processed_seg_1mm/*.npz")))
    cases = {}
    for f in files:
        c = str(np.load(f, allow_pickle=True)["case"]); cases.setdefault(c, []).append(f)
    case_ids = sorted(cases); rng = np.random.RandomState(a.seed); rng.shuffle(case_ids)
    test_c = case_ids[:max(1, len(case_ids)//5)]
    test_cases = [TS.load_seg(f) for c in test_c for f in cases[c]]
    print(f"FROC-corrected: {Path(a.ckpt).name} test={len(test_c)} cases / {len(test_cases)} volumes")
    curve, n_scan = froc(a.ckpt, "dynunet", test_cases, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    json.dump({"ckpt": a.ckpt, "seed": a.seed, "n_scan": n_scan, "froc": curve,
               "lesion_def": "one expert mask = one tumour; detect if a pred component IoU>=0.1; "
                             "FP = pred components >= %d vox not matching the tumour, per volume" % MIN_PRED_VOX},
              open(OUT/f"{a.tag}.json", "w"), indent=2)
    print("wrote", OUT/f"{a.tag}.json")

if __name__ == "__main__":
    main()
