#!/usr/bin/env python3
"""Lesion-level FROC analysis for lung-tumor segmentation.
Connected-component lesion matching: per-GT-lesion detection sensitivity vs false-positives/scan,
swept over the model's foreground probability threshold. The proper detection metric (vs the crude
case-level Dice>0.1 proxy). Runs inference from saved checkpoints; CPU-safe (does not contend GPU).

A predicted component is a TRUE POSITIVE for a GT lesion if their overlap (IoU or centroid-in-GT)
meets a criterion; unmatched predicted components are FALSE POSITIVES; unmatched GT lesions are misses.
"""
import sys, json, glob, argparse, numpy as np, torch
from pathlib import Path
from scipy.ndimage import label as cc_label
sys.path.insert(0, "/home/holiday/lung_ct/scripts")
import train_seg as TS
from monai.inferers import sliding_window_inference

ROOT = Path("/home/holiday/lung_ct"); OUT = ROOT/"ssl_study"
# GPU is shared with another project (WSI distillation, ~24GB). Cap our share to coexist safely.
if torch.cuda.is_available():
    try: torch.cuda.set_per_process_memory_fraction(0.18)   # ~6GB of 32GB — fits in the free 7GB
    except Exception: pass
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
PATCH = TS.PATCH

MIN_LESION_VOX = 10   # drop resampling-fragment components (<10 voxels) before lesion matching

def _label_filtered(mask):
    """connected components with sub-threshold fragments removed (relabelled 1..k)."""
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
    """Efficient lesion-level matching via label maps (O(voxels)), fragments <MIN_LESION_VOX removed.
    A predicted component is a TP for a GT lesion if their IoU>=thr. Returns (n_gt, n_detected_gt, n_fp)."""
    gl, ng, gt_size = _label_filtered(gt)
    pl, npd, pr_size = _label_filtered(pred)
    if ng == 0:
        return 0, 0, npd
    # overlap counts between every (pred k, gt j) via a 2D histogram on overlapping voxels
    mask = (gl > 0) & (pl > 0)
    inter = {}
    if mask.any():
        pairs = pl[mask].astype(np.int64) * (ng + 1) + gl[mask].astype(np.int64)
        for code, c in zip(*np.unique(pairs, return_counts=True)):
            pk, gj = divmod(int(code), ng + 1)
            inter[(pk, gj)] = c
    detected = set(); used_pred = set()
    # greedy: for each gt, find a pred whose IoU>=thr (try largest-overlap first)
    by_gt = {}
    for (pk, gj), c in inter.items():
        by_gt.setdefault(gj, []).append((c, pk))
    for gj in range(1, ng + 1):
        for c, pk in sorted(by_gt.get(gj, []), reverse=True):
            if pk in used_pred: continue
            union = gt_size[gj] + pr_size[pk] - c
            if union > 0 and c / union >= iou_thr:
                detected.add(gj); used_pred.add(pk); break
    n_fp = npd - len(used_pred)
    return ng, len(detected), n_fp

def build_model(model_name):
    cfg = argparse.Namespace(model=model_name)
    # reuse train_seg's make(): replicate the dynunet branch directly to avoid full main()
    from monai.networks.nets import DynUNet, BasicUNet
    if model_name == "dynunet":
        net = DynUNet(spatial_dims=3, in_channels=1, out_channels=2,
                      kernel_size=[3,3,3,3,3], strides=[1,2,2,2,2],
                      upsample_kernel_size=[2,2,2,2], filters=[16,32,64,128,256])
    elif model_name == "basicunet":
        net = BasicUNet(spatial_dims=3, in_channels=1, out_channels=2, features=(32,32,64,128,256,32))
    else:
        raise ValueError(model_name)
    return net.to(DEVICE).eval()

def froc_for_checkpoint(ckpt, model_name, test_cases, thresholds):
    net = build_model(model_name)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    # collect per-scan foreground prob volumes once, then sweep thresholds
    probs, gts = [], []
    with torch.no_grad():
        for i, (img, msk, _) in enumerate(test_cases):
            if msk.sum() == 0: continue
            x = torch.from_numpy(img)[None,None].to(DEVICE)
            logit = sliding_window_inference(x, PATCH, 4, net, overlap=0.25, mode="gaussian")
            p = torch.softmax(logit, 1)[0,1].cpu().numpy().astype(np.float32)  # foreground prob
            probs.append(p); gts.append(msk.astype(np.uint8))
            print(f"    inferred {i+1}/{len(test_cases)}", flush=True)
    n_scan = len(probs)
    curve = []
    for t in thresholds:
        tot_gt, tot_det, tot_fp = 0, 0, 0
        for p, g in zip(probs, gts):
            ng, d, fp = lesion_match(g, (p >= t).astype(np.uint8))
            tot_gt += ng; tot_det += d; tot_fp += fp
        curve.append({"thr": float(t), "sensitivity": tot_det/max(tot_gt,1),
                      "fp_per_scan": tot_fp/max(n_scan,1), "n_gt_lesions": tot_gt})
        print(f"  thr={t:.1f} sens={curve[-1]['sensitivity']:.3f} FP/scan={curve[-1]['fp_per_scan']:.2f}", flush=True)
    return curve, n_scan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="dynunet")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="froc")
    a = ap.parse_args()
    # reproduce the SAME patient-level test split train_seg uses for this seed
    import glob as _g
    files = sorted(_g.glob(str(ROOT/"data/processed_seg_1mm/*.npz")))
    cases = {}
    for f in files:
        c = str(np.load(f, allow_pickle=True)["case"]); cases.setdefault(c, []).append(f)
    case_ids = sorted(cases); rng = np.random.RandomState(a.seed); rng.shuffle(case_ids)
    n = len(case_ids); n_test = max(1, n//5); n_val = max(1, n//5)
    test_c = case_ids[:n_test]
    test_cases = [TS.load_seg(f) for c in test_c for f in cases[c]]
    print(f"FROC: {a.ckpt.split('/')[-1]}  test={len(test_c)} cases / {len(test_cases)} volumes")
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    curve, n_scan = froc_for_checkpoint(a.ckpt, a.model, test_cases, thresholds)
    for r in curve:
        print(f"  thr={r['thr']:.1f}  sens={r['sensitivity']:.3f}  FP/scan={r['fp_per_scan']:.2f}  (n_gt={r['n_gt_lesions']})")
    res = {"ckpt": a.ckpt, "model": a.model, "seed": a.seed, "n_scan": n_scan, "froc": curve}
    outp = OUT/f"froc_{a.tag}.json"; json.dump(res, open(outp,"w"), indent=2)
    print("wrote", outp)

if __name__ == "__main__":
    main()
