"""Connected-component statistics of the expert masks, before and after resampling.

Counts 26-connectivity-1 (scipy default) components in (a) every original
native-spacing mask under /data3 (read-only) and (b) every stored 1 mm
nearest-neighbor-resampled mask in data/processed_seg_1mm. Writes
results/mask_component_stats.json. Quoted in the manuscript's fragmentation
passage; no number is hand-entered.
"""
import glob
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "mask_component_stats.json"

def summarize(counts, fracs):
    c = np.array(counts)
    f = np.array([x for x in fracs if x is not None])
    return {
        "n_masks": int(len(c)),
        "n_multi_component": int((c > 1).sum()),
        "n_empty": int((c == 0).sum()),
        "components_median": float(np.median(c)),
        "components_mean": float(c.mean()),
        "components_max": int(c.max()),
        "largest_component_volume_fraction_median": float(np.median(f)),
        "largest_component_volume_fraction_min": float(f.min()),
    }

orig_counts, orig_fracs, empty_files = [], [], []
for fp in sorted(glob.glob("/data3/ct/lph_ct/img_cancer_label/*.nii.gz")):
    m = np.asanyarray(nib.load(fp).dataobj) > 0.5
    lab, n = label(m)
    orig_counts.append(n)
    if n == 0:
        orig_fracs.append(None)
        empty_files.append(Path(fp).name)
    else:
        sizes = np.bincount(lab.ravel())[1:]
        orig_fracs.append(float(sizes.max() / sizes.sum()))

proc_counts, proc_fracs = [], []
for fp in sorted(glob.glob(str(ROOT / "data" / "processed_seg_1mm" / "*.npz"))):
    with np.load(fp) as d:
        m = d["mask"] > 0
    lab, n = label(m)
    proc_counts.append(n)
    if n == 0:
        proc_fracs.append(None)
    else:
        sizes = np.bincount(lab.ravel())[1:]
        proc_fracs.append(float(sizes.max() / sizes.sum()))

out = {
    "note": "Components via scipy.ndimage.label (6-connectivity). Original masks binarized at >0.5.",
    "original_native_spacing": summarize(orig_counts, orig_fracs),
    "resampled_1mm_stored": summarize(proc_counts, proc_fracs),
    "empty_original_masks": empty_files,
}
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
