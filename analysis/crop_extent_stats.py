"""Crop-extent statistics for the stored lesion-centered volumes.

Both cohorts are stored as tumor-bounding-box + 80 mm margin crops at 1 mm
isotropic spacing (see preprocess/prepare_data). This script computes the
per-axis extent statistics quoted in the manuscript's Methods and writes
them to results/crop_extent_stats.json. No number is hand-entered.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "crop_extent_stats.json"

def stats(pattern):
    shapes = []
    for f in sorted(glob.glob(pattern)):
        with np.load(f) as d:
            shapes.append(d["image"].shape)
    a = np.array(shapes)
    return {
        "n_volumes": int(len(a)),
        "median_extent_mm": [float(x) for x in np.median(a, 0)],
        "min_extent_mm": [int(x) for x in a.min(0)],
        "max_extent_mm": [int(x) for x in a.max(0)],
    }

out = {
    "note": "1 mm isotropic voxels, so voxel counts equal mm. Crops are tumor bbox + 80 mm margin.",
    "in_site": stats(str(ROOT / "data" / "processed_seg_1mm" / "*.npz")),
    "external": stats(str(ROOT / "data" / "external_nsclc_processed_1mm" / "ext_*.npz")),
}
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
