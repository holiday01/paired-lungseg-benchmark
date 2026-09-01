"""Cohort integrity and volume statistics for the stored segmentation volumes.

Audits data/processed_seg_1mm: file/patient counts, empty masks (excluded from
evaluation by the empty-ground-truth rule), byte-identical duplicate masks, and
tumor-volume statistics over the non-empty masks. Writes
results/seg_cohort_stats.json; quoted in the manuscript's Cohort paragraph.
"""
import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "seg_cohort_stats.json"

vols, cases, empty, mask_hashes = [], [], [], defaultdict(list)
for f in sorted(glob.glob(str(ROOT / "data" / "processed_seg_1mm" / "*.npz"))):
    name = Path(f).name
    with np.load(f) as d:
        m = (d["mask"] > 0)
        case = str(d["case"])
    v = float(m.sum()) / 1000.0  # cm^3 at 1 mm isotropic
    vols.append(v)
    cases.append(case)
    if v == 0:
        empty.append({"file": name, "case": case})
    mask_hashes[hashlib.sha1(m.tobytes()).hexdigest()].append({"file": name, "case": case})

v = np.array(vols)
nz = v[v > 0]
mc = Counter(cases)
dups = [files for files in mask_hashes.values() if len(files) > 1 and files[0] not in empty]
dups = [g for g in dups if all(e["file"] not in [x["file"] for x in empty] for e in g)]

out = {
    "n_mask_files": int(len(v)),
    "n_patients": len(mc),
    "n_patients_multiple_files": sum(1 for c in mc.values() if c > 1),
    "n_empty_masks": len(empty),
    "empty_masks": empty,
    "duplicate_nonempty_mask_groups": dups,
    "nonempty": {
        "n": int(len(nz)),
        "volume_cm3_min": float(nz.min()),
        "volume_cm3_median": float(np.median(nz)),
        "volume_cm3_max": float(nz.max()),
        "frac_below_1cm3": float((nz < 1).mean()),
        "n_below_1cm3": int((nz < 1).sum()),
    },
    "all_files_including_empty": {
        "volume_cm3_median": float(np.median(v)),
        "frac_below_1cm3": float((v < 1).mean()),
        "note": "the manuscript's earlier 4.3 cm3 / 18.6% figures were computed over this set, "
                "with the 7 empty masks counted as <1 cm3",
    },
}
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
