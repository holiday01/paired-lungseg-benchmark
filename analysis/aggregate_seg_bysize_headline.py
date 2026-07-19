#!/usr/bin/env python3
"""Cross-seed CIs for the by-lesion-size Dice profile of the headline strategy (best DynUNet,
40k iterations, 5 seeds: 7/42/101/1337/2024). Answers the manuscript's
\\PENDING{cross-seed by-size CIs}. A band with no lesions in a given seed's test fold is
skipped for that seed (noted via n_seeds per band, not assumed zero).
"""
import glob
import json
from pathlib import Path
import numpy as np

RES = Path("/home/holiday/lung_ct/results")
OUT = Path("/home/holiday/lung_ct/ssl_study/results_bysize_5seed.json")
BANDS = ["<1cm3", "1-10cm3", "10-50cm3", ">=50cm3"]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}

per_seed = {}
for f in sorted(glob.glob(str(RES / "seg_best_dynunet_s*.json"))):
    d = json.load(open(f))
    per_seed[int(d["seed"])] = d["dice_by_size"]

seeds = sorted(per_seed)
out = {"seeds": seeds, "bands": {}}
for b in BANDS:
    vals = [per_seed[s][b][0] for s in seeds if per_seed[s][b][0] is not None]
    n_lesions = sum(per_seed[s][b][1] for s in seeds)
    n = len(vals)
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else None
    if n > 1:
        half = T975.get(n - 1, 1.96) * np.std(vals, ddof=1) / np.sqrt(n)
        ci95 = [round(mean - half, 4), round(mean + half, 4)]
    else:
        ci95 = None
    out["bands"][b] = {"dice_mean": round(mean, 4), "dice_sd": None if sd is None else round(sd, 4),
                        "dice_ci95": ci95, "n_seeds": n, "n_lesions_total": n_lesions}

OUT.write_text(json.dumps(out, indent=2))
print(f"wrote {OUT}")
for b, r in out["bands"].items():
    ci = f"[{r['dice_ci95'][0]:.3f},{r['dice_ci95'][1]:.3f}]" if r["dice_ci95"] else "(n<2)"
    print(f"{b:10s} dice={r['dice_mean']:.3f} {ci} n_seeds={r['n_seeds']} n_lesions={r['n_lesions_total']}")
