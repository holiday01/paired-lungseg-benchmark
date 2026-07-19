#!/usr/bin/env python3
"""Aggregate nnU-Net 3d_fullres validation Dice across the trained patient-grouped folds.

nnU-Net writes per-fold validation/summary.json with a foreground-mean Dice. This collects
whatever folds have finished, reports per-fold and mean +/- SD, and writes
ssl_study/results_nnunet_ceiling.json. Honest about how many folds completed (the ceiling is
nnU-Net's CV Dice, not a held-out 20-patient test, so it is an optimistic upper reference for
the DynUNet 0.659 +/- 0.028 and should be labelled as such in the paper).
"""
import json
from pathlib import Path
import numpy as np

RESULTS = Path("/home/holiday/lung_ct/data/nnUNet_results/Dataset001_LungTumor")
OUT = Path("/home/holiday/lung_ct/ssl_study/results_nnunet_ceiling.json")


def find_fold_dices():
    """Return the fold-Dice dict for the trainer directory with the most complete folds.

    Folds trained under different trainer classes (e.g. differing epoch budgets) are not
    comparable, so this picks ONE trainer directory rather than merging across them, and
    warns if more than one trainer directory has completed folds (so a mixed-epoch-budget
    ceiling doesn't silently sneak into the aggregate).
    """
    if not RESULTS.exists():
        return {}, None

    candidates = {}
    for trainer_dir in RESULTS.glob("*__nnUNetPlans__3d_fullres"):
        folds = {}
        for fdir in sorted(trainer_dir.glob("fold_*")):
            summ = fdir / "validation" / "summary.json"
            if not summ.exists():
                continue
            d = json.load(open(summ))
            # nnU-Net v2: foreground_mean.Dice
            dice = d.get("foreground_mean", {}).get("Dice")
            if dice is None and "mean" in d:
                # fall back: mean over labels excluding background
                labels = {k: v for k, v in d["mean"].items() if k not in ("0", "background")}
                dice = np.mean([v["Dice"] for v in labels.values()]) if labels else None
            if dice is not None:
                folds[fdir.name] = float(dice)
        if folds:
            candidates[trainer_dir.name] = folds

    if not candidates:
        return {}, None
    if len(candidates) > 1:
        print(f"WARNING: {len(candidates)} trainer directories have completed folds "
              f"({', '.join(candidates)}) -- folds trained under different trainer classes "
              "are not comparable (e.g. different epoch budgets). Using the most complete "
              "one and IGNORING the rest; verify this is intentional.")

    # Prefer the trainer with the most completed folds; on a tie prefer the FULL default
    # trainer ("nnUNetTrainer__...") over reduced-epoch variants ("nnUNetTrainer_250epochs__...").
    def pref(t):
        return (len(candidates[t]), t.startswith("nnUNetTrainer__"))
    best_trainer = max(candidates, key=pref)
    return candidates[best_trainer], best_trainer


folds, trainer = find_fold_dices()
if not folds:
    print("No completed nnU-Net validation summaries found yet "
          f"(looked under {RESULTS}). Train at least one fold first.")
    raise SystemExit(0)

vals = np.array(list(folds.values()), float)
out = {
    "trainer_plan": trainer,
    "n_folds_done": len(folds),
    "per_fold_dice": folds,
    "cv_dice_mean": round(float(vals.mean()), 4),
    "cv_dice_sd": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None,
    "comparison": {
        "dynunet_5seed_dice": 0.659,
        "dynunet_5seed_sd": 0.028,
        "note": ("nnU-Net value is patient-grouped 5-fold CROSS-VALIDATION Dice (more training "
                 "data per model and CV rather than a single 20-patient held-out test), so it is "
                 "an optimistic upper reference, not a head-to-head on identical test patients."),
    },
}
OUT.write_text(json.dumps(out, indent=2))
print(f"wrote {OUT}")
print(f"nnU-Net {trainer}: {len(folds)} fold(s), CV Dice mean = {vals.mean():.4f}"
      + (f" +/- {vals.std(ddof=1):.4f}" if len(vals) > 1 else " (single fold)"))
for k, v in folds.items():
    print(f"  {k}: {v:.4f}")
print(f"vs DynUNet 5-seed 0.659 +/- 0.028")
