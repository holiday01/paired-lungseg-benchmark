#!/usr/bin/env python3
"""Aggregate the multi-seed segmentation strategy runs into a ranking WITH cross-seed CIs
(Study 2 / JMI headline). Reads the per-run timestamped JSONs in results/, groups by strategy
across seeds {7, 42, 1337}, dedupes duplicate retries (latest mtime per seed), and reports
mean / SD / 95% t-CI of test Dice per strategy plus a rank. Emits seg_ranking_3seed.json and a
paste-ready LaTeX table body.

Run this once the SEGMS_* reruns are all VERIFIED. It computes only from real run files; no
number is hand-entered. Honest about n_seeds per strategy (prints a warning if any strategy has
fewer than 3 seeds so the ranking is not over-stated).
"""
import glob
import json
import os
import re
from pathlib import Path
import numpy as np

RES = Path("/home/holiday/lung_ct/results")
OUT = Path("/home/holiday/lung_ct/ssl_study/seg_ranking_3seed.json")

# strategy base tag -> human label for the table
STRATEGIES = {
    "S1_basicunet_plain":      "BasicUNet, supervised",
    "S2_dynunet_plain":        "DynUNet, supervised",
    "S3_segresnet_plain":      "SegResNet, supervised",
    "S4_dynunet_aug":          "DynUNet + augmentation",
    "S5_dynunet_aug_pp":       "DynUNet + aug + post-proc",
    "S6_dynunet_aug_tversky":  "DynUNet + aug + pp + Tversky",
    "S7_dynunet_meanteacher":  "DynUNet + Mean-Teacher (SSL)",
    "S8_attentionunet_aug_pp": "AttentionUNet + aug + pp",
}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # two-sided t 0.975 by dof (n-1)


def collect(base):
    """Return {seed: (test_dice, mtime, path)} for one strategy base tag, latest per seed."""
    by_seed = {}
    for f in glob.glob(str(RES / f"seg_{base}_*.json")):
        # exclude other strategies that share a prefix is unnecessary (tags are unique full)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tag = str(d.get("tag", ""))
        # accept seed-7 base tag (== base) or rerun tags base_s42 / base_s1337
        if tag != base and not re.fullmatch(rf"{re.escape(base)}_s\d+", tag):
            continue
        seed = int(d.get("seed"))
        td = d.get("test_dice", d.get("dice"))
        if td is None:
            continue
        mt = os.path.getmtime(f)
        if seed not in by_seed or mt > by_seed[seed][1]:
            by_seed[seed] = (float(td), mt, f)
    return by_seed


rows = []
warn = []
for base, label in STRATEGIES.items():
    bs = collect(base)
    seeds = sorted(bs)
    vals = np.array([bs[s][0] for s in seeds], float)
    n = len(vals)
    if n == 0:
        warn.append(f"{base}: NO runs found")
        continue
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        half = T975.get(n - 1, 1.96) * sd / np.sqrt(n)
        ci = [round(mean - half, 4), round(mean + half, 4)]
    else:
        ci = None
    if n < 3:
        warn.append(f"{base}: only {n} seed(s) {seeds} -- ranking CI under-powered")
    rows.append({"strategy": base, "label": label, "n_seeds": n, "seeds": seeds,
                 "dice_mean": round(mean, 4), "dice_sd": None if np.isnan(sd) else round(sd, 4),
                 "dice_ci95": ci, "per_seed": {str(s): round(bs[s][0], 4) for s in seeds}})

rows.sort(key=lambda r: r["dice_mean"], reverse=True)
for i, r in enumerate(rows, 1):
    r["rank"] = i

out = {"strategies": rows, "warnings": warn,
       "note": "test Dice, patient-grouped; mean/SD/95% t-CI across seeds; n_seeds per strategy stated."}
OUT.write_text(json.dumps(out, indent=2))

print(f"wrote {OUT}")
if warn:
    print("\n*** WARNINGS (do not finalize ranking until resolved) ***")
    for w in warn:
        print("  -", w)
print("\nRank | Strategy                              | n | Dice mean (95% CI)")
for r in rows:
    ci = f"[{r['dice_ci95'][0]:.3f}, {r['dice_ci95'][1]:.3f}]" if r["dice_ci95"] else "(single seed)"
    print(f"{r['rank']:>4} | {r['label']:<37} | {r['n_seeds']} | {r['dice_mean']:.3f} {ci}")

print("\n% --- paste-ready LaTeX table body (booktabs) ---")
for r in rows:
    ci = f"{r['dice_ci95'][0]:.3f}--{r['dice_ci95'][1]:.3f}" if r["dice_ci95"] else "---"
    sd = f"{r['dice_sd']:.3f}" if r["dice_sd"] is not None else "---"
    print(f"{r['rank']} & {r['label']} & {r['n_seeds']} & {r['dice_mean']:.3f} & {sd} & {ci} \\\\")
