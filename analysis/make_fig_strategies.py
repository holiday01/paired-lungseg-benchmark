#!/usr/bin/env python3
"""Study-2 Figure 1 (fig_strategies_tradeoff.pdf).
A: eight-strategy test-Dice ranking with marginal 95% CIs (they overlap).
B: paired per-seed differences, DynUNet-plain minus each competitor -- all positive on every seed,
   the separable ordering the marginal view hides.
Reads seg_ranking_3seed.json + seg_paired.json (auto-updates to n=5). Provenance guard included.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/holiday/lung_ct")
RANK = json.load(open(ROOT / "ssl_study/seg_ranking_3seed.json"))
PAIR = json.load(open(ROOT / "ssl_study/seg_paired.json"))
FIG = ROOT / "manuscripts/study2_segmentation/figures/fig_strategies_tradeoff.pdf"

rows = RANK["strategies"]
# guard: DynUNet-plain is rank 1 and wins every paired comparison in the mean
top = rows[0]
assert top["strategy"] == "S2_dynunet_plain", "DynUNet-plain not rank 1"
assert all(c["mean_diff"] > 0 for c in PAIR["comparisons"]), "a paired mean diff went non-positive"

# JCAT artwork rule: no figure may exceed 7 in in width or length, and the short axis
# must be >= 4 in. The two panels carry long strategy labels, so they are stacked.
fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.0, 6.9))

# Panel A: ranking with CIs
labels = [r["label"] for r in rows][::-1]
means = [r["dice_mean"] for r in rows][::-1]
cis = [r["dice_ci95"] for r in rows][::-1]
y = np.arange(len(rows))
lo = [m - c[0] for m, c in zip(means, cis)]
hi = [c[1] - m for m, c in zip(means, cis)]
colors = ["#2c7fb8" if r["strategy"] == "S2_dynunet_plain" else "0.55" for r in rows][::-1]
axA.errorbar(means, y, xerr=[lo, hi], fmt="o", capsize=3, ecolor="0.7",
             markerfacecolor="none", mec="none")
for yi, m, c in zip(y, means, colors):
    axA.plot(m, yi, "o", color=c, ms=7, zorder=3)
axA.set_yticks(y)
axA.set_yticklabels(labels, fontsize=8)
axA.set_xlabel("Test Dice (mean $\\pm$ 95% CI)")
axA.set_title("A  Strategy ranking with marginal 95% CIs", fontsize=10, loc="left")
axA.set_xlim(0, 1)
axA.grid(axis="x", alpha=0.3)

# Panel B: paired differences
comps = PAIR["comparisons"]
comps = sorted(comps, key=lambda c: c["mean_diff"])
yB = np.arange(len(comps))
for yi, c in zip(yB, comps):
    diffs = list(c["per_seed_diff"].values())
    axB.plot(diffs, [yi] * len(diffs), "o", color="#d95f0e", ms=5, alpha=0.6, zorder=2)
    axB.plot(c["mean_diff"], yi, "D", color="#7a2f00", ms=8, zorder=3)
axB.axvline(0, color="k", lw=1)
axB.set_yticks(yB)
axB.set_yticklabels([c["label"] for c in comps], fontsize=8)
axB.set_xlabel("Paired $\\Delta$Dice: DynUNet-plain $-$ competitor")
axB.set_title("B  Paired per-seed Dice differences", fontsize=10, loc="left")
axB.grid(axis="x", alpha=0.3)

plt.tight_layout()
FIG.parent.mkdir(exist_ok=True)
fig.tight_layout()
plt.savefig(FIG)
plt.savefig(str(FIG).replace(".pdf", ".png"), dpi=600)
print(f"wrote {FIG}  (n_ref_seeds={PAIR['n_reference_seeds']}, guard PASS)")
