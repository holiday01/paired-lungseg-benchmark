#!/usr/bin/env python3
"""Regenerate two Study-2 figures from the FINAL 5-seed data (review fix 2026-07-13):
 - fig3_by_size.pdf: Dice by lesion-volume band, 5-seed mean +/- 95% CI (replaces a stale single-run fig).
 - fig5_froc.pdf: corrected per-tumour FROC (one expert mask = one tumour) vs the naive
   connected-component number, showing detection hinges on the lesion definition.
Provenance guards assert the stored means before saving.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/holiday/lung_ct")
FIGD = ROOT/"manuscripts/study2_segmentation/figures"
BY = json.load(open(ROOT/"ssl_study/results_bysize_5seed.json"))
FR = json.load(open(ROOT/"ssl_study/froc_corrected_5seed.json"))

# ---------- Figure: Dice by size band ----------
order = ["<1cm3", "1-10cm3", "10-50cm3", ">=50cm3"]
labels = ["$<$1", "1--10", "10--50", "$\\geq$50"]
means = [BY["bands"][b]["dice_mean"] for b in order]
cis = [BY["bands"][b]["dice_ci95"] for b in order]
ns = [BY["bands"][b]["n_lesions_total"] for b in order]
nseed = [BY["bands"][b]["n_seeds"] for b in order]
assert abs(means[0]-0.560) < 0.002 and abs(means[2]-0.7775) < 0.002, "by-size means moved"
lo = [m-c[0] for m, c in zip(means, cis)]
hi = [c[1]-m for m, c in zip(means, cis)]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
x = np.arange(4)
ax.bar(x, means, color="#4c9fd0", width=0.62, zorder=2)
ax.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor="0.25", capsize=4, lw=1.3, zorder=3)
for i, (m, n, s) in enumerate(zip(means, ns, nseed)):
    ax.text(i, 0.03, f"n={n}" + ("" if s == 5 else f"\n({s}/5 seeds)"),
            ha="center", va="bottom", fontsize=7.5, color="white")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Lesion volume band (cm$^3$)")
ax.set_ylabel("Test Dice (5-seed mean $\\pm$ 95% CI)")
ax.set_ylim(0, 1)
ax.set_title("Dice by lesion volume", fontsize=10)
plt.tight_layout()
plt.savefig(FIGD/"fig3_by_size.pdf", bbox_inches="tight")
plt.savefig(FIGD/"fig3_by_size.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Figure: corrected FROC ----------
agg = FR["froc_aggregate"]
thr = sorted(agg, key=float)
fp = [agg[t]["fp_mean"] for t in thr]
sens = [agg[t]["sens_mean"] for t in thr]
sd = [agg[t]["sens_sd"] for t in thr]
assert abs(agg["0.5"]["sens_mean"]-0.953) < 0.005, "corrected FROC sens moved"
fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.errorbar(fp, sens, yerr=sd, fmt="o-", color="#2c7fb8", capsize=3, ms=4,
            label="per-tumour (one expert mask = one tumour)")
ax.axhline(1.0, ls="--", color="0.5", lw=1, label="case-level detection $\\approx$1.0")
ax.plot(4.3, 0.385, "s", color="#d62728", ms=9, zorder=4,
        label="naive connected-component (0.385)")
ax.annotate("lesion-definition artifact\n(resampling fragments)", xy=(4.3, 0.385),
            xytext=(6.5, 0.45), fontsize=7.5, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
ax.set_xlabel("False positives per volume")
ax.set_ylabel("Detection sensitivity")
ax.set_ylim(0, 1.05); ax.set_xlim(0, 12)
ax.legend(fontsize=7.5, loc="lower right")
ax.set_title("Detection depends on the lesion definition", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGD/"fig5_froc.pdf", bbox_inches="tight")
plt.savefig(FIGD/"fig5_froc.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote fig3_by_size.pdf and fig5_froc.pdf (guards passed)")
print(f"by-size means: {[round(m,3) for m in means]}")
print(f"corrected FROC sens@0.5: {agg['0.5']['sens_mean']:.3f}")
