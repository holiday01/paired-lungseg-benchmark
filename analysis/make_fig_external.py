#!/usr/bin/env python3
"""Study-2 external-validation figure (fig_external.pdf).
Left: paired per-seed test Dice, in-site headline DynUNet vs external TCIA NSCLC-Radiomics
(same 5 seeds, identical checkpoints). Right: external lesion-level FROC curve.
Provenance guard: asserts recomputed means reproduce the stored result-JSON values before saving.
"""
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/holiday/lung_ct")
EXT = json.load(open(ROOT / "ssl_study/results_external_nsclc.json"))          # per-seed Dice
EXTF = json.load(open(ROOT / "ssl_study/results_external_froc_corrected.json"))  # per-tumor FROC
INF = json.load(open(ROOT / "ssl_study/froc_corrected_5seed.json"))             # in-site per-tumor FROC
FIG = ROOT / "manuscripts/study2_segmentation/figures/fig_external.pdf"

# in-site headline per-seed Dice
insite = {}
for f in glob.glob(str(ROOT / "results/seg_best_dynunet_s*.json")):
    d = json.load(open(f))
    insite[int(d["seed"])] = float(d["test_dice"])
ext = {int(k): v["dice_mean"] for k, v in EXT["per_seed"].items()}
seeds = sorted(set(insite) & set(ext))
in_v = np.array([insite[s] for s in seeds])
ex_v = np.array([ext[s] for s in seeds])

# ---- provenance guard ----
assert abs(in_v.mean() - 0.659) < 0.002, f"in-site mean {in_v.mean():.4f} != 0.659"
assert abs(ex_v.mean() - EXT["dice_mean_across_seeds"]) < 1e-6, "external mean mismatch"
assert abs(EXT["dice_mean_across_seeds"] - 0.449) < 0.002, "external mean != 0.449"
assert len(seeds) == 5, f"expected 5 shared seeds, got {seeds}"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 4.3))

# Left: paired dot plot
for i, s in enumerate(seeds):
    ax1.plot([0, 1], [in_v[i], ex_v[i]], "-", color="0.7", lw=1, zorder=1)
ax1.scatter(np.zeros(len(seeds)), in_v, s=45, color="#2c7fb8", zorder=3, label="in-site")
ax1.scatter(np.ones(len(seeds)), ex_v, s=45, color="#d95f0e", zorder=3, label="external")
ax1.errorbar(0, in_v.mean(), yerr=in_v.std(ddof=1), fmt="_", color="#2c7fb8", capsize=6, ms=22, mew=2)
ax1.errorbar(1, ex_v.mean(), yerr=ex_v.std(ddof=1), fmt="_", color="#d95f0e", capsize=6, ms=22, mew=2)
ax1.set_xlim(-0.4, 1.4)
ax1.set_xticks([0, 1])
ax1.set_xticklabels([f"In-site\n0.659$\\pm$0.028", f"External\n0.449$\\pm$0.014"])
ax1.set_ylabel("Test Dice")
ax1.set_ylim(0.35, 0.75)
ax1.set_title("A  In-site versus external test Dice", fontsize=10, loc="left")

# Right: per-tumor FROC, in-site vs external (SAME lesion definition)
def curve(agg):
    thr = sorted(agg, key=float)
    return ([agg[t]["fp_mean"] for t in thr],
            [agg[t]["sens_mean"] for t in thr],
            [agg[t]["sens_sd"] or 0 for t in thr])
efp, esens, esd = curve(EXTF["froc_aggregate"])
ifp, isens, isd = curve(INF["froc_aggregate"])
# guard: per-tumor operating points reproduce the stored values
assert abs(EXTF["froc_aggregate"]["0.5"]["sens_mean"]-0.788) < 0.005, "ext per-tumor sens moved"
assert abs(INF["froc_aggregate"]["0.5"]["sens_mean"]-0.953) < 0.005, "in-site per-tumor sens moved"
ax2.errorbar(ifp, isens, yerr=isd, fmt="o-", color="#2c7fb8", capsize=3, ms=4, label="in-site")
ax2.errorbar(efp, esens, yerr=esd, fmt="s-", color="#d95f0e", capsize=3, ms=4, label="external")
ax2.set_xlabel("False positives per cropped volume")
ax2.set_ylabel("Per-tumor detection sensitivity")
ax2.set_ylim(0, 1); ax2.set_xlim(left=0)
ax2.legend(fontsize=8, loc="lower right")
ax2.set_title("B  Per-tumor detection, in-site versus external", fontsize=9, loc="left")
ax2.grid(alpha=0.3)

plt.tight_layout()
FIG.parent.mkdir(exist_ok=True)
fig.tight_layout()
plt.savefig(FIG)
plt.savefig(str(FIG).replace(".pdf", ".png"), dpi=600)
print(f"wrote {FIG}  (in-site {in_v.mean():.3f}, external {ex_v.mean():.3f}, guard PASS)")
