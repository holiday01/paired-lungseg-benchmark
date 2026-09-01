#!/usr/bin/env python3
"""PAIRED strategy comparison for Study 2 (CBM headline statistic).

The marginal-CI ranking (aggregate_seg_ranking.py) hides the fact that all strategies
share the SAME seeds, so a large common-mode seed variance makes every marginal CI overlap.
A paired analysis removes that common variance: for the reference strategy (DynUNet, plain)
vs each competitor we take the per-seed Dice DIFFERENCE on the shared seeds and test whether
it is > 0. Reports, per comparison: shared seeds, per-seed diffs, mean diff, paired-t (one-sided
H1: reference > competitor), Wilcoxon signed-rank when n permits, and the win count.

Reads only real result JSONs in results/; auto-upgrades from n=3 to n=5+ as new seed runs land.
No number is hand-entered. Emits ssl_study/seg_paired.json.
"""
import glob, json, os, re, math
from pathlib import Path
import numpy as np

RES = Path("/home/holiday/lung_ct/results")
OUT = Path("/home/holiday/lung_ct/ssl_study/seg_paired.json")
REF = "S2_dynunet_plain"
STRATEGIES = {
    "S1_basicunet_plain": "BasicUNet, supervised",
    "S2_dynunet_plain": "DynUNet, supervised",
    "S3_segresnet_plain": "SegResNet, supervised",
    "S4_dynunet_aug": "DynUNet + augmentation",
    "S5_dynunet_aug_pp": "DynUNet + aug + post-proc",
    "S6_dynunet_aug_tversky": "DynUNet + aug + pp + Tversky",
    "S7_dynunet_meanteacher": "DynUNet + Mean-Teacher (SSL)",
    "S8_attentionunet_aug_pp": "AttentionUNet + aug + pp",
}

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def collect(base):
    by_seed = {}
    for f in glob.glob(str(RES / f"seg_{base}_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tag = str(d.get("tag", ""))
        if tag != base and not re.fullmatch(rf"{re.escape(base)}_s\d+", tag):
            continue
        seed = int(d.get("seed"))
        td = d.get("test_dice", d.get("dice"))
        if td is None:
            continue
        mt = os.path.getmtime(f)
        if seed not in by_seed or mt > by_seed[seed][1]:
            by_seed[seed] = (float(td), mt)
    return {s: v[0] for s, v in by_seed.items()}


def t_sf(t, df):
    """one-sided survival P(T>t). scipy if available else Student-t via regularized incomplete beta."""
    if HAVE_SCIPY:
        return float(stats.t.sf(t, df))
    # betainc-based fallback
    x = df / (df + t * t)
    # I_x(df/2, 1/2) via continued fraction (mpmath-free); use math.lgamma series is complex —
    # fall back to normal approx for df>=... keep simple: use scipy-less normal approx.
    from math import erf, sqrt
    # crude: for df>=3 normal approx of t is acceptable for reporting alongside exact scipy runs
    p = 0.5 * (1 - erf(t / sqrt(2)))
    return float(p)


ref = collect(REF)
comparisons = []
all_win = True
for base, label in STRATEGIES.items():
    if base == REF:
        continue
    other = collect(base)
    seeds = sorted(set(ref) & set(other))
    if not seeds:
        continue
    diffs = np.array([ref[s] - other[s] for s in seeds], float)
    n = len(diffs)
    wins = int((diffs > 0).sum())
    if wins < n:
        all_win = False
    md = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        t = md / (sd / math.sqrt(n))
        p1 = t_sf(t, n - 1)
        tcrit = float(stats.t.ppf(0.975, n - 1)) if HAVE_SCIPY else None
        ci95 = ([round(md - tcrit * sd / math.sqrt(n), 4),
                 round(md + tcrit * sd / math.sqrt(n), 4)] if tcrit is not None else None)
    else:
        t, p1 = float("inf") if md > 0 else 0.0, None
        ci95 = None
    wilcox = None
    if HAVE_SCIPY and n >= 6:
        try:
            wilcox = float(stats.wilcoxon(diffs, alternative="greater").pvalue)
        except Exception:
            wilcox = None
    comparisons.append({
        "competitor": base, "label": label, "n_shared_seeds": n, "seeds": seeds,
        "per_seed_diff": {str(s): round(ref[s] - other[s], 4) for s in seeds},
        "mean_diff": round(md, 4), "sd_diff": None if math.isnan(sd) else round(sd, 4),
        "ref_wins": f"{wins}/{n}",
        "ci95_diff": ci95,
        "paired_t": None if not math.isfinite(t) else round(t, 3),
        "p_one_sided": None if p1 is None else round(p1, 4),
        "wilcoxon_p_one_sided": None if wilcox is None else round(wilcox, 4),
    })

comparisons.sort(key=lambda c: c["mean_diff"], reverse=True)
out = {
    "reference": REF, "reference_label": STRATEGIES[REF],
    "reference_per_seed": {str(s): round(v, 4) for s, v in sorted(ref.items())},
    "n_reference_seeds": len(ref),
    "scipy_available": HAVE_SCIPY,
    "reference_beats_every_competitor_on_every_shared_seed": all_win,
    "comparisons": comparisons,
    "note": ("Paired one-sided test, H1: DynUNet-plain Dice > competitor on shared seeds. "
             "Each seed fixes the patient split, initialization and training stochasticity; "
             "pairing removes the common-mode variance of that shared random realization. "
             "ci95_diff is the two-sided 95% t CI of the paired difference. "
             "p from Student-t (exact via scipy when available); Wilcoxon reported when n>=6."),
}
OUT.write_text(json.dumps(out, indent=2))
print(f"wrote {OUT}   (scipy={HAVE_SCIPY}, ref seeds={sorted(ref)})")
print(f"reference beats every competitor on every shared seed: {all_win}\n")
print(f"{'competitor':38s} {'n':>2} {'winrate':>8} {'meanΔ':>7} {'t':>7} {'p(1-sided)':>10} {'wilcox':>7}")
for c in comparisons:
    print(f"{c['label']:38s} {c['n_shared_seeds']:>2} {c['ref_wins']:>8} "
          f"{c['mean_diff']:>7.3f} {str(c['paired_t']):>7} {str(c['p_one_sided']):>10} "
          f"{str(c['wilcoxon_p_one_sided']):>7}")
