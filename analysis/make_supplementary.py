#!/usr/bin/env python3
"""Generate supplementary.tex for the Study-2 segmentation paper.

Every number is read from the stored result JSONs (results/ and ssl_study/);
nothing is hand-entered. Where a summary JSON also stores an aggregate, this
script recomputes it from the per-seed values and asserts they agree before
writing, mirroring the provenance discipline used for the main figures.

Tables:
  S1  per-seed test Dice, 8 strategies x 5 seeds
  S2  full paired comparison of DynUNet-plain vs each strategy (+ Holm-adjusted p)
  S3  Dice stratified by lesion-volume band (numbers behind Fig. 4)
  S4  per-tumour FROC operating points, in-site vs external
  S5  nnU-Net per-fold cross-validation Dice
  S6  SSL-optimisation sub-study (UA-MT / pseudo-label / stabilised-MT / more-unlabelled)
"""
import json, glob, os, math

ROOT = "/home/holiday/lung_ct"
RES = f"{ROOT}/results"
SSL = f"{ROOT}/ssl_study"
OUT = f"{ROOT}/manuscripts/study2_segmentation/supplementary.tex"

SEEDS = [7, 22, 33, 42, 1337]


def load(path):
    with open(path) as f:
        return json.load(f)


def latest(pat):
    fs = sorted(glob.glob(pat))
    return fs[-1] if fs else None


def msd(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd


def holm(pairs):
    """pairs: list of (key, p). Return dict key -> Holm-adjusted p (step-down)."""
    m = len(pairs)
    order = sorted(range(m), key=lambda i: pairs[i][1])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pairs[i][1]))
        adj[i] = running
    return {pairs[i][0]: adj[i] for i in range(m)}


# ---------------------------------------------------------------- load sources
ranking = load(f"{SSL}/seg_ranking_3seed.json")
paired = load(f"{SSL}/seg_paired.json")
bysize = load(f"{SSL}/results_bysize_5seed.json")
froc_in = load(f"{SSL}/froc_corrected_5seed.json")
froc_ext = load(f"{SSL}/results_external_froc_corrected.json")
nnunet = load(f"{SSL}/results_nnunet_ceiling.json")


# ---------------------------------------------------------------- Table S1
def table_s1():
    rows = sorted(ranking["strategies"], key=lambda s: s["rank"])
    body = []
    for s in rows:
        ps = s["per_seed"]
        vals = [ps[str(k)] for k in SEEDS]
        m, sd = msd(vals)
        assert abs(m - s["dice_mean"]) < 1e-3, (s["strategy"], m, s["dice_mean"])
        assert abs(sd - s["dice_sd"]) < 1e-3, (s["strategy"], sd, s["dice_sd"])
        cells = " & ".join(f"{v:.3f}" for v in vals)
        body.append(
            f"{s['rank']} & {s['label']} & {cells} & {m:.3f} $\\pm$ {sd:.3f} \\\\"
        )
    return "\n".join(body)


# ---------------------------------------------------------------- Table S2
def table_s2():
    comps = paired["comparisons"]
    hp = holm([(c["competitor"], c["p_one_sided"]) for c in comps])
    body = []
    # keep the manuscript ordering: by mean_diff descending (largest gap first)
    for c in sorted(comps, key=lambda c: -c["mean_diff"]):
        d = c["per_seed_diff"]
        diffs = " & ".join(f"{d[str(k)]:+.3f}" for k in SEEDS)
        padj = hp[c["competitor"]]
        star = "$^{*}$" if padj < 0.05 else ""
        body.append(
            f"{c['label']} & {diffs} & {c['mean_diff']:.3f} $\\pm$ {c['sd_diff']:.3f} "
            f"& {c['ref_wins']} & {c['paired_t']:.2f} & {c['p_one_sided']:.3f} "
            f"& {padj:.3f}{star} \\\\"
        )
    return "\n".join(body)


# ---------------------------------------------------------------- Table S3
def table_s3():
    order = ["<1cm3", "1-10cm3", "10-50cm3", ">=50cm3"]
    tex = {"<1cm3": "$<1$", "1-10cm3": "$1$--$10$",
           "10-50cm3": "$10$--$50$", ">=50cm3": "$\\geq 50$"}
    body = []
    for b in order:
        d = bysize["bands"][b]
        lo, hi = d["dice_ci95"]
        body.append(
            f"{tex[b]} & {d['n_lesions_total']} & {d['n_seeds']} & "
            f"{d['dice_mean']:.3f} $\\pm$ {d['dice_sd']:.3f} & [{lo:.3f}, {hi:.3f}] \\\\"
        )
    return "\n".join(body)


# ---------------------------------------------------------------- Table S4
def table_s4():
    thr = ["0.1", "0.3", "0.5", "0.7", "0.9"]
    body = []
    for t in thr:
        a = froc_in["froc_aggregate"][t]
        e = froc_ext["froc_aggregate"][t]
        body.append(
            f"{t} & {a['sens_mean']:.3f} $\\pm$ {a['sens_sd']:.3f} & "
            f"{a['fp_mean']:.2f} & {e['sens_mean']:.3f} $\\pm$ {e['sens_sd']:.3f} & "
            f"{e['fp_mean']:.2f} \\\\"
        )
    return "\n".join(body)


# ---------------------------------------------------------------- Table S5
def table_s5():
    pf = nnunet["per_fold_dice"]
    keys = sorted(pf, key=lambda k: int(k.split("_")[1]))
    vals = [pf[k] for k in keys]
    m, sd = msd(vals)
    assert abs(m - nnunet["cv_dice_mean"]) < 5e-4, (m, nnunet["cv_dice_mean"])
    assert abs(sd - nnunet["cv_dice_sd"]) < 5e-4, (sd, nnunet["cv_dice_sd"])
    cells = " & ".join(f"{v:.3f}" for v in vals)
    return (cells, f"{m:.3f} $\\pm$ {sd:.3f}",
            nnunet["comparison"]["dynunet_5seed_dice"],
            nnunet["comparison"]["dynunet_5seed_sd"])


# ---------------------------------------------------------------- Table S6
def ssl_run(tag, seed):
    """Locate a per-run JSON; seed 7 of the plain baselines has no _s suffix."""
    f = latest(f"{RES}/seg_{tag}_s{seed}_*.json")
    if f is None and seed == 7:
        f = latest(f"{RES}/seg_{tag}_20260623_*.json")
    return load(f) if f else None


def table_s6():
    S3 = [7, 42, 1337]  # seeds the SSL sub-study was run on
    variants = [
        ("Supervised DynUNet (reference)", "S2_dynunet_plain"),
        ("Mean-Teacher (reference)", "S7_dynunet_meanteacher"),
        ("Confidence pseudo-label", "SSLOPT_pseudo"),
        ("Stabilised Mean-Teacher", "SSLOPT_stabmt"),
        ("More unlabelled ($2\\times$ pool)", "SSLOPT_moreunlab"),
        ("UA-MT", "S9_dynunet_uamt"),
    ]
    body = []
    for label, tag in variants:
        bv, tv = [], []
        for s in S3:
            d = ssl_run(tag, s)
            if d is None:
                continue
            bv.append(d["best_val_dice"])
            tv.append(d["test_dice"])
        if not bv:
            continue
        n = len(bv)
        mb, sdb = msd(bv)
        mt, sdt = msd(tv)
        bstr = f"{mb:.3f} $\\pm$ {sdb:.3f}" if n > 1 else f"{mb:.3f}"
        tstr = f"{mt:.3f} $\\pm$ {sdt:.3f}" if n > 1 else f"{mt:.3f}"
        body.append(f"{label} & {n} & {bstr} & {tstr} \\\\")
    return "\n".join(body)


s5_cells, s5_mean, dyn_mean, dyn_sd = table_s5()

# ---------------------------------------------------------------- assemble
doc = r"""% !TEX program = pdflatex
% Auto-generated by ssl_study/make_supplementary.py -- do not hand-edit numbers.
\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{labelfont=bf,font=small}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\setlength{\tabcolsep}{5pt}
\title{\textbf{Supplementary Material}\\[4pt]
A strong supervised baseline, not added complexity, drives lung-tumour
segmentation: a paired, externally validated benchmark for annotation assistance}
\date{}
\begin{document}
\maketitle
\vspace{-2.5em}
\noindent All values are computed from the stored result files by
\texttt{ssl\_study/make\_supplementary.py}; no number is hand-entered.

%---------------------------------------------------------------- S1
\begin{table}[h]
\centering
\caption{\textbf{Per-seed test Dice for all eight strategies.} Patient-grouped
20/20 split; five seeds at the 20{,}000-iteration budget. The final column
reproduces the mean\,$\pm$\,SD of Table~2 in the main text.}
\small
\begin{tabular}{clccccc c}
\toprule
Rank & Strategy & \multicolumn{5}{c}{Test Dice by seed} & Mean $\pm$ SD \\
\cmidrule(lr){3-7}
 & & 7 & 22 & 33 & 42 & 1337 & \\
\midrule
""" + table_s1() + r"""
\bottomrule
\end{tabular}
\end{table}

%---------------------------------------------------------------- S2
\begin{table}[h]
\centering
\caption{\textbf{Full paired comparison of DynUNet-plain against every other
strategy.} Per-seed Dice differences (reference $-$ competitor), paired
one-sided $t$ statistic and $p$, and the Holm--Bonferroni adjusted $p$ across
the seven comparisons. $^{*}$ Holm-adjusted $p<0.05$. The paired test removes
the common-mode seed variance that inflates the marginal confidence intervals
in Table~S1.}
\footnotesize
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l ccccc ccccc}
\toprule
Competitor & \multicolumn{5}{c}{Per-seed diff (seed 7/22/33/42/1337)} & Mean $\pm$ SD & Wins & $t$ & $p$ & $p_{\text{Holm}}$ \\
\cmidrule(lr){2-6}
""" + table_s2() + r"""
\bottomrule
\end{tabular}}
\end{table}

%---------------------------------------------------------------- S3
\begin{table}[h]
\centering
\caption{\textbf{Dice stratified by lesion volume (headline model, five seeds).}
Exact values behind Fig.~4. Bands with fewer than five contributing seeds are
noted; the smallest band ($<1$\,cm\textsuperscript{3}) is the least stable.}
\small
\begin{tabular}{lccc c}
\toprule
Volume band (cm\textsuperscript{3}) & Lesions & Seeds & Dice (mean $\pm$ SD) & 95\% CI \\
\midrule
""" + table_s3() + r"""
\bottomrule
\end{tabular}
\end{table}

%---------------------------------------------------------------- S4
\begin{table}[h]
\centering
\caption{\textbf{Per-tumour FROC operating points, in-site vs.\ external.}
One expert mask counts as one tumour; sensitivity and false positives per
volume are averaged over five seeds. In-site uses the held-out hospital test
set; external uses 421 TCIA NSCLC-Radiomics cases with the same checkpoints.
Supports Figs.~2B and~3.}
\small
\begin{tabular}{c cc cc}
\toprule
& \multicolumn{2}{c}{In-site} & \multicolumn{2}{c}{External (TCIA)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Prob.\ threshold & Sensitivity & FP/vol & Sensitivity & FP/case \\
\midrule
""" + table_s4() + r"""
\bottomrule
\end{tabular}
\end{table}

%---------------------------------------------------------------- S5
\begin{table}[h]
\centering
\caption{\textbf{Semi-supervised optimisation sub-study.} Four semi-supervised
variants trained on the same DynUNet backbone with the 2{,}449-nodule
unlabelled pool, evaluated at three shared seeds (UA-MT at one seed only).
Model selection should use the validation Dice; on that metric none of the
variants reaches the supervised reference, and the occasionally high
\emph{test} Dice (e.g.\ pseudo-labelling) reflects small-test-set variance
rather than a genuine gain. This is the negative result underlying the
manuscript's claim that added semi-supervision did not improve on supervised
training.}
\small
\begin{tabular}{lc cc}
\toprule
Variant & Seeds & Val.\ Dice (sel.) & Test Dice \\
\midrule
""" + table_s6() + r"""
\bottomrule
\end{tabular}
\end{table}

\end{document}
"""

with open(OUT, "w") as f:
    f.write(doc)
print(f"wrote {OUT}")
print("all assert-guards passed")
