# Lung-tumour segmentation benchmark — code and results

Code and result files for the paper:

> **A strong supervised baseline, not added complexity, drives lung-tumour segmentation:
> a paired, externally validated benchmark for annotation assistance.**

A benchmark of eight lung-CT tumour-segmentation strategies (architecture / augmentation / loss /
semi-supervision) evaluated with a paired multi-seed analysis, per-tumour lesion-level detection,
and external validation on TCIA NSCLC-Radiomics.

## What each script produces (paper artifact)

| Script | Produces | Paper artifact |
|---|---|---|
| `scripts/train_seg.py` | trains one segmentation strategy (writes `results/seg_<tag>_<seed>.json` + checkpoint) | Table 1 hyperparameters; the 8-strategy runs; the 40k headline model; external checkpoints |
| `scripts/train.py` | shared training utilities (seeding, cuDNN config) imported by `train_seg.py` | — |
| `scripts/nnunet_patient_split.py`, `convert_to_nnunet.py`, `run_nnunet_ceiling.sh`, `nnunet_ceiling_dice.py` | patient-grouped 5-fold nnU-Net ceiling | `results/summary/results_nnunet_ceiling.json` (0.658 ± 0.036; Discussion) |
| `analysis/aggregate_seg_ranking.py` | strategy ranking (mean/SD/95% CI across seeds) | `seg_ranking_3seed.json` → **Table 2** |
| `analysis/aggregate_seg_paired.py` | paired across-seed test + Holm/BH | `seg_paired.json` → **Table 3** |
| `analysis/aggregate_seg_bysize_headline.py` | Dice stratified by lesion volume | `results_bysize_5seed.json` → **Fig. 4** |
| `analysis/froc_corrected.py` | in-site per-tumour detection (one mask = one tumour) | `froc_corrected_5seed.json` (0.953; **Fig. 3**, Results) |
| `analysis/froc_external_corrected.py` | external per-tumour detection, same definition | `results_external_froc_corrected.json` (0.788; **Fig. 2B**) |
| `analysis/froc_analysis.py` | naive connected-component FROC (kept for contrast) | the 0.385 artifact discussed in Results |
| `analysis/preprocess_external_nsclc.py`, `eval_external_nsclc.py` | external Dice + naive FROC on 421 TCIA cases | `results_external_nsclc.json` (Dice 0.449; **Fig. 2A**) |
| `analysis/make_fig_strategies.py`, `make_fig_external.py`, `make_fig_bysize_froc.py` | the figures (each with an assert-guard reproducing the stored numbers) | Figs. 1–4 |

## Provenance
Every number in the paper is computed by these scripts from the result JSONs; no value is
hand-entered. The figure scripts assert that recomputed values reproduce the stored JSON numbers
before saving. The result JSONs are included under `results/` so the tables and figures can be
regenerated **without** the raw imaging data:

```bash
python analysis/aggregate_seg_ranking.py      # -> Table 2
python analysis/aggregate_seg_paired.py        # -> Table 3
python analysis/make_fig_strategies.py         # -> Fig. 1
```

## Data availability
- **Single-site cohort** (145 tumour masks / 102 patients): not public, owing to institutional
  data-sharing restrictions (IRB114-168-B, Lotung Pohai Hospital). Derived, de-identified result
  JSONs are included here.
- **External cohort**: TCIA **NSCLC-Radiomics** ("Lung1"), publicly available
  (doi:10.7937/K9/TCIA.2015.PF0M9REI; Aerts et al. 2014, Nat. Commun.; Clark et al. 2013,
  J. Digit. Imaging). The external scripts run on this public collection after preprocessing.

## Environment
Python 3.12; key package versions in `requirements.txt` (PyTorch 2.11 / CUDA 12.8, MONAI 1.5.2,
nnU-Net v2.7.0). A CUDA GPU is required for training and inference.

## License
MIT (see `LICENSE`). The external cohort is TCIA NSCLC-Radiomics under its own CC BY-NC license.

## Note on paths
The scripts were run from a fixed layout with root `/home/holiday/lung_ct` (paths appear near the
top of each file and in `sys.path.insert(...)`). Adjust the root/`SEG_DIR`/`EXT_DIR` constants to
your environment before running. The code is released for transparency and reuse; turn-key
reproduction of the in-site numbers additionally requires the non-public cohort.
