# paired-lungseg-benchmark

Code and pipeline for the paper:

> **A strong supervised baseline, not added complexity, drives lung-tumour segmentation:
> a paired, externally validated benchmark for annotation assistance.**

A benchmark that puts several lung-CT tumour-segmentation models on one footing and compares them
fairly. It spans four architectures (BasicUNet, DynUNet, SegResNet, AttentionUNet) combined with
augmentation, post-processing, a Focal–Tversky loss, and Mean-Teacher semi-supervision, plus a
self-configuring nnU-Net as an independent reference. All strategies are evaluated under one
patient-grouped, multi-seed protocol with a **paired** analysis, per-tumour lesion-level detection,
and external validation on TCIA NSCLC-Radiomics.

## Run it on your own data

You bring a folder of CT images, a folder of matching tumour masks, and a CSV that lists which
image goes with which mask. The pipeline preprocesses them and runs the benchmark; no editing of
source paths is needed.

```
data/
  img/            your CT images        (.nii / .nii.gz / .nrrd)
  msk/            matching tumour masks (same file names as the images)
  manifest.csv    which image pairs with which mask
```

`data/manifest.csv` needs a header and one row per tumour volume:

```csv
image,mask,case
patientA_B00.nii.gz,patientA_B00.nii.gz,patientA
patientA_B01.nii.gz,patientA_B01.nii.gz,patientA
patientB.nii.gz,patientB.nii.gz,patientB
```

- `image`, `mask` — file names inside `--img-dir` / `--msk-dir`.
- `case` — optional patient id. All rows with the same `case` are kept on the same side of every
  train/val/test split, so multiple tumours from one patient never leak across the split. If the
  column is omitted, the case id is the image name with a trailing `_B<number>` stripped.

Then run the three steps:

```bash
# 1. preprocess: CSV + data/img + data/msk -> per-volume .npz (1 mm iso, HU [-1000, 400], bbox crop)
python scripts/prepare_data.py \
    --manifest data/manifest.csv --img-dir data/img --msk-dir data/msk \
    --out data/processed_seg --spacing 1.0

# 2. train one strategy/seed -> one result JSON per run (repeat over strategies and seeds)
python scripts/train_seg.py --seg-dir data/processed_seg --tag S2_dynunet_plain --seed 7

# 3. aggregate + figures (assert-guarded: recomputed values must match the stored run outputs)
python analysis/aggregate_seg_ranking.py       # -> strategy ranking (Table 2)
python analysis/aggregate_seg_paired.py         # -> paired comparison (Table 3)
python analysis/make_fig_strategies.py          # -> Fig. 1
python analysis/make_supplementary.py           # -> Supplementary Tables S1–S5
```

## What each script produces (paper artifact)

| Script | Produces | Paper artifact |
|---|---|---|
| `scripts/prepare_data.py` | CSV + `data/img` + `data/msk` → preprocessed `.npz` volumes | the input to every training run |
| `scripts/train_seg.py` | trains one strategy (writes `results/seg_<tag>_<seed>.json` + checkpoint) | Table 1 hyperparameters; the 8-strategy runs; the 40k headline model; external checkpoints |
| `scripts/train.py` | shared training utilities (seeding, cuDNN config) imported by `train_seg.py` | — |
| `scripts/nnunet_patient_split.py`, `convert_to_nnunet.py`, `run_nnunet_ceiling.sh`, `nnunet_ceiling_dice.py` | patient-grouped 5-fold nnU-Net reference | `results_nnunet_ceiling.json` (0.658 ± 0.036; Discussion) |
| `analysis/aggregate_seg_ranking.py` | strategy ranking (mean/SD/95% CI across seeds) | `seg_ranking_3seed.json` → **Table 2** |
| `analysis/aggregate_seg_paired.py` | paired across-seed test + Holm/BH | `seg_paired.json` → **Table 3** |
| `analysis/aggregate_seg_bysize_headline.py` | Dice stratified by lesion volume | `results_bysize_5seed.json` → **Fig. 4** |
| `analysis/froc_corrected.py` | in-site per-tumour detection (one mask = one tumour) | `froc_corrected_5seed.json` (0.953; **Fig. 3**) |
| `analysis/froc_external_corrected.py` | external per-tumour detection, same definition | `results_external_froc_corrected.json` (0.788; **Fig. 2B**) |
| `analysis/froc_analysis.py` | naive connected-component FROC (kept for contrast) | the 0.385 artifact discussed in Results |
| `analysis/preprocess_external_nsclc.py`, `eval_external_nsclc.py` | external Dice + naive FROC on 421 TCIA cases | `results_external_nsclc.json` (Dice 0.449; **Fig. 2A**) |
| `analysis/make_fig_strategies.py`, `make_fig_external.py`, `make_fig_bysize_froc.py`, `make_supplementary.py` | the figures and supplementary tables (each assert-guarded against the stored numbers) | Figs. 1–4, Tables S1–S5 |

## Provenance
Every number in the paper is computed by these scripts from per-run result files written by the
training code; no value is hand-entered, and the figure/table scripts assert that recomputed values
reproduce the stored numbers before saving. **This repository provides the code and pipeline only.**
The derived result files and the raw imaging data are not distributed (see Data availability).

## Data availability
- **Single-site cohort** (145 tumour masks / 102 patients): not public, owing to institutional
  data-sharing restrictions (IRB114-168-B, Lotung Pohai Hospital). Derived result files are held by
  the authors and available on reasonable request; they are not distributed in this repository.
- **External cohort**: TCIA **NSCLC-Radiomics** ("Lung1"), publicly available
  (doi:10.7937/K9/TCIA.2015.PF0M9REI; Aerts et al. 2014, Nat. Commun.; Clark et al. 2013,
  J. Digit. Imaging). The external scripts run on this public collection after preprocessing.

## Environment
Python 3.12; key package versions in `requirements.txt` (PyTorch 2.11 / CUDA 12.8, MONAI 1.5.2,
nnU-Net v2.7.0, SimpleITK). A CUDA GPU is required for training and inference.

## License
MIT (see `LICENSE`). The external cohort is TCIA NSCLC-Radiomics under its own CC BY-NC license.
