#!/bin/bash
# nnU-Net 3d_fullres ceiling for lung-tumour segmentation (Study 2).
# Patient-grouped 5-fold CV on Dataset001_LungTumor (145 cases / 102 patients).
# Run ONLY when the GPU is free (the seg reruns + distillation must not be competing),
# because nnU-Net 3d_fullres is heavy. Training is checkpointed/resumable, so a re-run
# continues rather than restarting.
#
# Usage:
#   bash scripts/run_nnunet_ceiling.sh            # all 5 folds, full schedule
#   FOLDS="0" bash scripts/run_nnunet_ceiling.sh  # single fold (quick first estimate)
#   TRAINER=nnUNetTrainer_250epochs bash scripts/run_nnunet_ceiling.sh   # faster ceiling
set -e
cd /home/holiday/lung_ct
export nnUNet_raw="/home/holiday/lung_ct/data/nnUNet_raw"
export nnUNet_preprocessed="/home/holiday/lung_ct/data/nnUNet_preprocessed"
export nnUNet_results="/home/holiday/lung_ct/data/nnUNet_results"

FOLDS="${FOLDS:-0 1 2 3 4}"
TRAINER="${TRAINER:-nnUNetTrainer}"

# 1. apply the PATIENT-GROUPED split (overwrites nnU-Net's default random split)
.venv/bin/python scripts/nnunet_patient_split.py

# 2. train each fold (3d_fullres). --c resumes if a checkpoint exists.
for f in $FOLDS; do
  echo "=== training fold $f ($TRAINER) ==="
  .venv/bin/nnUNetv2_train 1 3d_fullres "$f" -tr "$TRAINER" --c || \
  .venv/bin/nnUNetv2_train 1 3d_fullres "$f" -tr "$TRAINER"
done

# 3. aggregate the cross-validation Dice across the trained folds
.venv/bin/python scripts/nnunet_ceiling_dice.py
echo "nnU-Net ceiling run complete."
