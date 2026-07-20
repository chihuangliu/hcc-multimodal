#!/bin/bash
# Complete the bbox + frozen + n=all contrastive family. The ablation already has
# one bbox frozen n=all model — 92b9afed (λ=0.1, slice split). This submits the
# three remaining λ × split-unit combinations so the bbox frozen group matches the
# raw frozen Group 3 (dc7e1d10 / 5e3f71a0 / a64b245f / 06c598c0):
#
#   1. bbox, λ=0.0, frozen, n=all, slice
#   2. bbox, λ=0.0, frozen, n=all, patient
#   3. bbox, λ=0.1, frozen, n=all, patient
#
# Every hyperparameter matches 92b9afed (mri_type=raw_bbox, gene_set=all,
# freeze_backbone, n_per_axis=all, axes=0, bbox_pad=10, bs=32, lr=1e-4, seed=42),
# except epochs=10 to match the raw frozen Group 3 models (dc7e1d10 / a64b245f /
# 5e3f71a0 / 06c598c0) rather than 92b9afed's 5. Only --lam and --split-unit vary.
# Each run mints its own random run_id and writes to training/contrastive/<run_id>/.
# Submits one PBS job per config.
#
# Usage:  bash scripts/submit_bbox_frozen_train.sh
set -euo pipefail

PROJECT_DIR=/rds/general/user/cl3225/home/hcc-multimodal

# tag:lam:split_unit for the three configs to train.
CONFIGS=(
  "bbox_l0_slice:0.0:slice"
  "bbox_l0_patient:0.0:patient"
  "bbox_l1_patient:0.1:patient"
)

for CFG in "${CONFIGS[@]}"; do
  TAG="${CFG%%:*}"
  REST="${CFG#*:}"
  LAM="${REST%%:*}"
  SPLIT_UNIT="${REST##*:}"
  echo "Submitting bbox-frozen training: ${TAG} (lam=${LAM}, split=${SPLIT_UNIT})"
  qsub -N "${TAG}" \
    -v LAM="${LAM}",SPLIT_UNIT="${SPLIT_UNIT}",TAG="${TAG}",PROJECT_DIR="${PROJECT_DIR}" <<'PBS'
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -q v1_gpu72

cd "${PROJECT_DIR}"

# --- logging: timestamped log/err files (PBS -o/-e can't expand variables) ---
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
JOBID=${PBS_JOBID%%.*}
LOG_BASE="$LOG_DIR/${TAG}_${TS}_${JOBID}"
exec >"${LOG_BASE}.log" 2>"${LOG_BASE}.err"

echo "=== job ${PBS_JOBID} started at $(date) on $(hostname) — ${TAG} (lam=${LAM}, split=${SPLIT_UNIT}) ==="

source .venv/bin/activate

# Matches run 92b9afed (bbox, frozen, n=all, slice) except --lam / --split-unit /
# --epochs (10 to match the raw frozen Group 3 models).
python -m hcc_multimodal.contrastive.train \
  --model vit_b_32 \
  --gene_set all \
  --freeze_backbone \
  --lam "${LAM}" \
  --split-unit "${SPLIT_UNIT}" \
  --n_per_axis all \
  --axes 0 \
  --outcome_col rfs_2year \
  --mri_type raw_bbox \
  --bbox_pad 10 \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --num_workers 6

echo "=== job ${PBS_JOBID} finished at $(date) with exit code $? ==="
PBS
done

echo "Submitted ${#CONFIGS[@]} bbox-frozen training jobs."
