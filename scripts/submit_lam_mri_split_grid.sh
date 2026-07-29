#!/bin/bash
# 2x2x2 contrastive training grid: lam x mri_type x split-unit.
#
#   lam        : 0.1 (outcome-regularised) | 0.0 (pure contrastive)
#   mri_type   : raw (full resampled slice) | raw_bbox (cropped to tumour bbox, pad=10)
#   split-unit : patient (no leakage) | slice (leaky, matches the older runs)
#
# Every other hyperparameter is fixed and matches the gene-order replicate sweep
# (scripts/submit_gene_order_reps.sh): vit_b_32, frozen backbone, embed_dim=128,
# gene_hidden_dim=256, temperature=0.07, reg_mode=per_modality, gene_set=all,
# n_per_axis=all, img_size=224, val_split=0.1, epochs=50, patience=2,
# checkpoint_interval=10, bs=32, lr=1e-4, wd=1e-4, seed=42.
#
# --bbox_pad is not passed: its default (10) already matches the earlier bbox
# runs. --axes is deliberately not passed either, so all three axes are sampled
# (train.py does `axes=args.axes or None`, and `--axes 0` would mean sagittal only).
#
# Each run mints its own random run_id and writes to training/contrastive/<run_id>/.
# One PBS job per config, 8 jobs total.
#
# Usage:  bash scripts/submit_lam_mri_split_grid.sh
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/rds/general/user/cl3225/home/hcc-multimodal}

# tag:lam:mri_type:split_unit
CONFIGS=(
  "grid_l1_raw_pat:0.1:raw:patient"
  "grid_l1_raw_sli:0.1:raw:slice"
  "grid_l1_bbox_pat:0.1:raw_bbox:patient"
  "grid_l1_bbox_sli:0.1:raw_bbox:slice"
  "grid_l0_raw_pat:0.0:raw:patient"
  "grid_l0_raw_sli:0.0:raw:slice"
  "grid_l0_bbox_pat:0.0:raw_bbox:patient"
  "grid_l0_bbox_sli:0.0:raw_bbox:slice"
)

for CFG in "${CONFIGS[@]}"; do
  IFS=: read -r TAG LAM MRI_TYPE SPLIT_UNIT <<<"${CFG}"
  echo "Submitting ${TAG} (lam=${LAM}, mri_type=${MRI_TYPE}, split=${SPLIT_UNIT})"
  qsub -N "${TAG}" \
    -v LAM="${LAM}",MRI_TYPE="${MRI_TYPE}",SPLIT_UNIT="${SPLIT_UNIT}",TAG="${TAG}",PROJECT_DIR="${PROJECT_DIR}" <<'PBS'
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=L40S
# 48h rather than the usual 24h: n_per_axis=all over all three axes for 50 epochs.
#PBS -l walltime=48:00:00
#PBS -q v1_gpu72

cd "${PROJECT_DIR}"

# --- logging: timestamped log/err files (PBS -o/-e can't expand variables) ---
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
JOBID=${PBS_JOBID%%.*}
LOG_BASE="$LOG_DIR/${TAG}_${TS}_${JOBID}"
exec >"${LOG_BASE}.log" 2>"${LOG_BASE}.err"

echo "=== job ${PBS_JOBID} started at $(date) on $(hostname) — ${TAG} (lam=${LAM}, mri_type=${MRI_TYPE}, split=${SPLIT_UNIT}) ==="

source .venv/bin/activate

python -m hcc_multimodal.contrastive.train \
  --model vit_b_32 \
  --embed_dim 128 \
  --gene_hidden_dim 256 \
  --freeze_backbone \
  --temperature 0.07 \
  --lam "${LAM}" \
  --reg_mode per_modality \
  --gene_set all \
  --n_per_axis all \
  --outcome_col rfs_2year \
  --img_size 224 \
  --mri_type "${MRI_TYPE}" \
  --split-unit "${SPLIT_UNIT}" \
  --val_split 0.1 \
  --epochs 50 \
  --patience 2 \
  --checkpoint_interval 10 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --seed 42 \
  --num_workers 6

echo "=== job ${PBS_JOBID} finished at $(date) with exit code $? ==="
PBS
done

echo "Submitted ${#CONFIGS[@]} grid training jobs."
