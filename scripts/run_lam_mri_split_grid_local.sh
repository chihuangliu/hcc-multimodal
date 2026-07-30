#!/bin/bash
# Local (no PBS) runner for the lam x mri_type x split-unit contrastive grid.
# Same hyperparameters as scripts/submit_lam_mri_split_grid.sh, but runs the
# configs sequentially on this machine (CUDA / MPS / CPU, whichever train.py
# picks) instead of submitting one qsub job each.
#
#   lam        : 0.1 (outcome-regularised) | 0.0 (pure contrastive)
#   mri_type   : raw (full resampled slice) | raw_bbox (cropped to tumour bbox, pad=10)
#   split-unit : patient (no leakage) | slice (leaky, matches the older runs)
#
# --bbox_pad is not passed: its default (10) already matches the earlier bbox
# runs. --axes is deliberately not passed either, so all three axes are sampled
# (train.py does `axes=args.axes or None`, and `--axes 0` would mean sagittal only).
#
# Each run mints its own random run_id and writes to training/contrastive/<run_id>/.
# Per-run stdout/stderr go to logs/<tag>_<timestamp>.log / .err.
#
# Usage:
#   bash scripts/run_lam_mri_split_grid_local.sh                  # default 7 configs
#   bash scripts/run_lam_mri_split_grid_local.sh tag:lam:mri:split ...
#
# Each positional argument is one "tag:lam:mri_type:split_unit" config.
set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-"${PROJECT_DIR}/.venv/bin/python"}
EPOCHS=${EPOCHS:-50}
NUM_WORKERS=${NUM_WORKERS:-6}

# Default order: all raw_bbox first, then raw x patient, then raw x slice x 0.0.
# raw x slice x 0.1 is deliberately omitted (already covered elsewhere), so 7 of
# the 8 grid cells run here.
DEFAULT_CONFIGS=(
  "grid_l1_bbox_sli:0.1:raw_bbox:slice"
  "grid_l0_bbox_sli:0.0:raw_bbox:slice"
  "grid_l1_bbox_pat:0.1:raw_bbox:patient"
  "grid_l0_bbox_pat:0.0:raw_bbox:patient"
  "grid_l1_raw_pat:0.1:raw:patient"
  "grid_l0_raw_pat:0.0:raw:patient"
  "grid_l0_raw_sli:0.0:raw:slice"
)

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

cd "${PROJECT_DIR}"
LOG_DIR=logs
mkdir -p "$LOG_DIR"

FAILED=()
for CFG in "${CONFIGS[@]}"; do
  IFS=: read -r TAG LAM MRI_TYPE SPLIT_UNIT <<<"${CFG}"
  TS=$(date +%Y%m%d_%H%M%S)
  LOG_BASE="${LOG_DIR}/${TAG}_${TS}"
  echo "=== ${TAG} (lam=${LAM}, mri_type=${MRI_TYPE}, split=${SPLIT_UNIT}) started at $(date) → ${LOG_BASE}.log"

  "${PYTHON}" -m hcc_multimodal.contrastive.train \
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
    --epochs "${EPOCHS}" \
    --patience 2 \
    --checkpoint_interval 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --seed 42 \
    --num_workers "${NUM_WORKERS}" \
    >"${LOG_BASE}.log" 2>"${LOG_BASE}.err"
  RC=$?

  if [ "$RC" -ne 0 ]; then
    echo "=== ${TAG} FAILED (exit ${RC}) at $(date) — see ${LOG_BASE}.err"
    FAILED+=("${TAG}")
  else
    echo "=== ${TAG} finished at $(date)"
  fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Finished with ${#FAILED[@]} failed config(s): ${FAILED[*]}"
  exit 1
fi
echo "All ${#CONFIGS[@]} configs finished."
