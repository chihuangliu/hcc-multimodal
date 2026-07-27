#!/bin/bash
# Gene-order replicate sweep: re-train run dc7e1d10's configuration 15 times.
# Every hyperparameter matches dc7e1d10 (vit_b_32, base_model=3e598f36, frozen
# backbone, gene_set=all, n_per_axis=all, mri_type=raw, lam=0.1,
# reg_mode=per_modality, temperature=0.07, val_split=0.1, bs=32, lr=1e-4,
# wd=1e-4, seed=42) except:
#
#   --epochs 50               (dc7e1d10 trained 5)
#   --checkpoint_interval 10  (saves epoch_010.pt … epoch_050.pt)
#   --patience 0              (disables the new early-stopping default of 3, which
#                             would otherwise stop these runs long before epoch 50
#                             and defeat the periodic checkpoints; dc7e1d10 had no
#                             early stopping)
#   --split-unit slice        (not a change: dc7e1d10 predates --split-unit, and
#                             the behaviour back then was the slice-level split)
#
# --axes is deliberately NOT passed. dc7e1d10's metadata records `axes: 0`, but
# that is argparse's *scalar* default meaning "flag absent", and train.py does
# `axes=args.axes or None` — 0 is falsy, so None reaches build_dataset and all
# three axes are sampled. Passing `--axes 0` yields [0] and would train on
# sagittal only. Omitting the flag reproduces dc7e1d10's all-axes sampling.
#
# The 15 runs differ only in gene -> GeneEncoder input-slot assignment. That
# order comes from `set` iteration in _load_gene_matrix, which Python randomises
# per process; PYTHONHASHSEED is pinned to the replicate index here so each job
# gets a *distinct* order that is also reproducible, and so the sweep cannot
# silently collapse to 15 identical orders if the site sets PYTHONHASHSEED.
# Each run mints its own run_id, writes to training/contrastive/<run_id>/, and
# records its resolved order as `gene_order` in metadata.json.
#
# Note: base_model 3e598f36 predates gene_order recording, so its pretrained
# gene_enc slot mapping is unknown and does not line up with any replicate's
# order. That is inherited from dc7e1d10's configuration, not introduced here.
# Set BASE_MODEL="" to train the gene encoder from scratch instead.
#
# Usage:  bash scripts/submit_gene_order_reps.sh
#         REPS=5 bash scripts/submit_gene_order_reps.sh          # fewer replicates
#         BASE_MODEL="" bash scripts/submit_gene_order_reps.sh   # no warm start
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/rds/general/user/cl3225/home/hcc-multimodal}
REPS=${REPS:-15}
BASE_MODEL=${BASE_MODEL-3e598f36}

for REP in $(seq 1 "${REPS}"); do
  TAG=$(printf "gorder%02d" "${REP}")
  echo "Submitting gene-order replicate ${REP}/${REPS}: ${TAG} (PYTHONHASHSEED=${REP})"
  qsub -N "${TAG}" \
    -v REP="${REP}",TAG="${TAG}",BASE_MODEL="${BASE_MODEL}",PROJECT_DIR="${PROJECT_DIR}" <<'PBS'
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=L40S
# 48h rather than the usual 24h: this config samples every slice of all three
# axes, and 50 epochs is 10x what dc7e1d10 ran.
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

echo "=== job ${PBS_JOBID} started at $(date) on $(hostname) — ${TAG} (replicate ${REP}) ==="

source .venv/bin/activate

# Pins this replicate's gene -> input-slot order; must be set before Python starts.
export PYTHONHASHSEED="${REP}"

# --base_model is omitted entirely when BASE_MODEL is empty.
BASE_ARGS=()
if [ -n "${BASE_MODEL}" ]; then
  BASE_ARGS=(--base_model "${BASE_MODEL}")
fi

python -m hcc_multimodal.contrastive.train \
  --model vit_b_32 \
  "${BASE_ARGS[@]}" \
  --embed_dim 128 \
  --gene_hidden_dim 256 \
  --freeze_backbone \
  --temperature 0.07 \
  --lam 0.1 \
  --reg_mode per_modality \
  --gene_set all \
  --n_per_axis all \
  --outcome_col rfs_2year \
  --img_size 224 \
  --mri_type raw \
  --split-unit slice \
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

echo "Submitted ${REPS} gene-order replicate jobs."
