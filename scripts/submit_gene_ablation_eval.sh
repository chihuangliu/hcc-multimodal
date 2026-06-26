#!/bin/bash
# Embedding evaluation for the gene-ablation sweep. Companion to
# submit_gene_ablation.sh: for each of the 20 ablation runs (each trained with a
# single gene dropped from the 2y_before_cv set, committed in 4f119f9) it runs
# `eval --mode embedding`, training LR/RF on the run's resection image
# embeddings and evaluating on the ablation cohort. One JSON per run is written,
# named by the dropped gene so each gene's contribution can be compared.
#
# A single GPU job loops over all runs (embedding extraction is light), so the
# resection/ablation cohorts are processed once per model in sequence.
#
# Usage:
#   bash scripts/submit_gene_ablation_eval.sh
#   ABLATION_SETS="soramic" TARGET=rfs_2year bash scripts/submit_gene_ablation_eval.sh
set -euo pipefail

PROJECT_DIR=/rds/general/user/cl3225/home/hcc-multimodal

# Evaluation cohorts + target (override via environment). Both ablation cohorts
# are evaluated by default; the sweep used rfs_2year.
ABLATION_SETS="${ABLATION_SETS:-soramic lusanne}"
TARGET="${TARGET:-rfs_2year}"

# run_id:dropped_gene for the 20 complete ablation runs (commit 4f119f9).
RUNS=(
  1de37286:SGSM1       1e6d7adf:H19         23b4007a:HIGD2B      2558318b:AC025198.1
  267d877c:ZMYND12     2ad23ab1:OR52N5      32b4d615:AL445235.1  48fe85c8:CAMK2N2
  4e4c4049:CSF2        85b81928:LACC1       88f00847:AC063947.2  af11f733:RBMXL3
  b10b59b4:AC025580.2  b7b61e8f:AC130366.1  c9170f09:HNRNPA1P9   ce63c6a3:AL449283.1
  cf3f7c65:AC093826.2  e302b2fb:AC004241.5  edc8c62a:AC093525.8  eee579a9:AC138647.1
)

echo "Submitting embedding-eval job for ${#RUNS[@]} ablation runs (sets=${ABLATION_SETS}, target=${TARGET})"

qsub -N "abl_eval" \
  -v RUNS="${RUNS[*]}",ABLATION_SETS="${ABLATION_SETS}",TARGET="${TARGET}",PROJECT_DIR="${PROJECT_DIR}" <<'PBS'
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=12:00:00
#PBS -q v1_gpu72

cd "${PROJECT_DIR}"

# --- logging: timestamped log/err files (PBS -o/-e can't expand variables) ---
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
JOBID=${PBS_JOBID%%.*}
LOG_BASE="$LOG_DIR/abl_eval_${TS}_${JOBID}"
exec >"${LOG_BASE}.log" 2>"${LOG_BASE}.err"

echo "=== job ${PBS_JOBID} started at $(date) on $(hostname) — embedding eval on [${ABLATION_SETS}]/${TARGET} ==="

source .venv/bin/activate

for ABLATION_SET in ${ABLATION_SETS}; do
  OUT_DIR="results/eval/${ABLATION_SET}/gene_ablation"
  mkdir -p "$OUT_DIR"
  for ENTRY in ${RUNS}; do
    RUN_ID="${ENTRY%%:*}"
    GENE="${ENTRY##*:}"
    echo "--- ${ABLATION_SET}: run ${RUN_ID} (dropped ${GENE}) at $(date) ---"
    python -m hcc_multimodal.eval.eval \
      --mode embedding \
      --model-id "${RUN_ID}" \
      --ablation-set "${ABLATION_SET}" \
      --target "${TARGET}" \
      --multi-lesion both \
      --batch-size 32 \
      --num-workers 6 \
      --output "${OUT_DIR}/embedding_${GENE}_${RUN_ID}.json"
  done
done

echo "=== job ${PBS_JOBID} finished at $(date) with exit code $? ==="
PBS

echo "Submitted embedding-eval job."
