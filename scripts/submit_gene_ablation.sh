#!/bin/bash
# Gene-ablation sweep: re-train the best Soramic config (run 9109a6c2) once per
# gene, each time dropping a single gene from the 20-gene 2y_before_cv set, to
# measure each gene's contribution. Submits one PBS job per gene (20 total).
#
# Usage:  bash scripts/submit_gene_ablation.sh
set -euo pipefail

PROJECT_DIR=/rds/general/user/cl3225/home/hcc-multimodal

# The 20 genes of RNA_2Y_BEFORE_CV_GENES (gene_set=2y_before_cv).
GENES=(
  AC004241.5 AC025198.1 AC025580.2 AC063947.2 AC093525.8
  AC093826.2 AC130366.1 AC138647.1 AL445235.1 AL449283.1
  CAMK2N2 CSF2 H19 HIGD2B HNRNPA1P9
  LACC1 OR52N5 RBMXL3 SGSM1 ZMYND12
)

for GENE in "${GENES[@]}"; do
  echo "Submitting ablation job dropping gene: ${GENE}"
  qsub -N "abl_${GENE}" -v DROP_GENE="${GENE}",PROJECT_DIR="${PROJECT_DIR}" <<'PBS'
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
LOG_BASE="$LOG_DIR/abl_${DROP_GENE}_${TS}_${JOBID}"
exec >"${LOG_BASE}.log" 2>"${LOG_BASE}.err"

echo "=== job ${PBS_JOBID} started at $(date) on $(hostname) — dropping ${DROP_GENE} ==="

source .venv/bin/activate

# Same configuration as run 9109a6c2 (best AUC on Soramic), dropping one gene.
python -m hcc_multimodal.contrastive.train \
  --model vit_b_32 \
  --gene_set 2y_before_cv \
  --drop-genes "${DROP_GENE}" \
  --n_per_axis 10 \
  --axes 0 \
  --outcome_col rfs_2year \
  --mri_type raw \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --num_workers 6

echo "=== job ${PBS_JOBID} finished at $(date) with exit code $? ==="
PBS
done

echo "Submitted ${#GENES[@]} gene-ablation jobs."
