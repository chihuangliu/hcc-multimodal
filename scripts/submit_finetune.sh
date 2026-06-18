#!/bin/bash
#PBS -N dino_finetune
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00
#PBS -q v1_gpu72

cd /rds/general/user/cl3225/home/hcc-multimodal

# --- logging: timestamped log/err files (PBS -o/-e can't expand variables) ---
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
JOBID=${PBS_JOBID%%.*}
LOG_BASE="$LOG_DIR/dino_finetune_${TS}_${JOBID}"
# redirect all stdout/stderr from here on
exec >"${LOG_BASE}.log" 2>"${LOG_BASE}.err"

echo "=== job ${PBS_JOBID} started at $(date) on $(hostname) ==="

source .venv/bin/activate

python -m hcc_multimodal.finetune.finetune_dino \
  --cohorts resection \
  --phases arterial \
  --base_model dinov2_vitb14 \
  --epochs 50 \
  --batch_size 16 \
  --axes 0 \
  --num_workers 6

echo "=== job ${PBS_JOBID} finished at $(date) with exit code $? ==="
