  #!/bin/bash
  #PBS -N dino_finetune
  #PBS -l select=1:ncpus=8:mem=48gb:ngpus=1:gpu_type=L40S
  #PBS -l walltime=72:00:00
  #PBS -q v1_gpu72

  cd /rds/general/user/cl3225/home/hcc-multimodal

  source .venv/bin/activate

  python -m hcc_multimodal.finetune.finetune_dino \
    --cohorts resection\
    --phases arterial \
    --base_model dinov2_vitb14 \
    --epochs 50 \
    --batch_size 16 \
    --axes 0 \
    --num_workers 6
