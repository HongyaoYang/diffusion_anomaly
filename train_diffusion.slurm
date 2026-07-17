#!/bin/bash
#SBATCH --job-name=brats_diff
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/brats_diff_%j.out
#SBATCH --error=slurm_logs/brats_diff_%j.err

cd /gpfs/work/bio/hongyaoyang23/SURF/diffusion-anomaly

source ~/.bashrc
mamba activate env_python

echo "=============================="
echo "Job: BraTS diffusion training"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"
echo "=============================="

echo ""
echo "========== SLURM GPU INFO =========="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "===================================="
echo ""

echo "========== NVIDIA-SMI =========="
which nvidia-smi
nvidia-smi
echo "================================"
echo ""

echo "========== PYTORCH CUDA CHECK =========="
python - <<'PY'
import torch
import os

print("Python executable:", os.sys.executable)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if torch.cuda.is_available():
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("Current GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    x = torch.randn(2, 2).cuda()
    print("Test tensor device:", x.device)
else:
    print("WARNING: PyTorch cannot use CUDA. This job may run on CPU.")
PY
echo "======================================="
echo ""

python scripts/image_train.py \
  --data_dir ./data/brats_2d_z27_75 \
  --save_dir ./results/brats_paper_diffusion \
  --batch_size 1 \
  --num_workers 4

echo "=============================="
echo "End time: $(date)"
echo "=============================="