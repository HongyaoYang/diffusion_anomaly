#!/bin/bash
#SBATCH --job-name=brats_cls
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/brats_cls_%j.out
#SBATCH --error=slurm_logs/brats_cls_%j.err

set -u
set -o pipefail

PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/SURF/diffusion-anomaly"
DATA_DIR="${PROJECT_DIR}/data/brats_2d_z27_75"
SAVE_DIR="${PROJECT_DIR}/results/brats_paper_classifier"
LOG_DIR="${PROJECT_DIR}/slurm_logs"

VISDOM_HOST="127.0.0.1"
VISDOM_PORT=8850
VISDOM_PID=""

cd "$PROJECT_DIR" || {
    echo "ERROR: Cannot enter project directory: $PROJECT_DIR"
    exit 1
}

mkdir -p "$LOG_DIR" "$SAVE_DIR"

# 激活 Python 环境
source "$HOME/.bashrc"
mamba activate env_python

# 防止 ~/.local 中的包覆盖 Conda 环境
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "========================================"
echo "BraTS classifier training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "CPU cores: ${SLURM_CPUS_PER_TASK:-not-set}"
echo "Data: $DATA_DIR"
echo "Output: $SAVE_DIR"
echo "========================================"

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# 检查 GPU 是否已分配
if ! nvidia-smi; then
    echo "ERROR: nvidia-smi failed. GPU may not be assigned."
    exit 1
fi

echo "========================================"
echo "Checking PyTorch CUDA"
echo "========================================"

python - <<'PY'
import os
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.version.cuda is None:
    raise SystemExit("ERROR: CPU-only PyTorch build.")

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable.")

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(device))
print("Capability:", torch.cuda.get_device_capability(device))

# 实际执行 GPU 运算
x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)
z = x @ y

torch.cuda.synchronize()

print("Tensor device:", z.device)
print("Allocated memory:",
      round(torch.cuda.memory_allocated(device) / 1024**2, 2),
      "MiB")
print("CUDA test: PASSED")

del x, y, z
torch.cuda.empty_cache()
PY

CUDA_STATUS=$?

if [ "$CUDA_STATUS" -ne 0 ]; then
    echo "ERROR: CUDA verification failed."
    exit "$CUDA_STATUS"
fi

# 作业退出时自动关闭 Visdom
cleanup() {
    if [ -n "${VISDOM_PID:-}" ] && kill -0 "$VISDOM_PID" 2>/dev/null; then
        echo "Stopping Visdom server: PID $VISDOM_PID"
        kill "$VISDOM_PID" 2>/dev/null || true
        wait "$VISDOM_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

echo "========================================"
echo "Starting Visdom"
echo "========================================"

VISDOM_LOG="${LOG_DIR}/visdom_${SLURM_JOB_ID:-manual}.log"

python -m visdom.server \
    --hostname "$VISDOM_HOST" \
    -port "$VISDOM_PORT" \
    -bind_local \
    > "$VISDOM_LOG" 2>&1 &

VISDOM_PID=$!

echo "Visdom PID: $VISDOM_PID"
echo "Visdom log: $VISDOM_LOG"

# 最多等待 15 秒
VISDOM_READY=0

for _ in {1..15}; do
    if ! kill -0 "$VISDOM_PID" 2>/dev/null; then
        break
    fi

    if python - "$VISDOM_HOST" "$VISDOM_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    result = sock.connect_ex((host, port))

raise SystemExit(0 if result == 0 else 1)
PY
    then
        VISDOM_READY=1
        break
    fi

    sleep 1
done

if [ "$VISDOM_READY" -eq 1 ]; then
    echo "Visdom is available at ${VISDOM_HOST}:${VISDOM_PORT}"
else
    echo "WARNING: Visdom failed to start."
    tail -n 20 "$VISDOM_LOG" 2>/dev/null || true

    if kill -0 "$VISDOM_PID" 2>/dev/null; then
        kill "$VISDOM_PID" 2>/dev/null || true
        wait "$VISDOM_PID" 2>/dev/null || true
    fi

    VISDOM_PID=""
fi

echo "========================================"
echo "Starting classifier training"
echo "========================================"

TRAIN_LOG="${SAVE_DIR}/classifier_training_${SLURM_JOB_ID:-manual}.log"

python scripts/classifier_train.py \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --batch_size 1 \
    --num_workers 4 \
    --classifier_pool attention \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_STATUS=${PIPESTATUS[0]}

echo "========================================"
echo "Training finished"
echo "========================================"
echo "Exit code: $TRAIN_STATUS"
echo "End time: $(date)"
echo "Training log: $TRAIN_LOG"
echo "Output directory: $SAVE_DIR"

if [ "$TRAIN_STATUS" -eq 0 ]; then
    echo "Classifier training completed successfully."
else
    echo "ERROR: Classifier training failed."
fi

echo "========================================"

exit "$TRAIN_STATUS"