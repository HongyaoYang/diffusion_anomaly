#!/bin/bash
#SBATCH --job-name=brats_sample
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/brats_sample_%j.out
#SBATCH --error=slurm_logs/brats_sample_%j.err

set -u
set -o pipefail

PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/SURF/diffusion-anomaly"
DATA_DIR="${PROJECT_DIR}/data/brats_2d_z27_75"

DIFF_MODEL="${PROJECT_DIR}/results/brats_paper_diffusion/emabrats2update_0.9999_140000.pt"
CLS_MODEL="${PROJECT_DIR}/results/brats_paper_classifier/modelbratsclass019999.pt"

SAVE_DIR="${PROJECT_DIR}/results/brats_paper_sampling"
LOG_DIR="${PROJECT_DIR}/slurm_logs"

VISDOM_PORT=8850
VISDOM_LOG="${LOG_DIR}/visdom_sample_${SLURM_JOB_ID:-manual}.log"
VISDOM_PID=""

visdom_is_ready() {
    python - "$VISDOM_PORT" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

cd "$PROJECT_DIR" || {
    echo "ERROR: Cannot enter project directory: $PROJECT_DIR"
    exit 1
}

mkdir -p "$SAVE_DIR" "$LOG_DIR"

source "$HOME/.bashrc" || {
    echo "ERROR: Failed to source ~/.bashrc"
    exit 1
}

mamba activate env_python || {
    echo "ERROR: Failed to activate env_python"
    exit 1
}

# 防止 ~/.local/lib/python3.11/site-packages 污染 Conda 环境
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "========================================"
echo "BraTS guided sampling"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Data directory: $DATA_DIR"
echo "Diffusion model: $DIFF_MODEL"
echo "Classifier model: $CLS_MODEL"
echo "Output directory: $SAVE_DIR"
echo "Visdom port: $VISDOM_PORT"
echo "========================================"

# ============================================================
# 1. 检查数据和模型
# ============================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -f "$DIFF_MODEL" ]; then
    echo "ERROR: Diffusion checkpoint does not exist: $DIFF_MODEL"
    exit 1
fi

if [ ! -f "$CLS_MODEL" ]; then
    echo "ERROR: Classifier checkpoint does not exist: $CLS_MODEL"
    exit 1
fi

DATA_FILE_COUNT=$(find "$DATA_DIR" -type f | wc -l)

echo "Input file count: $DATA_FILE_COUNT"
echo "Diffusion checkpoint size: $(du -h "$DIFF_MODEL" | cut -f1)"
echo "Classifier checkpoint size: $(du -h "$CLS_MODEL" | cut -f1)"

if [ "$DATA_FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No data files found in $DATA_DIR"
    exit 1
fi

# ============================================================
# 2. 检查 Python 和 CUDA
# ============================================================

echo "========================================"
echo "Python and CUDA environment check"
echo "========================================"

python - <<'PY'
import os
import sys
import torch

print("Python executable:", sys.executable)
print("Python version:", sys.version.replace("\n", " "))
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable. Sampling will not continue.")

device = torch.device("cuda:0")

print("Selected device:", device)
print("GPU count visible to PyTorch:", torch.cuda.device_count())
print("Current CUDA device index:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name(0))

properties = torch.cuda.get_device_properties(0)

print(
    "Total GPU memory:",
    round(properties.total_memory / 1024**3, 2),
    "GB",
)

# 做一个实际 GPU tensor 运算，验证计算确实发生在 GPU
x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)
z = x @ y

torch.cuda.synchronize()

print("CUDA tensor device:", z.device)
print("CUDA test result shape:", tuple(z.shape))
print(
    "Allocated GPU memory:",
    round(torch.cuda.memory_allocated(0) / 1024**2, 2),
    "MB",
)
print(
    "Reserved GPU memory:",
    round(torch.cuda.memory_reserved(0) / 1024**2, 2),
    "MB",
)

print("CUDA test: PASSED")
PY

CUDA_STATUS=$?

if [ "$CUDA_STATUS" -ne 0 ]; then
    echo "ERROR: CUDA environment check failed."
    exit "$CUDA_STATUS"
fi

echo "========================================"
echo "NVIDIA GPU status before sampling"
echo "========================================"

nvidia-smi || {
    echo "WARNING: nvidia-smi failed, but PyTorch CUDA test passed."
}

# ============================================================
# 3. 启动 Visdom
# ============================================================

echo "========================================"
echo "Starting Visdom server"
echo "========================================"

# 如果当前节点的 8850 端口已经有旧进程，先提示
if command -v lsof >/dev/null 2>&1; then
    EXISTING_PID=$(lsof -ti:"$VISDOM_PORT" 2>/dev/null || true)

    if [ -n "$EXISTING_PID" ]; then
        echo "WARNING: Port $VISDOM_PORT is already in use."
        echo "Existing PID: $EXISTING_PID"
        echo "The script will attempt to use the existing Visdom server."
    fi
fi

# 只有端口未被占用时才启动新 Visdom
if ! visdom_is_ready; then

    python -m visdom.server \
        --hostname 127.0.0.1 \
        -port "$VISDOM_PORT" \
        -bind_local \
        >"$VISDOM_LOG" 2>&1 &

    VISDOM_PID=$!

    echo "Visdom process started."
    echo "Visdom PID: $VISDOM_PID"
    echo "Visdom log: $VISDOM_LOG"

    # 最多等待约 30 秒
    VISDOM_READY=0

    for ATTEMPT in $(seq 1 15); do
        if ! kill -0 "$VISDOM_PID" 2>/dev/null; then
            echo "ERROR: Visdom process exited unexpectedly."
            echo "Last Visdom log lines:"
            tail -n 50 "$VISDOM_LOG"
            exit 1
        fi

        if visdom_is_ready; then
            VISDOM_READY=1
            break
        fi

        echo "Waiting for Visdom server: attempt $ATTEMPT/15"
        sleep 2
    done

    if [ "$VISDOM_READY" -ne 1 ]; then
        echo "ERROR: Visdom did not become ready on port $VISDOM_PORT."
        echo "Last Visdom log lines:"
        tail -n 50 "$VISDOM_LOG"

        kill "$VISDOM_PID" 2>/dev/null || true
        wait "$VISDOM_PID" 2>/dev/null || true

        exit 1
    fi
else
    echo "Visdom server is already available on port $VISDOM_PORT."
fi

echo "Visdom connection check: PASSED"
echo "Visdom address on compute node:"
echo "http://127.0.0.1:${VISDOM_PORT}"
echo ""
echo "To view Visdom from your Mac while this job is running, replace <login-node>:"
echo "ssh -N -L ${VISDOM_PORT}:$(hostname):${VISDOM_PORT} hongyaoyang23@<login-node>"
echo "Then open: http://localhost:${VISDOM_PORT}"

# 无论采样成功还是失败，退出时关闭本脚本启动的 Visdom
cleanup() {
    if [ -n "${VISDOM_PID:-}" ]; then
        if kill -0 "$VISDOM_PID" 2>/dev/null; then
            echo "Stopping Visdom server: PID $VISDOM_PID"
            kill "$VISDOM_PID" 2>/dev/null || true
            wait "$VISDOM_PID" 2>/dev/null || true
        fi
    fi
}

trap cleanup EXIT INT TERM

# ============================================================
# 4. 开始采样
# ============================================================

echo "========================================"
echo "Starting guided sampling"
echo "========================================"

SAMPLE_LOG="${SAVE_DIR}/sampling_${SLURM_JOB_ID:-manual}.log"

python scripts/classifier_sample_known.py \
    --dataset brats \
    --data_dir "$DATA_DIR" \
    --model_path "$DIFF_MODEL" \
    --classifier_path "$CLS_MODEL" \
    --save_dir "$SAVE_DIR" \
    --classifier_pool attention \
    --num_samples 10 \
    --batch_size 1 \
    --num_workers 4 \
    --use_visdom true \
    --visdom_port "$VISDOM_PORT" \
    2>&1 | tee "$SAMPLE_LOG"

SAMPLE_STATUS=${PIPESTATUS[0]}

# ============================================================
# 5. 输出最终状态
# ============================================================

echo "========================================"
echo "Sampling finished"
echo "========================================"
echo "Exit code: $SAMPLE_STATUS"
echo "End time: $(date)"
echo "Sampling log: $SAMPLE_LOG"
echo "Output directory: $SAVE_DIR"

if [ "$SAMPLE_STATUS" -eq 0 ]; then
    echo "Guided sampling completed successfully."

    echo "Generated files:"
    find "$SAVE_DIR" \
        -maxdepth 2 \
        -type f \
        -printf "%TY-%Tm-%Td %TH:%TM  %10s bytes  %p\n" \
        | sort
else
    echo "ERROR: Guided sampling failed."

    echo "Last 50 lines of sampling log:"
    tail -n 50 "$SAMPLE_LOG"
fi

echo "========================================"

exit "$SAMPLE_STATUS"
