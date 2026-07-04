#!/usr/bin/env bash
set -euo pipefail

# Wait for compute_norm_stats.py to finish, then launch two-GPU training.
# If the direct training command fails, fall back to wait_for_free_gpu_and_train.sh.

PROJECT_DIR="${PROJECT_DIR:-/home/lq/VLA/lerobot/robomme_policy_learning}"
DATASET_PATH="${DATASET_PATH:-/home/Dataset2/lq/datasets/robomme}"
EXP_NAME="${EXP_NAME:-pi05_baseline-1}"
TRAIN_GPUS="${TRAIN_GPUS:-2,4}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs}"
NORM_PATTERN="${NORM_PATTERN:-scripts/compute_norm_stats.py}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set. Export it before running this script."
  echo "Example: export WANDB_API_KEY='wandb_v1_...'"
  exit 1
fi

mkdir -p "${LOG_DIR}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
train_log="${LOG_DIR}/train_after_norm_${timestamp}.log"
fallback_log="${LOG_DIR}/fallback_wait_for_gpu_${timestamp}.log"

echo "Watching for running norm-stats process matching: ${NORM_PATTERN}"
echo "Check interval: ${CHECK_INTERVAL_SECONDS}s"

while pgrep -u "${USER}" -f "${NORM_PATTERN}" >/dev/null; do
  date '+%F %T: compute_norm_stats.py is still running; waiting...'
  sleep "${CHECK_INTERVAL_SECONDS}"
done

date '+%F %T: compute_norm_stats.py is no longer running; starting training.'
cd "${PROJECT_DIR}"

set +e
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
uv run scripts/train.py pi05_baseline \
  --exp-name="${EXP_NAME}" \
  --batch-size=64 \
  --num-workers=4 \
  --fsdp-devices=2 \
  --dataset-path="${DATASET_PATH}" 2>&1 | tee "${train_log}"
train_status=${PIPESTATUS[0]}
set -e

if [[ "${train_status}" -eq 0 ]]; then
  echo "Training finished successfully. Log: ${train_log}"
  exit 0
fi

echo "Direct two-GPU training failed with exit code ${train_status}. Log: ${train_log}"
echo "Starting fallback script: ./scripts/wait_for_free_gpu_and_train.sh"

./scripts/wait_for_free_gpu_and_train.sh 2>&1 | tee "${fallback_log}"
