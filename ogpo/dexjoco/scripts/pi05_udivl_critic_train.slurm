#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_gemma_udivl
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-gemma-udivl-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-gemma-udivl-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}
CONFIG=${OGPO_CRITIC_CONFIG:-configs/ogpo/pi05_gemma_udivl_critic.yaml}
TRAIN_REPLAY=${OGPO_TRAIN_REPLAY:-${ROOT}/outputs/ogpo/click_mouse_pi05_replay_train.pt}
VALIDATION_REPLAY=${OGPO_VALIDATION_REPLAY:-${ROOT}/outputs/ogpo/click_mouse_pi05_replay_validation.pt}
GEMMA_DIR=${OGPO_GEMMA_DIR:-/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/gemma-3-270m}
SIGLIP_DIR=${OGPO_SIGLIP_DIR:-/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/siglip2-so400m-patch14-224-fixed}
CRITIC_CHECKPOINT=${OGPO_CRITIC_CHECKPOINT:-${ROOT}/outputs/ogpo/click_mouse_gemma_udivl.pt}
CALIBRATED_CHECKPOINT=${OGPO_CALIBRATED_CHECKPOINT:-${ROOT}/outputs/ogpo/click_mouse_gemma_udivl_calibrated.pt}
CALIBRATION_OUTPUT=${OGPO_CALIBRATION_OUTPUT:-${ROOT}/outputs/ogpo/click_mouse_gemma_calibration.json}
CONFIG_PATH=${CONFIG}
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT}/${CONFIG_PATH}"
fi

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_file "${OPENPI_ENV}/bin/python"
require_file "${TRAIN_REPLAY}"
require_file "${VALIDATION_REPLAY}"
require_file "${CONFIG_PATH}"
require_dir "${GEMMA_DIR}"
require_dir "${SIGLIP_DIR}"

mkdir -p "${ROOT}/outputs/ogpo/logs" "${ROOT}/outputs/ogpo/tensorboard"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "train_start=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID:-manual} host=$(hostname) config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${ROOT}"
TRAIN_CMD=("${OPENPI_ENV}/bin/python" scripts/train_udivl_critic.py --config "${CONFIG_PATH}")
if [[ -n "${OGPO_CRITIC_RESUME:-}" ]]; then
  require_file "${OGPO_CRITIC_RESUME}"
  TRAIN_CMD+=(--resume "${OGPO_CRITIC_RESUME}")
fi
if [[ -n "${OGPO_CONFIG_OVERLAY:-}" ]]; then
  require_file "${OGPO_CONFIG_OVERLAY}"
  TRAIN_CMD+=(--overlay "${OGPO_CONFIG_OVERLAY}")
fi
printf 'critic_command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

require_file "${CRITIC_CHECKPOINT}"
CALIBRATE_CMD=("${OPENPI_ENV}/bin/python" scripts/evaluate_ogpo.py
  --checkpoint "${CRITIC_CHECKPOINT}"
  --config "${CONFIG_PATH}"
  --fit-conformal
  --calibration-output "${CALIBRATION_OUTPUT}"
  --calibrated-checkpoint-output "${CALIBRATED_CHECKPOINT}")
if [[ -n "${OGPO_CONFIG_OVERLAY:-}" ]]; then
  CALIBRATE_CMD+=(--overlay "${OGPO_CONFIG_OVERLAY}")
fi
printf 'calibration_command:'
printf ' %q' "${CALIBRATE_CMD[@]}"
printf '\n'
"${CALIBRATE_CMD[@]}"

require_file "${CALIBRATED_CHECKPOINT}"
require_file "${CALIBRATION_OUTPUT}"
if [[ -n "${OGPO_MIN_RANKING_ACCURACY:-}" ]]; then
  "${OPENPI_ENV}/bin/python" - \
    "${CALIBRATION_OUTPUT}" "${OGPO_MIN_RANKING_ACCURACY}" <<'PY'
import json
import sys

path, threshold_text = sys.argv[1:]
with open(path, "r", encoding="utf-8") as stream:
    metrics = json.load(stream)
ranking = float(metrics["pairwise_ranking_accuracy"])
threshold = float(threshold_text)
print(f"critic_quality_gate pairwise_ranking_accuracy={ranking:.6f} threshold={threshold:.6f}")
if ranking < threshold:
    raise SystemExit(
        f"critic quality gate failed: pairwise_ranking_accuracy={ranking:.6f} < {threshold:.6f}"
    )
PY
fi
echo "calibrated_checkpoint=${CALIBRATED_CHECKPOINT}"
echo "train_end=$(date --iso-8601=seconds)"
