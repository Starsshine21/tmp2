#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_full_ogpo
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-full-train-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-full-train-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 6-00:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}
CONFIG=${OGPO_FULL_CONFIG:-configs/ogpo/pi05_full_ogpo.yaml}
TRAIN_REPLAY=${OGPO_TRAIN_REPLAY:-${ROOT}/outputs/ogpo/click_mouse_pi05_replay_train.pt}
VALIDATION_REPLAY=${OGPO_VALIDATION_REPLAY:-${ROOT}/outputs/ogpo/click_mouse_pi05_replay_validation.pt}
PI05_CHECKPOINT=${OGPO_PI05_CHECKPOINT:-${ROOT}/checkpoints/pi05_pytorch/click_mouse/model.safetensors}
GEMMA_DIR=${OGPO_GEMMA_DIR:-/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/gemma-3-270m}
SIGLIP_DIR=${OGPO_SIGLIP_DIR:-/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/siglip2-so400m-patch14-224-fixed}
CALIBRATED_CHECKPOINT=${OGPO_CALIBRATED_CHECKPOINT:-${ROOT}/outputs/ogpo/click_mouse_gemma_udivl_calibrated.pt}
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
require_file "${CONFIG_PATH}"
require_file "${TRAIN_REPLAY}"
require_file "${VALIDATION_REPLAY}"
require_file "${PI05_CHECKPOINT}"
require_file "${CALIBRATED_CHECKPOINT}"
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
CMD=("${OPENPI_ENV}/bin/python" scripts/train_full_ogpo.py --config "${CONFIG_PATH}")
if [[ -n "${OGPO_FULL_RESUME:-}" ]]; then
  require_file "${OGPO_FULL_RESUME}"
  CMD+=(--resume "${OGPO_FULL_RESUME}")
fi
if [[ -n "${OGPO_CONFIG_OVERLAY:-}" ]]; then
  require_file "${OGPO_CONFIG_OVERLAY}"
  CMD+=(--overlay "${OGPO_CONFIG_OVERLAY}")
fi
printf 'train_command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
echo "train_end=$(date --iso-8601=seconds)"
