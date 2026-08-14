#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_gemma_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-gemma-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-gemma-smoke-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-01:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}
GEMMA_DIR=${OGPO_GEMMA_DIR:-/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/gemma-3-270m}
SIGLIP_DIR=${OGPO_SIGLIP_DIR:-/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/siglip2-so400m-patch14-224-fixed}

for path in "${OPENPI_ENV}/bin/python" "${ROOT}/outputs/ogpo/click_mouse_pi05_replay_train.pt"; do
  [[ -s "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
for path in "${GEMMA_DIR}" "${SIGLIP_DIR}"; do
  [[ -d "${path}" ]] || { echo "Missing required directory: ${path}" >&2; exit 1; }
done

mkdir -p "${ROOT}/outputs/ogpo/logs"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${ROOT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${OPENPI_ENV}/bin/python" scripts/run_gemma_udivl_smoke.py \
  --config configs/ogpo/pi05_gemma_udivl_critic_smoke.yaml
