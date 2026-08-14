#!/bin/bash
#SBATCH -J pi05_flash_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-flash-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-flash-smoke-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 0-00:30:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}

mkdir -p "${ROOT}/outputs/ogpo/logs"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "smoke_start=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${ROOT}"
"${OPENPI_ENV}/bin/python" scripts/train_flash_ogpo.py \
  --config configs/ogpo/pi05_flash_ogpo_gpu_smoke.yaml

echo "smoke_end=$(date --iso-8601=seconds)"
