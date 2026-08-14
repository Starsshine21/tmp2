#!/bin/bash
#SBATCH -J pi05_jax_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-jax-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-jax-smoke-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 0-01:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}
CONFIG=${OGPO_JAX_SMOKE_CONFIG:-configs/ogpo/pi05_jax_flash_ogpo_gpu_smoke.yaml}

mkdir -p "${ROOT}/outputs/ogpo/logs"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# JAX + PyTorch critic cohabit on one GPU: cap JAX preallocation.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "jax_smoke_start=$(date --iso-8601=seconds)"
echo "host=$(hostname) config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${ROOT}"
"${OPENPI_ENV}/bin/python" scripts/train_flash_ogpo.py --config "${CONFIG}"

echo "jax_smoke_end=$(date --iso-8601=seconds)"
