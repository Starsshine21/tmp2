#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J origin_pi_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo-origin/logs/origin-pi-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo-origin/logs/origin-pi-smoke-%j.err
#SBATCH -p r8nv-gpu-hw-80g
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 0-02:00:00
#SBATCH --exclude=r8a100-d05

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
mkdir -p "${ROOT}/outputs/ogpo-origin/logs"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.72
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${ROOT}"
srun \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --gpus-per-task=1 \
  --gpu-bind=closest \
  "${ROOT}/.conda/openpi/bin/python" scripts/train_full_ogpo.py \
  --config configs/ogpo/pi05_ogpo_origin_actor_smoke.yaml

