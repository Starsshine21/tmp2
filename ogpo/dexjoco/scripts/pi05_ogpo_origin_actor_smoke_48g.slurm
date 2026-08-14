#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J origin_pi_smoke48
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo-origin/logs/origin-pi-smoke48-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo-origin/logs/origin-pi-smoke48-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-02:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
mkdir -p \
  "${ROOT}/outputs/ogpo-origin/logs" \
  "${ROOT}/outputs/ogpo-origin/smoke48g"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.68
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${ROOT}"
echo "job_id=${SLURM_JOB_ID:-manual} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
exec "${ROOT}/.conda/openpi/bin/python" scripts/train_full_ogpo.py \
  --config configs/ogpo/pi05_ogpo_origin_actor_smoke_48g.yaml
