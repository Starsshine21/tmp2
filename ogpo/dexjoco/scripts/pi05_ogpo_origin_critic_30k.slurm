#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J origin_mc_q30k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo-origin/logs/origin-mc-q30k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo-origin/logs/origin-mc-q30k-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 6-00:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
mkdir -p "${ROOT}/outputs/ogpo-origin/logs" "${ROOT}/outputs/ogpo-origin/tensorboard"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${ROOT}"
exec "${ROOT}/.conda/openpi/bin/python" scripts/train_udivl_critic.py \
  --config configs/ogpo/pi05_ogpo_origin_critic_100ep.yaml
