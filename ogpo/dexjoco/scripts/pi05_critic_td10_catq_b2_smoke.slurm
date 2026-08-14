#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J catq_b2_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-catq-b2-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-catq-b2-smoke-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-01:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
mkdir -p "${ROOT}/outputs/ogpo/logs" "${ROOT}/outputs/ogpo/smoke"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${ROOT}"
exec "${ROOT}/.conda/openpi/bin/python" scripts/train_udivl_critic.py \
  --config configs/ogpo/pi05_gemma_udivl_critic_100ep_td10_catq_b2_smoke.yaml

