#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J td10_12k_cal
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-td10-12k-cal-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-td10-12k-cal-%j.err
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
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${ROOT}"
"${ROOT}/.conda/openpi/bin/python" scripts/evaluate_ogpo.py \
  --checkpoint outputs/ogpo/checkpoints/click_mouse_divl_td10_balanced/milestones/critic_step_12000.pt \
  --config configs/ogpo/pi05_gemma_udivl_critic_100ep_td10_balanced_30k_b32.yaml \
  --validation-replay outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt \
  --fit-conformal \
  --calibration-output outputs/ogpo/click_mouse_gemma_udivl_td10_balanced_step12000_calibration.json \
  --calibrated-checkpoint-output outputs/ogpo/click_mouse_gemma_udivl_100ep_td10_balanced_30k_b32_best_calibrated.pt
