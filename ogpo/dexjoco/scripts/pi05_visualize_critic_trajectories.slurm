#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_qv_visual
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-qv-visual-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-qv-visual-%j.err
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
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}
CHECKPOINT=${OGPO_CRITIC_CHECKPOINT:-${ROOT}/outputs/ogpo/click_mouse_gemma_udivl_100ep_calibrated.pt}
CONFIG=${OGPO_VIS_CONFIG:-configs/ogpo/pi05_gemma_udivl_critic_100ep.yaml}
REPLAY=${OGPO_VIS_REPLAY:-${ROOT}/outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt}
OUTPUT_DIR=${OGPO_VIS_OUTPUT_DIR:-${ROOT}/outputs/ogpo/visualizations/click_mouse_critic_15k}

mkdir -p "${ROOT}/outputs/ogpo/logs" "${OUTPUT_DIR}" "${ROOT}/.cache/matplotlib"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export MPLCONFIGDIR="${ROOT}/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${ROOT}"
"${OPENPI_ENV}/bin/python" scripts/visualize_critic_trajectories.py \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --replay "${REPLAY}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 8 \
  --device cuda
