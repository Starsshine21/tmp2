#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_eval_mse_hand_dbg
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-eval-hand-dbg-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-eval-hand-dbg-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH -t 0-03:00:00
###SBATCH --constraint=L40S|A100

set -euo pipefail

export PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
export CHECKPOINT_DIR="${PI05_EVAL_CHECKPOINT_DIR:-${PI05_REPO_ROOT}/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_dexhand_full_lora_pytorch_32/pi05_pickplace_dexhand_eef_delta_train_lora/20000}"
export DATA_DIR="${PI05_EVAL_DATA_DIR:-${PI05_REPO_ROOT}/data/lerobot_pick_place_dexhand}"
export TRAIN_CONFIG="${PI05_EVAL_TRAIN_CONFIG:-pi05_pickplace_dexhand_full_lora_pytorch_32}"
export ASSETS_BASE_DIR="${PI05_EVAL_ASSETS_BASE_DIR:-${PI05_REPO_ROOT}/openpi_official/assets_eef_delta_v2}"
export MAX_EPISODES="${PI05_EVAL_MAX_EPISODES:-5}"
export NUM_STEPS="${PI05_EVAL_NUM_STEPS:-10}"
export EVAL_BATCH_SIZE="${PI05_EVAL_BATCH_SIZE:-8}"
export DEVICE="${PI05_EVAL_DEVICE:-cuda}"

mkdir -p "${PI05_REPO_ROOT}/logs"
source /nfs_global/S/yangrongzheng/miniconda3-fresh/etc/profile.d/conda.sh
conda activate "${PI05_REPO_ROOT}/.conda-pi05-openpi-final"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PI05_REPO_ROOT}/openpi_official/src:${PI05_REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR=/tmp
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "${PI05_REPO_ROOT}"

echo "PI05 eval hand debug start: $(date '+%F %T')"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "ASSETS_BASE_DIR=${ASSETS_BASE_DIR}"
echo "MAX_EPISODES=${MAX_EPISODES}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
python scripts/eval_mse.py \
  --ckpt-dirs "${CHECKPOINT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --train-config "${TRAIN_CONFIG}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --max-episodes "${MAX_EPISODES}" \
  --device "${DEVICE}" \
  --num-steps "${NUM_STEPS}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --debug-hand

echo "PI05 eval hand debug end: $(date '+%F %T')"
