#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_click_actor
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 1-00:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OPENPI_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
DATASET_ROOT=${OUT}/data
DATA_ROOT=${DATASET_ROOT}/click_mouse
CKPT_ROOT=${OUT}/actor_checkpoints
EXP_NAME=recap_click_mouse_pistar06_acp_2k

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp"
test -f "${OUT}/VALUE_COMPLETE"
test -f "${DATA_ROOT}/meta/info.json"
test -x "${OPENPI_PY}"
test -d "${ROOT}/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse/params"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${OUT}/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export OPENPI_LEROBOT_ACP_ENABLE=1
export OPENPI_LEROBOT_ACP_INDICATOR_FIELD=complementary_info.acp_indicator
export OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB=0.3
export OPENPI_LEROBOT_ACP_TAG_KEY=Advantage
export OPENPI_LEROBOT_ACP_TAG_VALUES=negative,positive
export OPENPI_LEROBOT_ACP_SEED=27
export OPENPI_RECAP_LORA_ONLY=1
export DEXJOCO_RECAP_DATASET_ROOT=${DATASET_ROOT}
export DEXJOCO_RECAP_CKPTS_ROOT=${CKPT_ROOT}
export DEXJOCO_RECAP_WARMUP_STEPS=150
export DEXJOCO_RECAP_NUM_WORKERS=0
export DEXJOCO_RECAP_SAVE_INTERVAL=500
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "recap_actor_job_start=$(date --iso-8601=seconds)"
echo "value_dataset=${DATA_ROOT}"
echo "base_checkpoint=${ROOT}/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse/params"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${OPENPI_ROOT}"
"${OPENPI_PY}" scripts/compute_norm_stats.py click_mouse --batch-size=32 --num-workers=8

"${OPENPI_PY}" scripts/train.py click_mouse \
  --exp-name="${EXP_NAME}" \
  --overwrite \
  --num-train-steps=2000 \
  --num-workers=0 \
  --save-interval=500 \
  --log-interval=50 \
  --fsdp-devices=1

FINAL_ROOT=${CKPT_ROOT}/click_mouse/${EXP_NAME}
FINAL_STEP=$(find "${FINAL_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -1)
test -n "${FINAL_STEP}"
test -d "${FINAL_ROOT}/${FINAL_STEP}/params"
ln -sfn "${FINAL_ROOT}/${FINAL_STEP}" "${OUT}/actor_final"
touch "${OUT}/ACTOR_COMPLETE"
echo "actor_checkpoint=${FINAL_ROOT}/${FINAL_STEP}"
echo "recap_actor_job_end=$(date --iso-8601=seconds)"
