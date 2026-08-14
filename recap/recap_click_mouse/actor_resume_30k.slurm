#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_actor_r30k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-resume30k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-resume30k-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 0-06:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OPENPI_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
DATASET_ROOT=${OUT}/data
CKPT_ROOT=${OUT}/actor_checkpoints
EXP_NAME=recap_click_mouse_pistar06_acp_2k
RUN_ROOT=${CKPT_ROOT}/click_mouse/${EXP_NAME}
START_CHECKPOINT=${RUN_ROOT}/1999
TARGET_TRAIN_STEPS=32000

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp"
test -f "${OUT}/ACTOR_COMPLETE"
test -f "${DATASET_ROOT}/click_mouse/meta/info.json"
test -x "${OPENPI_PY}"
test -d "${START_CHECKPOINT}/params"
test -d "${START_CHECKPOINT}/train_state"

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
export DEXJOCO_RECAP_SAVE_INTERVAL=5000
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "actor_resume_start=$(date --iso-8601=seconds)"
echo "resume_checkpoint=${START_CHECKPOINT}"
echo "start_train_state_step=2000 additional_steps=30000 target_train_steps=${TARGET_TRAIN_STEPS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${OPENPI_ROOT}"
"${OPENPI_PY}" scripts/train.py click_mouse \
  --exp-name="${EXP_NAME}" \
  --resume \
  --num-train-steps="${TARGET_TRAIN_STEPS}" \
  --num-workers=0 \
  --save-interval=5000 \
  --keep-period=1999 \
  --log-interval=100 \
  --fsdp-devices=1

FINAL_STEP=$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -1)
if [[ "${FINAL_STEP}" != 31999 ]]; then
  echo "Expected final checkpoint 31999, got ${FINAL_STEP}" >&2
  exit 1
fi
test -d "${RUN_ROOT}/${FINAL_STEP}/params"
ln -sfn "${RUN_ROOT}/${FINAL_STEP}" "${OUT}/actor_final"
touch "${OUT}/ACTOR_30K_COMPLETE"
echo "actor_checkpoint=${RUN_ROOT}/${FINAL_STEP}"
echo "actor_resume_end=$(date --iso-8601=seconds)"
