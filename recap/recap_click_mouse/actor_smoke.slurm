#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_actor_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-smoke-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-02:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OPENPI_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
DATASET_ROOT=${OUT}/data_smoke
CKPT_ROOT=${OUT}/actor_smoke_checkpoints
EXP_NAME=recap_click_mouse_smoke

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp"
test -f "${OUT}/VALUE_SMOKE_COMPLETE"
test -f "${DATASET_ROOT}/click_mouse/meta/info.json"
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
export DEXJOCO_RECAP_WARMUP_STEPS=1
export DEXJOCO_RECAP_NUM_WORKERS=0
export DEXJOCO_RECAP_SAVE_INTERVAL=1
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "actor_smoke_start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${OPENPI_ROOT}"
"${OPENPI_PY}" scripts/compute_norm_stats.py click_mouse --batch-size=32 --num-workers=0

"${OPENPI_PY}" scripts/train.py click_mouse \
  --exp-name="${EXP_NAME}" \
  --overwrite \
  --num-train-steps=2 \
  --num-workers=0 \
  --save-interval=1 \
  --log-interval=1 \
  --fsdp-devices=1

FINAL_ROOT=${CKPT_ROOT}/click_mouse/${EXP_NAME}
FINAL_STEP=$(find "${FINAL_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -1)
test -n "${FINAL_STEP}"
test -d "${FINAL_ROOT}/${FINAL_STEP}/params"
touch "${OUT}/ACTOR_SMOKE_COMPLETE"
echo "actor_smoke_checkpoint=${FINAL_ROOT}/${FINAL_STEP}"
echo "actor_smoke_end=$(date --iso-8601=seconds)"
