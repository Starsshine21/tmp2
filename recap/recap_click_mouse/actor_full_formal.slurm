#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_full_formal
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-full-formal-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-full-formal-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=L40S&48G
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH -t 4-23:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OPENPI_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
DATASET_ROOT=${OUT}/data
CKPT_ROOT=${OUT}/actor_full_checkpoints

PROBE_B8=${RECAP_PROBE_B8_JOB:?RECAP_PROBE_B8_JOB must be set}
PROBE_B16=${RECAP_PROBE_B16_JOB:?RECAP_PROBE_B16_JOB must be set}
PROBE_B32=${RECAP_PROBE_B32_JOB:?RECAP_PROBE_B32_JOB must be set}

if [[ -f "${OUT}/FULL_PROBE_B32_${PROBE_B32}_PASS" ]]; then
  GLOBAL_BATCH=32
  TRAIN_STEPS=30000
elif [[ -f "${OUT}/FULL_PROBE_B16_${PROBE_B16}_PASS" ]]; then
  GLOBAL_BATCH=16
  TRAIN_STEPS=60000
elif [[ -f "${OUT}/FULL_PROBE_B8_${PROBE_B8}_PASS" ]]; then
  GLOBAL_BATCH=8
  TRAIN_STEPS=120000
else
  echo "No 48G full-parameter batch probe passed; refusing to start formal training." >&2
  exit 1
fi

EXP_NAME=recap_click_mouse_full_b${GLOBAL_BATCH}_${TRAIN_STEPS}s
RUN_ROOT=${CKPT_ROOT}/click_mouse_recap_full/${EXP_NAME}
FINAL_STEP=$((TRAIN_STEPS - 1))

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp" "${CKPT_ROOT}"
test -f "${DATASET_ROOT}/click_mouse/meta/info.json"
test -f "${OPENPI_ROOT}/assets/click_mouse_recap_full/local_repo/norm_stats.json"
test -x "${OPENPI_PY}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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
export OPENPI_RECAP_LORA_ONLY=0
export OPENPI_SKIP_INITIAL_IMAGE_LOG=1
export OPENPI_RECAP_SHARDED_WEIGHT_INIT=1
unset OPENPI_RECAP_SKIP_CHECKPOINT_SAVE
export DEXJOCO_RECAP_DATASET_ROOT=${DATASET_ROOT}
export DEXJOCO_RECAP_FULL_CKPTS_ROOT=${CKPT_ROOT}
export DEXJOCO_RECAP_FULL_BATCH_SIZE=${GLOBAL_BATCH}
export DEXJOCO_RECAP_FULL_TRAIN_STEPS=${TRAIN_STEPS}
export DEXJOCO_RECAP_WARMUP_STEPS=1000
export DEXJOCO_RECAP_NUM_WORKERS=0
export DEXJOCO_RECAP_SAVE_INTERVAL=5000
export DEXJOCO_RECAP_KEEP_PERIOD=10000
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "formal_start=$(date --iso-8601=seconds)"
echo "config=click_mouse_recap_full full_parameters=true global_batch=${GLOBAL_BATCH} fsdp_devices=8 train_steps=${TRAIN_STEPS} total_samples=$((GLOBAL_BATCH * TRAIN_STEPS))"
echo "optimizer=AdamW peak_lr=2e-5 warmup_steps=1000 cosine_decay_lr=2e-6"
echo "dataset=${DATASET_ROOT}/click_mouse source_checkpoint=/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse/params"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${OPENPI_ROOT}"
"${OPENPI_PY}" scripts/train.py click_mouse_recap_full \
  --exp-name="${EXP_NAME}" \
  --overwrite \
  --batch-size="${GLOBAL_BATCH}" \
  --num-train-steps="${TRAIN_STEPS}" \
  --num-workers=0 \
  --save-interval=5000 \
  --keep-period=10000 \
  --log-interval=50 \
  --fsdp-devices=8

test -d "${RUN_ROOT}/${FINAL_STEP}/params"
test -d "${RUN_ROOT}/${FINAL_STEP}/train_state"
ln -sfn "${RUN_ROOT}/${FINAL_STEP}" "${OUT}/actor_full_final"
touch "${OUT}/ACTOR_FULL_COMPLETE"
echo "actor_checkpoint=${RUN_ROOT}/${FINAL_STEP}"
echo "formal_end=$(date --iso-8601=seconds)"
