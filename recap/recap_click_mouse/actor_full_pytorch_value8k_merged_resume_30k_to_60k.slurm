#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_v8k_r60k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-full-v8k-resume60k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-full-v8k-resume60k-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --constraint=L40S&48G
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH -t 4-23:00:00

set -euo pipefail

ulimit -l unlimited
ulimit -v unlimited

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OPENPI_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
DATASET_ROOT=${OUT}/data
CKPT_ROOT=${OUT}/actor_full_pytorch_checkpoints
MERGED_MODEL=${ROOT}/dexjoco/checkpoints/pi05_pytorch/click_mouse_lora_merged/model.safetensors
GLOBAL_BATCH=32
START_STEP=${RECAP_RESUME_START_STEP:-30000}
TARGET_STEP=${RECAP_RESUME_TARGET_STEP:-60000}
LR_DECAY_STEPS=30000
EXP_NAME=recap_click_mouse_full_pytorch_mergedbase_value8k_acp30_b32_30k_seed27
RUN_ROOT=${CKPT_ROOT}/click_mouse_recap_full/${EXP_NAME}
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((24000 + SLURM_JOB_ID % 16000))

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp" "${CKPT_ROOT}"
test -f "${DATASET_ROOT}/click_mouse/meta/info.json"
test -s "${RUN_ROOT}/${START_STEP}/model.safetensors"
test -s "${RUN_ROOT}/${START_STEP}/optimizer.pt"
test -s "${RUN_ROOT}/${START_STEP}/metadata.pt"
test -s "${MERGED_MODEL}"
test -f "${OPENPI_ROOT}/assets/click_mouse_recap_full/local_repo/norm_stats.json"
test -x "${OPENPI_PY}"

LATEST_STEP=$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sed -n '/^[0-9][0-9]*$/p' | sort -n | tail -1)
if [[ "${LATEST_STEP}" != "${START_STEP}" ]]; then
  echo "Expected latest checkpoint ${START_STEP}, got ${LATEST_STEP}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export OPENPI_DATA_HOME=${OUT}/cache/openpi
export TMPDIR=${OUT}/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export WANDB_MODE=disabled
export OPENPI_LEROBOT_ACP_ENABLE=1
export OPENPI_LEROBOT_ACP_INDICATOR_FIELD=complementary_info.acp_indicator
export OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB=0.3
export OPENPI_LEROBOT_ACP_TAG_KEY=Advantage
export OPENPI_LEROBOT_ACP_TAG_VALUES=negative,positive
export OPENPI_LEROBOT_ACP_SEED=27
export DEXJOCO_RECAP_DATASET_ROOT=${DATASET_ROOT}
export DEXJOCO_RECAP_FULL_CKPTS_ROOT=${CKPT_ROOT}
export DEXJOCO_RECAP_FULL_BATCH_SIZE=${GLOBAL_BATCH}
export DEXJOCO_RECAP_FULL_TRAIN_STEPS=${TARGET_STEP}
export DEXJOCO_RECAP_FULL_PYTORCH_WEIGHT_PATH=${MERGED_MODEL}
export DEXJOCO_RECAP_WARMUP_STEPS=1000
export DEXJOCO_RECAP_NUM_WORKERS=0
export DEXJOCO_RECAP_SAVE_INTERVAL=5000
export DEXJOCO_RECAP_KEEP_PERIOD=10000
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p "${OPENPI_DATA_HOME}/big_vision"
cp /nfs_global/S/yangrongzheng/pi05/models/openpi/big_vision/paligemma_tokenizer.model \
  "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
touch "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model.ready"

echo "resume_start=$(date --iso-8601=seconds)"
echo "run_root=${RUN_ROOT} resume_from=${START_STEP} target_step=${TARGET_STEP} additional_steps=$((TARGET_STEP - START_STEP))"
echo "backend=pytorch full_parameters=true global_batch=32 per_gpu_batch=4 save_interval=5000"
echo "optimizer=AdamW restored=true lr_schedule=original_30k_cosine end_lr=2e-6 continuation_lr=2e-6"

srun --export=ALL --ntasks=8 --ntasks-per-node=8 bash -lc '
  ulimit -l unlimited
  ulimit -v unlimited
  export RANK=${SLURM_PROCID}
  export WORLD_SIZE=8
  export LOCAL_RANK=${SLURM_LOCALID}
  export MASTER_ADDR='"${MASTER_ADDR}"'
  export MASTER_PORT='"${MASTER_PORT}"'
  cd '"${OPENPI_ROOT}"'
  '"${OPENPI_PY}"' scripts/train_pytorch.py click_mouse_recap_full \
    --exp-name='"${EXP_NAME}"' \
    --checkpoint-base-dir='"${CKPT_ROOT}"' \
    --assets-base-dir='"${OPENPI_ROOT}"'/assets \
    --resume \
    --seed=27 \
    --batch-size=32 \
    --num-train-steps='"${TARGET_STEP}"' \
    --lr-schedule.decay-steps='"${LR_DECAY_STEPS}"' \
    --num-workers=0 \
    --save-interval=5000 \
    --log-interval=20 \
    --no-wandb-enabled
'

test -s "${RUN_ROOT}/${TARGET_STEP}/model.safetensors"
test -s "${RUN_ROOT}/${TARGET_STEP}/optimizer.pt"
ln -sfn "${RUN_ROOT}" "${OUT}/actor_full_pytorch_corrected_run"
touch "${OUT}/ACTOR_FULL_PYTORCH_CORRECTED_${TARGET_STEP}_COMPLETE"
echo "actor_run=${RUN_ROOT}"
echo "resume_end=$(date --iso-8601=seconds)"
