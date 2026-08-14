#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_pt_full
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-full-pytorch-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-full-pytorch-%j.err
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
GLOBAL_BATCH=${RECAP_FULL_BATCH:?RECAP_FULL_BATCH must be set from a passing probe}
TRAIN_STEPS=$((960000 / GLOBAL_BATCH))
EXP_NAME=recap_click_mouse_full_pytorch_value64k_acp30_b${GLOBAL_BATCH}_${TRAIN_STEPS}s_seed27
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((24000 + SLURM_JOB_ID % 16000))

if [[ $((GLOBAL_BATCH * TRAIN_STEPS)) -ne 960000 ]]; then
  echo "Invalid sample budget: batch=${GLOBAL_BATCH} steps=${TRAIN_STEPS}" >&2
  exit 1
fi

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp" "${CKPT_ROOT}"
test -f "${DATASET_ROOT}/click_mouse/meta/info.json"
test -f "${OUT}/VALUE_64K_ACP30_COMPLETE"
test -f "${OPENPI_ROOT}/assets/click_mouse_recap_full/local_repo/norm_stats.json"
test -f "${ROOT}/dexjoco/checkpoints/pi05_pytorch/click_mouse/model.safetensors"
test -x "${OPENPI_PY}"

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

mkdir -p "${OPENPI_DATA_HOME}/big_vision"
cp /nfs_global/S/yangrongzheng/pi05/models/openpi/big_vision/paligemma_tokenizer.model \
  "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
touch "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model.ready"

echo "formal_start=$(date --iso-8601=seconds)"
echo "backend=pytorch config=click_mouse_recap_full full_parameters=true global_batch=${GLOBAL_BATCH} per_gpu_batch=$((GLOBAL_BATCH / 8)) train_steps=${TRAIN_STEPS} total_samples=$((GLOBAL_BATCH * TRAIN_STEPS))"
echo "precision=bfloat16 optimizer=AdamW peak_lr=2e-5 warmup_steps=1000 cosine_decay_lr=2e-6 gradient_checkpointing=true"
echo "dataset=${DATASET_ROOT}/click_mouse value_steps=64000 acp_positive_ratio=0.3 indicator_dropout=0.3 tags=negative,positive source_checkpoint=${ROOT}/dexjoco/checkpoints/pi05_pytorch/click_mouse/model.safetensors"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

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
    --assets-base-dir='"${OPENPI_ROOT}/assets"' \
    $([[ ${RECAP_FULL_RESUME:-0} == 1 ]] && printf -- --resume || printf -- --overwrite) \
    --seed=27 \
    --batch-size='"${GLOBAL_BATCH}"' \
    --num-train-steps='"${TRAIN_STEPS}"' \
    --num-workers=0 \
    --save-interval=5000 \
    --log-interval=20 \
    --no-wandb-enabled
'

RUN_ROOT=${CKPT_ROOT}/click_mouse_recap_full/${EXP_NAME}
test -s "${RUN_ROOT}/${TRAIN_STEPS}/model.safetensors"
test -s "${RUN_ROOT}/${TRAIN_STEPS}/optimizer.pt"
ln -sfn "${RUN_ROOT}" "${OUT}/actor_full_pytorch_final_run"
touch "${OUT}/ACTOR_FULL_PYTORCH_COMPLETE"
echo "actor_run=${RUN_ROOT}"
echo "formal_end=$(date --iso-8601=seconds)"
