#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_pt_probe
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/full-pt-probe-b%x-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/full-pt-probe-b%x-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --constraint=L40S&48G
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH -t 0-01:00:00

set -euo pipefail

ulimit -l unlimited
ulimit -v unlimited

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OPENPI_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
DATASET_ROOT=${OUT}/data
PROBE_BATCH=${RECAP_FULL_PROBE_BATCH:?RECAP_FULL_PROBE_BATCH must be set}
PROBE_ROOT=${OUT}/full_pytorch_probe_checkpoints/b${PROBE_BATCH}_${SLURM_JOB_ID}
EXP_NAME=probe_full_pytorch_b${PROBE_BATCH}_${SLURM_JOB_ID}
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((24000 + SLURM_JOB_ID % 16000))

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp" "${PROBE_ROOT}"
test -f "${DATASET_ROOT}/click_mouse/meta/info.json"
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
export OPENPI_RECAP_SKIP_CHECKPOINT_SAVE=1
export DEXJOCO_RECAP_DATASET_ROOT=${DATASET_ROOT}
export DEXJOCO_RECAP_FULL_CKPTS_ROOT=${PROBE_ROOT}
export DEXJOCO_RECAP_FULL_BATCH_SIZE=${PROBE_BATCH}
export DEXJOCO_RECAP_FULL_TRAIN_STEPS=2
export DEXJOCO_RECAP_WARMUP_STEPS=1
export DEXJOCO_RECAP_NUM_WORKERS=0
export DEXJOCO_RECAP_SAVE_INTERVAL=1000
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p "${OPENPI_DATA_HOME}/big_vision"
cp /nfs_global/S/yangrongzheng/pi05/models/openpi/big_vision/paligemma_tokenizer.model \
  "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
touch "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model.ready"

echo "probe_start=$(date --iso-8601=seconds)"
echo "backend=pytorch config=click_mouse_recap_full full_parameters=true global_batch=${PROBE_BATCH} per_gpu_batch=$((PROBE_BATCH / 8)) train_steps=2"
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
    --checkpoint-base-dir='"${PROBE_ROOT}"' \
    --assets-base-dir='"${OPENPI_ROOT}/assets"' \
    --overwrite \
    --batch-size='"${PROBE_BATCH}"' \
    --num-train-steps=2 \
    --num-workers=0 \
    --save-interval=1000 \
    --log-interval=1 \
    --no-wandb-enabled
'

touch "${OUT}/FULL_PYTORCH_PROBE_B${PROBE_BATCH}_${SLURM_JOB_ID}_PASS"
echo "probe_pass backend=pytorch batch=${PROBE_BATCH} job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds)"
