#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_full_probe
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/full-probe-b%x-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/full-probe-b%x-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=A100&40G
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH -t 0-02:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OPENPI_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
DATASET_ROOT=${OUT}/data
PROBE_BATCH=${RECAP_FULL_PROBE_BATCH:?RECAP_FULL_PROBE_BATCH must be set}
PROBE_CKPT_ROOT=${OUT}/full_probe_checkpoints/b${PROBE_BATCH}_${SLURM_JOB_ID}
EXP_NAME=probe_full_b${PROBE_BATCH}_${SLURM_JOB_ID}

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp" "${PROBE_CKPT_ROOT}"
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
export OPENPI_RECAP_SKIP_CHECKPOINT_SAVE=1
export OPENPI_SKIP_INITIAL_IMAGE_LOG=1
export OPENPI_RECAP_SHARDED_WEIGHT_INIT=1
export DEXJOCO_RECAP_DATASET_ROOT=${DATASET_ROOT}
export DEXJOCO_RECAP_FULL_CKPTS_ROOT=${PROBE_CKPT_ROOT}
export DEXJOCO_RECAP_FULL_BATCH_SIZE=${PROBE_BATCH}
export DEXJOCO_RECAP_FULL_TRAIN_STEPS=2
export DEXJOCO_RECAP_WARMUP_STEPS=1
export DEXJOCO_RECAP_NUM_WORKERS=0
export DEXJOCO_RECAP_SAVE_INTERVAL=1000
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "probe_start=$(date --iso-8601=seconds)"
echo "config=click_mouse_recap_full global_batch=${PROBE_BATCH} fsdp_devices=8 train_steps=2"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${OPENPI_ROOT}"
"${OPENPI_PY}" scripts/train.py click_mouse_recap_full \
  --exp-name="${EXP_NAME}" \
  --overwrite \
  --batch-size="${PROBE_BATCH}" \
  --num-train-steps=2 \
  --num-workers=0 \
  --save-interval=1000 \
  --log-interval=1 \
  --fsdp-devices=8

touch "${OUT}/FULL_PROBE_B${PROBE_BATCH}_${SLURM_JOB_ID}_PASS"
echo "probe_pass batch=${PROBE_BATCH} job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds)"
