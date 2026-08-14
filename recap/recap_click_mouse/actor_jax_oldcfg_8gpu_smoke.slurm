#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_jax_8g_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-jax-8gpu-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-jax-8gpu-smoke-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -t 0-02:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
REFERENCE_ROOT=${RECAP_ROOT}/reference_evorl_95f8b6c
OPENPI_ROOT=${REFERENCE_ROOT}/dexjoco/openpi
OPENPI_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
RAW_NPZ_ROOT=${OUT}/jax_oldcfg_raw_npz_100ep
ACP_SIDECAR=${OUT}/tmp/acp_old_jax_smoke_${SLURM_JOB_ID}.npz
CKPT_ROOT=${OUT}/actor_jax_8gpu_smoke_checkpoints
EXP_NAME=smoke_${SLURM_JOB_ID}
RUN_ROOT=${CKPT_ROOT}/click_mouse/${EXP_NAME}

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp" "${CKPT_ROOT}"
test -x "${OPENPI_PY}"
test -f "${OUT}/JAX_OLDCFG_RAW_NPZ_100EP_COMPLETE"
test "$(find "${RAW_NPZ_ROOT}" -maxdepth 1 -name 'episode_*.npz' | wc -l)" -eq 100
test -d "${ROOT}/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse/params"
test -f "${RECAP_ROOT}/runtime/dexjoco/openpi/assets/click_mouse/local_repo/norm_stats.json"
test ! -e "${RUN_ROOT}"

"${ROOT}/dexjoco/.conda/dexjoco/bin/python" \
  "${RECAP_ROOT}/scripts/convert_acp_sidecar_to_old_jax.py" \
  --output "${ACP_SIDECAR}" \
  --expected-frames 66347 \
  --smoke-positive-ratio 0.3

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${SLURM_TMPDIR:-${OUT}/tmp}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export OPENPI_RAW_NPZ_ROOT=${RAW_NPZ_ROOT}
export LEROBOT_ACP_SIDECAR_NPZ=${ACP_SIDECAR}
export LEROBOT_ACP_DROPOUT=0.3
export OPENPI_DATASET_ROOT=${OUT}/data
export OPENPI_CKPTS_ROOT=${CKPT_ROOT}
export PYTHONPATH="${REFERENCE_ROOT}/src:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset OPENPI_LEROBOT_ACP_ENABLE OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "smoke_start=$(date --iso-8601=seconds)"
echo "devices=8 per_device_batch=2 global_batch=16 steps=3 fsdp_devices=8"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${OPENPI_ROOT}"
"${OPENPI_PY}" -c 'import jax; assert jax.device_count() == 8, jax.devices(); print("jax_devices=", jax.devices())'
"${OPENPI_PY}" scripts/train.py click_mouse \
  --exp-name="${EXP_NAME}" \
  --assets-base-dir="${RECAP_ROOT}/runtime/dexjoco/openpi/assets" \
  --weight-loader.params-path="${ROOT}/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse/params" \
  --data.root="${OUT}/data/click_mouse" \
  --overwrite \
  --batch-size=16 \
  --num-train-steps=3 \
  --num-workers=4 \
  --save-interval=2 \
  --keep-period=2 \
  --log-interval=1 \
  --fsdp-devices=8

test -d "${RUN_ROOT}/2/params"
touch "${OUT}/ACTOR_JAX_8GPU_SMOKE_${SLURM_JOB_ID}_COMPLETE"
echo "smoke_checkpoint=${RUN_ROOT}/2"
echo "smoke_end=$(date --iso-8601=seconds)"
