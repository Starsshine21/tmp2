#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_jax_g3s2_8g
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-jax-oldcfg-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-jax-oldcfg-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH -t 1-06:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
REFERENCE_ROOT=${RECAP_ROOT}/reference_evorl_95f8b6c
OPENPI_ROOT=${REFERENCE_ROOT}/dexjoco/openpi
OPENPI_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
VALUE_INFER_ROOT=${OUT}/value_infer_gemma3_270m_siglip2_224_10k_b32_8gpu_seed1000_oldcfg
ACP_SIDECAR_SOURCE=${VALUE_INFER_ROOT}/010000/value/acp_annotations.npz
ACP_SIDECAR=${VALUE_INFER_ROOT}/010000/value/acp_annotations_old_jax.npz
ZARR_ROOT=${OUT}/raw_rollouts/formal_seed27_100ep_796461
RAW_NPZ_ROOT=${OUT}/jax_oldcfg_raw_npz_100ep
RAW_NPZ_COMPLETE=${OUT}/JAX_OLDCFG_RAW_NPZ_100EP_COMPLETE
CKPT_ROOT=${OUT}/actor_jax_oldcfg_checkpoints
EXP_NAME=recap_click_mouse_gemma3value10k_100ep_oldcfg
RUN_ROOT=${CKPT_ROOT}/click_mouse/${EXP_NAME}

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp" "${CKPT_ROOT}"
test -f "${OUT}/VALUE_GEMMA3_SIGLIP2_10K_B32_8GPU_ACP30_COMPLETE"
test -f "${ACP_SIDECAR_SOURCE}"
test -d "${ZARR_ROOT}"
test -d "${ROOT}/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse/params"
test -f "${RECAP_ROOT}/runtime/dexjoco/openpi/assets/click_mouse/local_repo/norm_stats.json"
test -x "${OPENPI_PY}"
test ! -e "${RUN_ROOT}"

"${ROOT}/dexjoco/.conda/dexjoco/bin/python" \
  "${RECAP_ROOT}/scripts/convert_acp_sidecar_to_old_jax.py" \
  --input "${ACP_SIDECAR_SOURCE}" \
  --output "${ACP_SIDECAR}" \
  --expected-frames 66347

if [[ ! -f "${RAW_NPZ_COMPLETE}" ]]; then
  test ! -e "${RAW_NPZ_ROOT}"
  "${ROOT}/dexjoco/.conda/dexjoco/bin/python" \
    "${RECAP_ROOT}/scripts/dexjoco_zarr_to_old_jax_npz.py" \
    --input-root "${ZARR_ROOT}" \
    --output-root "${RAW_NPZ_ROOT}" \
    --expected-episodes 100 \
    --expected-frames 66347
  test "$(find "${RAW_NPZ_ROOT}" -maxdepth 1 -name 'episode_*.npz' | wc -l)" -eq 100
  touch "${RAW_NPZ_COMPLETE}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${SLURM_TMPDIR:-${OUT}/tmp}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
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
export JAX_COORDINATOR_ADDRESS="$(hostname):$((25000 + SLURM_JOB_ID % 20000))"

echo "actor_jax_oldcfg_start=$(date --iso-8601=seconds)"
echo "backend=jax config=click_mouse model=pi05 gemma_2b_lora+gemma_300m_lora devices=8"
echo "per_device_batch=1 global_batch=8 steps=30000 seed=42 action_horizon=30 max_token_len=250"
echo "optimizer=AdamW peak_lr=5e-5 warmup_steps=10000 decay_steps=1000000 decay_lr=5e-5"
echo "reference_commit=95f8b6cd3da782b84c41112c91db5c73fa60f705"
echo "base=${ROOT}/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse/params"
echo "raw_npz=${OPENPI_RAW_NPZ_ROOT} sidecar=${LEROBOT_ACP_SIDECAR_NPZ} requested_dropout=0.3"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${OPENPI_ROOT}"
srun --ntasks=8 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3,4,5,6,7 \
  "${OPENPI_PY}" scripts/train.py click_mouse \
  --exp-name="${EXP_NAME}" \
  --assets-base-dir="${RECAP_ROOT}/runtime/dexjoco/openpi/assets" \
  --weight-loader.params-path="${ROOT}/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse/params" \
  --data.root="${OUT}/data/click_mouse" \
  --overwrite \
  --batch-size=8 \
  --num-train-steps=30000 \
  --num-workers=0 \
  --save-interval=10000 \
  --keep-period=10000 \
  --log-interval=100 \
  --fsdp-devices=8

test -d "${RUN_ROOT}/29999/params"
ln -sfn "${RUN_ROOT}/29999" "${OUT}/actor_jax_oldcfg_final"
touch "${OUT}/ACTOR_JAX_OLDCFG_GEMMA3VALUE10K_100EP_COMPLETE"
echo "actor_jax_oldcfg_checkpoint=${RUN_ROOT}/29999"
echo "actor_jax_oldcfg_end=$(date --iso-8601=seconds)"
