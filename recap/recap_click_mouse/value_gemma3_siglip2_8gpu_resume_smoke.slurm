#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_v_resume_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-resume-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-resume-smoke-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -t 0-00:20:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
VALUE_DIR=${OUT}/value_pistar06_gemma3_270m_siglip2_224_10k_b32_8gpu_seed1000_oldcfg
RESUME_CKPT=${VALUE_DIR}/checkpoints/004000
RESUME_CONFIG=${RESUME_CKPT}/pretrained_model/value_train_config.json
SMOKE_DIR=${OUT}/value_resume_smoke_4k_nw0_${SLURM_JOB_ID}
VALUE_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
LOCAL_TMP=/tmp/${USER}/recap_value_smoke_${SLURM_JOB_ID}

mkdir -p "${OUT}/logs" "${OUT}/cache" "${LOCAL_TMP}"
trap 'rm -rf "${LOCAL_TMP}"' EXIT
test -s "${RESUME_CKPT}/pretrained_model/model.safetensors"
test -s "${RESUME_CKPT}/training_state/optimizer_state.safetensors"
test -s "${RESUME_CONFIG}"
test ! -e "${SMOKE_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${LOCAL_TMP}
export PYTHONPATH="${RECAP_ROOT}/pydeps:${RECAP_ROOT}/lerobot_src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "resume_smoke_start=$(date --iso-8601=seconds)"
echo "resume_checkpoint=${RESUME_CKPT} steps=4000_to_4002 num_workers=0 tmpdir=${TMPDIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${VALUE_PY}" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_value_train \
  --config_path="${RESUME_CONFIG}" \
  --resume=true \
  --output_dir="${SMOKE_DIR}" \
  --steps=4002 \
  --num_workers=0 \
  --log_freq=1 \
  --save_checkpoint=true \
  --save_freq=2 \
  --wandb.enable=false

test -s "${SMOKE_DIR}/checkpoints/004002/pretrained_model/model.safetensors"
test -s "${SMOKE_DIR}/checkpoints/004002/training_state/optimizer_state.safetensors"
touch "${OUT}/VALUE_GEMMA3_SIGLIP2_8GPU_RESUME_SMOKE_COMPLETE"
echo "resume_smoke_complete=$(date --iso-8601=seconds) checkpoint=${SMOKE_DIR}/checkpoints/004002"
