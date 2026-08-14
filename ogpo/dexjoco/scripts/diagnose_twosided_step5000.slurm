#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J ogpo_twoside_diag
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/twoside-step5000-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/twoside-step5000-%j.err
#SBATCH -p r8nv-gpu-hw-80g
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --nodelist=r8a100-d04
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH -t 02:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
OPENPI_ENV=${ROOT}/.conda/openpi
CONFIG=configs/ogpo/pi05_jax_flash_ogpo_td10_tgr_b8_8gpu_trueppo2_gn200_resume3000_10000.yaml
CHECKPOINT=outputs/ogpo/checkpoints/click_mouse_pi05_jax_flash_td10_tgr_b8_8gpu_trueppo2_gn200_resume3000_10000/step_5000/click_mouse_pi05_jax_flash_td10_tgr_b8_8gpu_trueppo2_gn200_resume3000_10000_final.pt
OUTPUT=outputs/ogpo/diagnostics/twoside_step5000_seed23.json

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
ulimit -v unlimited 2>/dev/null || true

mkdir -p "${ROOT}/outputs/ogpo/logs" "${ROOT}/outputs/ogpo/diagnostics"
cd "${ROOT}"
srun \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --gpus-per-task=1 \
  --gpu-bind=closest \
  "${OPENPI_ENV}/bin/python" scripts/diagnose_twosided_baselines.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --states-per-split 32 \
    --group-size 4 \
    --microbatch-size 8 \
    --seed 23
