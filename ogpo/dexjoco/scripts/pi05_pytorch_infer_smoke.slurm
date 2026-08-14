#!/bin/bash
#SBATCH -J pi05_pt_infer
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-infer-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-infer-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-00:20:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${ROOT}/checkpoints/pi05_pytorch/click_mouse}
LOCAL_CHECKPOINT=${SLURM_TMPDIR:-/tmp}/pi05-pytorch-click-mouse-${SLURM_JOB_ID}

mkdir -p "${ROOT}/outputs/ogpo/logs" "${LOCAL_CHECKPOINT}"
rsync -a "${SOURCE_CHECKPOINT}/" "${LOCAL_CHECKPOINT}/"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "infer_start=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${OPENPI_ENV}/bin/python" "${ROOT}/scripts/pi05_pytorch_infer_smoke.py" \
  --checkpoint "${LOCAL_CHECKPOINT}" \
  --config click_mouse \
  --device cuda \
  --num-steps 2 \
  --repeats 2

echo "infer_end=$(date --iso-8601=seconds)"
