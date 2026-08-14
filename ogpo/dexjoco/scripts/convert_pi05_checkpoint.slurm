#!/bin/bash
#SBATCH -J dexjoco_pi05_convert
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-convert-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-convert-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 0-01:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse}
OUTPUT_CHECKPOINT=${OUTPUT_CHECKPOINT:-${ROOT}/checkpoints/pi05_pytorch/click_mouse}
OPENPI_ENV=${OPENPI_ENV:-/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final}
TRAIN_CONFIG=${TRAIN_CONFIG:-click_mouse}
LOCAL_ROOT=${SLURM_TMPDIR:-/tmp}/pi05-click-mouse-${SLURM_JOB_ID}

mkdir -p "${ROOT}/outputs/ogpo/logs" "${OUTPUT_CHECKPOINT}" "${LOCAL_ROOT}"

echo "conversion_start=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "source=${SOURCE_CHECKPOINT}"
echo "local=${LOCAL_ROOT}"
echo "output=${OUTPUT_CHECKPOINT}"

rsync -a "${SOURCE_CHECKPOINT}/" "${LOCAL_ROOT}/"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/openpi/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

"${OPENPI_ENV}/bin/python" "${ROOT}/openpi/examples/convert_jax_model_to_pytorch.py" \
  --checkpoint-dir "${LOCAL_ROOT}" \
  --config-name "${TRAIN_CONFIG}" \
  --output-path "${OUTPUT_CHECKPOINT}" \
  --precision bfloat16

"${OPENPI_ENV}/bin/python" -c \
  'import pathlib,sys; from safetensors import safe_open; p=pathlib.Path(sys.argv[1]); f=safe_open(p,framework="pt",device="cpu"); print(f"safetensors={p} bytes={p.stat().st_size} keys={len(f.keys())}")' \
  "${OUTPUT_CHECKPOINT}/model.safetensors"

echo "conversion_end=$(date --iso-8601=seconds)"
