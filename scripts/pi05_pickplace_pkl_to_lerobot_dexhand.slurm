#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_pkl2lerobot_dexhand
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-pkl2lerobot-dexhand-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-pkl2lerobot-dexhand-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-12:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/pi05
INPUT_DIR="${PI05_PICKPLACE_PKL_INPUT_DIR:-/nfs_global/S/yangrongzheng/pick_place_raw_data}"
OUTPUT_DIR="${PI05_PICKPLACE_LEROBOT_OUTPUT_DIR:-$ROOT/data/lerobot_pick_place_dexhand}"
REPO_ID="${PI05_PICKPLACE_LEROBOT_REPO_ID:-local/pi05-pickplace-il}"
CONTROL_MODE="${PI05_PICKPLACE_CONTROL_MODE:-delta_eef}"
IMAGE_HEIGHT="${PI05_PICKPLACE_IMAGE_HEIGHT:-224}"
IMAGE_WIDTH="${PI05_PICKPLACE_IMAGE_WIDTH:-224}"
MAX_EPISODES="${PI05_PICKPLACE_MAX_EPISODES:-}"
STRIDE="${PI05_PICKPLACE_STRIDE:-1}"
STATE_MODE="${PI05_PICKPLACE_STATE_MODE:-joint_eef_hand}"

mkdir -p "$ROOT/logs" "$ROOT/data"
source /nfs_global/S/yangrongzheng/miniconda3-fresh/etc/profile.d/conda.sh
conda activate "$ROOT/.conda-pi05-openpi-final"

export PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

CMD=(python "$ROOT/scripts/convert_pick_place_to_lerobot.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repo-id "$REPO_ID" \
  --action-mode eef_delta \
  --control-mode "$CONTROL_MODE" \
  --state-mode "$STATE_MODE" \
  --stride "$STRIDE" \
  --image-height "$IMAGE_HEIGHT" \
  --image-width "$IMAGE_WIDTH" \
  --overwrite)

if [[ -n "$MAX_EPISODES" ]]; then
  CMD+=(--max-episodes "$MAX_EPISODES")
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
