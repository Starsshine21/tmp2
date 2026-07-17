#!/bin/bash
#SBATCH -J pi05_real_demo
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-real-demo-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-real-demo-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-00:20:00
#SBATCH --constraint=L40S|A100|L40

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/pi05
mkdir -p "$ROOT/logs"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-$ROOT/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_dexhand_full_lora_pytorch_32/pi05_pickplace_dexhand_eef_delta_train_lora/60000}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_pickplace_dexhand_full_lora_pytorch_32}
PROMPT=${PROMPT:-pick and place}
ROBOT_IP=${ROBOT_IP:-192.168.1.109}
HAND_PORT=${HAND_PORT:-/dev/ttyUSB0}
CONTROL_MODE=${CONTROL_MODE:-delta_eef}
EEF_DELTA_SCALE=${EEF_DELTA_SCALE:-1.0}
MAX_EEF_DELTA=${MAX_EEF_DELTA:-0.02}
MAX_HAND_ABS=${MAX_HAND_ABS:-2000}
ACTION_EMA_ALPHA=${ACTION_EMA_ALPHA:-0.20}
CONTROL_HZ=${CONTROL_HZ:-10}
RECORD_DIR=${RECORD_DIR:-$ROOT/real_robot_demos}
RECORD_FPS=${RECORD_FPS:-10}
MAX_STEPS=${MAX_STEPS:-120}

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate $ROOT/.conda-pi05-openpi-final
source "$ROOT/scripts/use_local_openpi_env.sh"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$ROOT/openpi_official/src:${PYTHONPATH:-}"

cd "$ROOT"
python "$ROOT/scripts/pi05_real_robot_infer.py" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --train-config "$TRAIN_CONFIG" \
  --prompt "$PROMPT" \
  --robot-ip "$ROBOT_IP" \
  --hand-port "$HAND_PORT" \
  --control-hz "$CONTROL_HZ" \
  --control-mode "$CONTROL_MODE" \
  --eef-delta-scale "$EEF_DELTA_SCALE" \
  --max-eef-delta "$MAX_EEF_DELTA" \
  --max-hand-abs "$MAX_HAND_ABS" \
  --action-ema-alpha "$ACTION_EMA_ALPHA" \
  --record-dir "$RECORD_DIR" \
  --record-fps "$RECORD_FPS" \
  --max-steps "$MAX_STEPS"
