#!/bin/bash
#SBATCH -J pi05_adj_filt
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/Evo-RL/logs/pi05-adj-filt-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/Evo-RL/logs/pi05-adj-filt-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 0-08:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/Evo-RL
PI05_ROOT=/nfs_global/S/yangrongzheng/pi05
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/.hf_cache/datasets" "$ROOT/outputs"

echo "PI05 FILTERED TRAIN start $(date '+%F %T')"
echo "Host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate robosuite

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.hf_cache"
export HF_DATASETS_CACHE="$ROOT/.hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$ROOT/local_models/siglip2-so400m-patch14-224-fixed"
rsync -a --delete /nfs_global/S/yangrongzheng/pi05/models/google/siglip2-so400m-patch14-224/ "$ROOT/local_models/siglip2-so400m-patch14-224-fixed/"
cp "$ROOT/local_models/siglip2-so400m-patch14-224-fixed/perprocessor_config.json" \
   "$ROOT/local_models/siglip2-so400m-patch14-224-fixed/preprocessor_config.json"

POLICY_OUTPUT_DIR="${POLICY_OUTPUT_DIR:-$ROOT/outputs/pi05_adjust_bottle_filtered_${SLURM_JOB_ID:-manual}}"
POLICY_JOB_NAME="${POLICY_JOB_NAME:-pi05_adjust_bottle_filtered_${SLURM_JOB_ID:-manual}}"
POLICY_STEPS="${POLICY_STEPS:-20000}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-8}"
POLICY_NUM_WORKERS="${POLICY_NUM_WORKERS:-0}"

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=ignored_local \
  --dataset.root=local_data/lerobot_adjust_bottle_filtered \
  --dataset.revision=v3.0 \
  --dataset.image_transforms.enable=true \
  --policy.type=pi05 \
  --policy.device=cuda \
  --policy.repo_id=$PI05_ROOT/model_robotwin \
  --policy.task_field=task \
  --policy.camera_features=[observation.images.front,observation.images.wrist] \
  --policy.state_feature=observation.state \
  --batch_size=$POLICY_BATCH_SIZE \
  --steps=$POLICY_STEPS \
  --num_workers=$POLICY_NUM_WORKERS \
  --save_checkpoint=true \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir=$POLICY_OUTPUT_DIR \
  --job_name=$POLICY_JOB_NAME

echo "PI05 FILTERED TRAIN end $(date '+%F %T')"
