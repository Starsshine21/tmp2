#!/bin/bash
#SBATCH -J value_adj_bottle
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/Evo-RL/logs/value-adj-bottle-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/Evo-RL/logs/value-adj-bottle-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1-4:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/Evo-RL
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/.hf_cache/datasets" "$ROOT/outputs"

echo "VALUE TRAIN start $(date '+%F %T')"
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

# Build a writable local mirror for SigLIP2 with the filename transformers expects.
mkdir -p "$ROOT/local_models/siglip2-so400m-patch14-224-fixed"
rsync -a --delete /nfs_global/S/yangrongzheng/pi05/models/google/siglip2-so400m-patch14-224/ "$ROOT/local_models/siglip2-so400m-patch14-224-fixed/"
cp "$ROOT/local_models/siglip2-so400m-patch14-224-fixed/perprocessor_config.json" \
   "$ROOT/local_models/siglip2-so400m-patch14-224-fixed/preprocessor_config.json"

# Rebuild the local value dataset to keep metadata/stats in sync.
python scripts/prepare_adjust_bottle_value_dataset.py

VALUE_STEPS="${VALUE_STEPS:-200000}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-8}"
VALUE_NUM_WORKERS="${VALUE_NUM_WORKERS:-0}"
VALUE_OUTPUT_DIR="${VALUE_OUTPUT_DIR:-$ROOT/outputs/value_train_adjust_bottle_${SLURM_JOB_ID:-manual}}"
VALUE_JOB_NAME="${VALUE_JOB_NAME:-adjust_bottle_value_${SLURM_JOB_ID:-manual}}"

python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=ignored_local \
  --dataset.root=local_data/lerobot_adjust_bottle_value \
  --dataset.revision=v3.0 \
  --value.type=pistar06 \
  --value.vision_repo_id=local_models/siglip2-so400m-patch14-224-fixed \
  --value.language_repo_id=local_models/gemma-3-270m \
  --value.task_field=task \
  --value.camera_features=[observation.images.front,observation.images.wrist] \
  --value.state_feature=observation.state \
  --batch_size="$VALUE_BATCH_SIZE" \
  --steps="$VALUE_STEPS" \
  --num_workers="$VALUE_NUM_WORKERS" \
  --log_freq=500 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --wandb.enable=false \
  --output_dir="$VALUE_OUTPUT_DIR" \
  --job_name="$VALUE_JOB_NAME"

echo "VALUE TRAIN end $(date '+%F %T')"
