#!/bin/bash
#SBATCH -J value_infer_100k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/Evo-RL/logs/value-infer-100k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/Evo-RL/logs/value-infer-100k-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/Evo-RL
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/.hf_cache/datasets" "$ROOT/outputs"

echo "VALUE INFER 100K start $(date '+%F %T')"
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

# Keep the SigLIP2 compatibility mirror consistent.
mkdir -p "$ROOT/local_models/siglip2-so400m-patch14-224-fixed"
rsync -a --delete /nfs_global/S/yangrongzheng/pi05/models/google/siglip2-so400m-patch14-224/ "$ROOT/local_models/siglip2-so400m-patch14-224-fixed/"
cp "$ROOT/local_models/siglip2-so400m-patch14-224-fixed/perprocessor_config.json" \
   "$ROOT/local_models/siglip2-so400m-patch14-224-fixed/preprocessor_config.json"

VALUE_INFER_OUTPUT_DIR="${VALUE_INFER_OUTPUT_DIR:-$ROOT/outputs/value_infer_adjust_bottle_100000_${SLURM_JOB_ID:-manual}}"
VALUE_INFER_BATCH_SIZE="${VALUE_INFER_BATCH_SIZE:-32}"
VALUE_INFER_NUM_WORKERS="${VALUE_INFER_NUM_WORKERS:-0}"
VALUE_CKPT_PATH="${VALUE_CKPT_PATH:-$ROOT/outputs/value_train_adjust_bottle_763533/checkpoints/100000}"

python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=ignored_local \
  --dataset.root=local_data/lerobot_adjust_bottle_value \
  --dataset.revision=v3.0 \
  --dataset.download_videos=false \
  --dataset.success_field=episode_success \
  --dataset.default_success=failure \
  --inference.checkpoint_path="$VALUE_CKPT_PATH" \
  --inference.checkpoint_ref=100000 \
  --runtime.device=cuda \
  --runtime.batch_size="$VALUE_INFER_BATCH_SIZE" \
  --runtime.num_workers="$VALUE_INFER_NUM_WORKERS" \
  --acp.enable=true \
  --viz.enable=false \
  --output_dir="$VALUE_INFER_OUTPUT_DIR"

echo "VALUE INFER 100K end $(date '+%F %T')"
