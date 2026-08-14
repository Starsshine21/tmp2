#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_value64k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value64k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value64k-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH -t 0-08:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
DATA_ROOT=${OUT}/data/click_mouse
VALUE_DIR=${OUT}/value_pistar06_64k_b8_seed27_v5
VALUE_INFER_DIR=${OUT}/value_infer_pistar06_64k_acp30_v5
VALUE_PY=/nfs_global/S/yangrongzheng/pi05/recap_workspace/.venv_recap311/bin/python

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp"
test -f "${OUT}/DATA_COMPLETE"
test -f "${DATA_ROOT}/meta/info.json"
test -x "${VALUE_PY}"
test ! -e "${VALUE_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${OUT}/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${RECAP_ROOT}/pydeps:${RECAP_ROOT}/lerobot_src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "value64k_start=$(date --iso-8601=seconds)"
echo "dataset=${DATA_ROOT} batch=8 steps=64000 total_samples=512000"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${VALUE_PY}" -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local_repo \
  --dataset.root="${DATA_ROOT}" \
  --dataset.revision=v3.0 \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.vision_repo_id="${ROOT}/Evo-RL/local_models/siglip2-so400m-patch14-224-fixed" \
  --value.language_repo_id="${ROOT}/Evo-RL/local_models/gemma-3-270m" \
  --value.task_field=task \
  --value.camera_features='[observation.images.ego_right,observation.images.wrist]' \
  --value.state_feature=observation.state \
  --value.normalization_mapping='{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: IDENTITY}' \
  --value.device=cuda \
  --value.push_to_hub=false \
  --value.scheduler_warmup_steps=4000 \
  --value.scheduler_decay_steps=64000 \
  --batch_size=8 \
  --steps=64000 \
  --num_workers=2 \
  --log_freq=200 \
  --save_checkpoint=true \
  --save_freq=8000 \
  --wandb.enable=false \
  --seed=27 \
  --output_dir="${VALUE_DIR}" \
  --job_name=click_mouse_recap_value_64k_b8

"${VALUE_PY}" -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=local_repo \
  --dataset.root="${DATA_ROOT}" \
  --dataset.revision=v3.0 \
  --dataset.download_videos=false \
  --dataset.success_field=episode_success \
  --dataset.default_success=failure \
  --inference.checkpoint_path="${VALUE_DIR}" \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size=16 \
  --runtime.num_workers=2 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.c_fail_coef=1.0 \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --viz.enable=false \
  --output_dir="${VALUE_INFER_DIR}" \
  --job_name=click_mouse_recap_value_64k_acp30

"${VALUE_PY}" - "${DATA_ROOT}" <<'PY'
import sys
from collections import Counter
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local_repo", root=sys.argv[1])
frames = dataset.hf_dataset.with_format(None)
indicators = [int(value) for value in frames["complementary_info.acp_indicator"]]
counts = Counter(indicators)
if set(counts) != {0, 1}:
    raise RuntimeError(f"Expected binary ACP indicators, got {dict(counts)}")
ratio = counts[1] / len(indicators)
if not 0.299 <= ratio <= 0.301:
    raise RuntimeError(f"Expected ACP positive ratio 0.3, got {ratio:.6f}")
print(f"value64k_labeling_complete frames={len(indicators)} counts={dict(counts)} ratio={ratio:.6f}")
PY

touch "${OUT}/VALUE_64K_ACP30_COMPLETE"
echo "value64k_end=$(date --iso-8601=seconds)"
