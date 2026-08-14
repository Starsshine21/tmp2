#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_click_value
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH -t 2-00:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
DATA_ROOT=${OUT}/data/click_mouse
VALUE_DIR=${OUT}/value_pistar06_8k
VALUE_INFER_DIR=${OUT}/value_infer_pistar06_8k
VALUE_PY=/nfs_global/S/yangrongzheng/pi05/recap_workspace/.venv_recap311/bin/python

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp"
test -f "${OUT}/DATA_COMPLETE"
test -f "${OUT}/ACTOR_SMOKE_COMPLETE"
test -f "${DATA_ROOT}/meta/info.json"
test -x "${VALUE_PY}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${OUT}/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${RECAP_ROOT}/pydeps:${RECAP_ROOT}/lerobot_src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "recap_value_job_start=$(date --iso-8601=seconds)"
echo "dataset=${DATA_ROOT}"
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
  --batch_size=8 \
  --steps=8000 \
  --num_workers=2 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=8000 \
  --wandb.enable=false \
  --output_dir="${VALUE_DIR}" \
  --job_name=click_mouse_recap_value_8k

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
  --acp.positive_ratio=0.2 \
  --acp.c_fail_coef=1.0 \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --viz.enable=false \
  --output_dir="${VALUE_INFER_DIR}" \
  --job_name=click_mouse_recap_value_infer

"${VALUE_PY}" - "${DATA_ROOT}" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local_repo", root=sys.argv[1])
frames = dataset.hf_dataset.with_format(None)
field = "complementary_info.acp_indicator"
if field not in frames.column_names:
    raise RuntimeError(f"Missing {field} after value inference")
indicators = [int(value) for value in frames[field]]
counts = {value: indicators.count(value) for value in sorted(set(indicators))}
if set(counts) != {0, 1}:
    raise RuntimeError(f"Expected both ACP indicator values, got {counts}")
print(f"value_labeling_complete frames={len(frames)} counts={counts}")
PY

touch "${OUT}/VALUE_COMPLETE"
echo "recap_value_job_end=$(date --iso-8601=seconds)"
