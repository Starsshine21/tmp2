#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_value_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-smoke-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-02:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
DATA_ROOT=${OUT}/data_smoke/click_mouse
VALUE_DIR=${OUT}/value_smoke
VALUE_INFER_DIR=${OUT}/value_infer_smoke
VALUE_PY=/nfs_global/S/yangrongzheng/pi05/recap_workspace/.venv_recap311/bin/python

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp"
test -f "${OUT}/DATA_SMOKE_COMPLETE"
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

echo "value_smoke_start=$(date --iso-8601=seconds)"
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
  --batch_size=2 \
  --steps=10 \
  --num_workers=0 \
  --log_freq=1 \
  --save_checkpoint=true \
  --save_freq=10 \
  --wandb.enable=false \
  --output_dir="${VALUE_DIR}" \
  --job_name=click_mouse_recap_value_smoke

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
  --runtime.batch_size=4 \
  --runtime.num_workers=0 \
  --acp.enable=true \
  --acp.n_step=10 \
  --acp.positive_ratio=0.2 \
  --acp.c_fail_coef=1.0 \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --viz.enable=false \
  --output_dir="${VALUE_INFER_DIR}" \
  --job_name=click_mouse_recap_value_infer_smoke

"${VALUE_PY}" - "${DATA_ROOT}" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local_repo", root=sys.argv[1])
frames = dataset.hf_dataset.with_format(None)
required = {
    "complementary_info.value",
    "complementary_info.advantage",
    "complementary_info.acp_indicator",
}
missing = sorted(required.difference(frames.column_names))
if missing:
    raise RuntimeError(f"Missing ReCap labels after value inference: {missing}")
indicators = [int(value) for value in frames["complementary_info.acp_indicator"]]
counts = {value: indicators.count(value) for value in sorted(set(indicators))}
if not indicators or any(value not in {0, 1} for value in indicators):
    raise RuntimeError(f"Invalid ACP indicators: {counts}")
if set(counts) != {0, 1}:
    raise RuntimeError(f"Smoke dataset must contain both ACP tags, got {counts}")
print(f"value_smoke_labels frames={len(frames)} counts={counts}")
PY

touch "${OUT}/VALUE_SMOKE_COMPLETE"
echo "value_smoke_end=$(date --iso-8601=seconds)"
