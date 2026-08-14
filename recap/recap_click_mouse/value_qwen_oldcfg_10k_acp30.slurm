#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_v_qwen_old
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-qwen-oldcfg-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-qwen-oldcfg-%j.err
#SBATCH -p r8nv-gpu-hw-80g
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH -t 1-06:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
REFERENCE_ROOT=${RECAP_ROOT}/reference_evorl_95f8b6c
OUT=${ROOT}/outputs/recap-click-mouse
DATA_ROOT=${OUT}/data/click_mouse
MODEL_ROOT=${ROOT}/Evo-RL/local_models
LANGUAGE_MODEL_DIR=${MODEL_ROOT}/qwen2.5-0.5b
VISION_MODEL_DIR=${ROOT}/dexjoco/checkpoints/lwd_critic/siglip-so400m-patch14-384
VALUE_DIR=${OUT}/value_pistar06_qwen05b_siglip384_10k_b16_seed1000_oldcfg
INFER_ROOT=${OUT}/value_infer_qwen05b_siglip384_10k_b16_seed1000_oldcfg
VALUE_PY=${ROOT}/dexjoco/.conda/openpi/bin/python

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp"
test -f "${OUT}/DATA_COMPLETE"
test -f "${OUT}/OLDCFG_BACKBONES_COMPLETE"
test -f "${DATA_ROOT}/meta/info.json"
test -f "${LANGUAGE_MODEL_DIR}/model.safetensors"
test -f "${VISION_MODEL_DIR}/model.safetensors"
test -x "${VALUE_PY}"
test ! -e "${VALUE_DIR}"
test ! -e "${INFER_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${OUT}/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${RECAP_ROOT}/pydeps:${REFERENCE_ROOT}/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "value_qwen_oldcfg_start=$(date --iso-8601=seconds)"
echo "reference_commit=95f8b6cd3da782b84c41112c91db5c73fa60f705"
echo "dataset=${DATA_ROOT} episodes=100 batch=16 steps=10000 seed=1000 visible_gpus=1"
echo "vision=siglip-so400m-patch14-384 language=Qwen2.5-0.5B full_train=true"
echo "optimizer=AdamW peak_lr=5e-5 warmup_steps=500 decay_steps=8000 decay_lr=1e-6"
echo "save_freq=2000 acp_n_step=50 acp_positive_ratio=0.3"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${VALUE_PY}" -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local_repo \
  --dataset.root="${DATA_ROOT}" \
  --dataset.revision=v3.0 \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.vision_repo_id="${VISION_MODEL_DIR}" \
  --value.language_repo_id="${LANGUAGE_MODEL_DIR}" \
  --value.task_field=task \
  --value.camera_features='[observation.images.ego_right,observation.images.wrist]' \
  --value.state_feature=observation.state \
  --value.normalization_mapping='{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: IDENTITY}' \
  --value.device=cuda \
  --value.push_to_hub=false \
  --value.freeze_vision_encoder=false \
  --value.freeze_language_model=false \
  --value.optimizer_lr=5e-5 \
  --value.optimizer_weight_decay=1e-5 \
  --value.scheduler_warmup_steps=500 \
  --value.scheduler_decay_steps=8000 \
  --value.scheduler_decay_lr=1e-6 \
  --batch_size=16 \
  --steps=10000 \
  --num_workers=4 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --wandb.enable=false \
  --seed=1000 \
  --output_dir="${VALUE_DIR}" \
  --job_name=click_mouse_pistar06_qwen05b_siglip384_10k_b16_oldcfg

for step in 2000 4000 6000 8000 10000; do
  step_ref=$(printf '%06d' "${step}")
  test -d "${VALUE_DIR}/checkpoints/${step_ref}"
  infer_dir=${INFER_ROOT}/${step_ref}

  "${VALUE_PY}" -m lerobot.scripts.lerobot_value_infer \
    --dataset.repo_id=local_repo \
    --dataset.root="${DATA_ROOT}" \
    --dataset.revision=v3.0 \
    --dataset.download_videos=false \
    --dataset.success_field=episode_success \
    --dataset.default_success=failure \
    --inference.checkpoint_path="${VALUE_DIR}" \
    --inference.checkpoint_ref="${step}" \
    --runtime.device=cuda \
    --runtime.batch_size=64 \
    --runtime.num_workers=4 \
    --acp.enable=true \
    --acp.n_step=50 \
    --acp.positive_ratio=0.3 \
    --acp.force_intervention_positive=true \
    --acp.c_fail_coef=1.0 \
    --acp.value_field=complementary_info.value \
    --acp.advantage_field=complementary_info.advantage \
    --acp.indicator_field=complementary_info.acp_indicator \
    --viz.enable=false \
    --output_dir="${infer_dir}" \
    --job_name="click_mouse_qwen_oldcfg_infer_${step_ref}"
done

test -f "${INFER_ROOT}/010000/value/acp_annotations.npz"
touch "${OUT}/VALUE_QWEN_OLDCFG_10K_ACP30_COMPLETE"
echo "value_qwen_oldcfg_checkpoint=${VALUE_DIR}/checkpoints/010000/pretrained_model"
echo "value_qwen_oldcfg_sidecar=${INFER_ROOT}/010000/value/acp_annotations.npz"
echo "value_qwen_oldcfg_end=$(date --iso-8601=seconds)"
