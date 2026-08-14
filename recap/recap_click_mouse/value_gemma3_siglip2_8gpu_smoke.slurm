#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_v_g3s2_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-g3s2-8gpu-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-g3s2-8gpu-smoke-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -t 0-02:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
DATA_ROOT=${OUT}/data_smoke/click_mouse
VALUE_DIR=${OUT}/value_smoke_gemma3_270m_siglip2_224_b32_8gpu_oldcfg
INFER_DIR=${OUT}/value_infer_smoke_gemma3_270m_siglip2_224_b32_8gpu_oldcfg
GEMMA_DIR=${ROOT}/Evo-RL/local_models/gemma-3-270m
SIGLIP_DIR=${ROOT}/Evo-RL/local_models/siglip2-so400m-patch14-224-fixed
VALUE_PY=${ROOT}/dexjoco/.conda/openpi/bin/python

mkdir -p "${OUT}/logs" "${OUT}/cache" "${OUT}/tmp"
test -f "${OUT}/DATA_SMOKE_COMPLETE"
test -f "${DATA_ROOT}/meta/info.json"
test -f "${GEMMA_DIR}/model.safetensors"
test -f "${SIGLIP_DIR}/model.safetensors"
test -x "${VALUE_PY}"
test ! -e "${VALUE_DIR}"
test ! -e "${INFER_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${OUT}/tmp
export PYTHONPATH="${RECAP_ROOT}/pydeps:${RECAP_ROOT}/lerobot_src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "smoke_start=$(date --iso-8601=seconds)"
echo "critic=Gemma-3-270M+SigLIP2-So400m-patch14-224 full_train=true"
echo "world_size=8 per_gpu_batch=4 global_batch=32 steps=5 seed=1000"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${VALUE_PY}" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local_repo \
  --dataset.root="${DATA_ROOT}" \
  --dataset.revision=v3.0 \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.vision_repo_id="${SIGLIP_DIR}" \
  --value.language_repo_id="${GEMMA_DIR}" \
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
  --batch_size=4 \
  --steps=5 \
  --num_workers=1 \
  --log_freq=1 \
  --save_checkpoint=true \
  --save_freq=5 \
  --wandb.enable=false \
  --seed=1000 \
  --output_dir="${VALUE_DIR}" \
  --job_name=click_mouse_gemma3_siglip2_8gpu_smoke

test -s "${VALUE_DIR}/checkpoints/000005/pretrained_model/model.safetensors"
test -s "${VALUE_DIR}/checkpoints/000005/training_state/optimizer_state.safetensors"

export CUDA_VISIBLE_DEVICES=0
export LEROBOT_VALUE_INFER_DRY_RUN=1
"${VALUE_PY}" -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=local_repo \
  --dataset.root="${DATA_ROOT}" \
  --dataset.revision=v3.0 \
  --dataset.download_videos=false \
  --dataset.success_field=episode_success \
  --dataset.default_success=failure \
  --inference.checkpoint_path="${VALUE_DIR}/checkpoints/000005/pretrained_model" \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size=16 \
  --runtime.num_workers=1 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.force_intervention_positive=true \
  --acp.c_fail_coef=1.0 \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --viz.enable=false \
  --output_dir="${INFER_DIR}" \
  --job_name=click_mouse_gemma3_siglip2_8gpu_smoke_infer

test -s "${INFER_DIR}/value/acp_annotations.npz"
touch "${OUT}/VALUE_GEMMA3_SIGLIP2_8GPU_SMOKE_COMPLETE"
echo "smoke_complete=$(date --iso-8601=seconds)"
