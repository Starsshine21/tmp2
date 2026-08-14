#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_v_g3s2_sinf
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-g3s2-smoke-infer-retry-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-g3s2-smoke-infer-retry-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 0-01:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
DATA_ROOT=${OUT}/data_smoke/click_mouse
VALUE_DIR=${OUT}/value_smoke_gemma3_270m_siglip2_224_b32_8gpu_oldcfg
INFER_DIR=${OUT}/value_infer_smoke_gemma3_270m_siglip2_224_b32_8gpu_oldcfg_retry
VALUE_PY=${ROOT}/dexjoco/.conda/openpi/bin/python

test -s "${VALUE_DIR}/checkpoints/000005/pretrained_model/model.safetensors"
test -s "${VALUE_DIR}/checkpoints/000005/training_state/optimizer_state.safetensors"
test ! -e "${INFER_DIR}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${OUT}/tmp
export PYTHONPATH="${RECAP_ROOT}/pydeps:${RECAP_ROOT}/lerobot_src"
export LEROBOT_VALUE_INFER_DRY_RUN=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

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
  --job_name=click_mouse_gemma3_siglip2_8gpu_smoke_infer_retry

test -s "${INFER_DIR}/value/acp_annotations.npz"
touch "${OUT}/VALUE_GEMMA3_SIGLIP2_8GPU_SMOKE_COMPLETE"
echo "smoke_checkpoint_and_inference_complete=$(date --iso-8601=seconds)"
