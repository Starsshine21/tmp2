#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_v_g3s2_10k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-g3s2-8gpu-10k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-g3s2-8gpu-10k-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -t 0-12:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
DATA_ROOT=${OUT}/data/click_mouse
VALUE_DIR=${OUT}/value_pistar06_gemma3_270m_siglip2_224_10k_b32_8gpu_seed1000_oldcfg
INFER_ROOT=${OUT}/value_infer_gemma3_270m_siglip2_224_10k_b32_8gpu_seed1000_oldcfg
RESUME_CKPT=${VALUE_DIR}/checkpoints/004000
RESUME_CONFIG=${RESUME_CKPT}/pretrained_model/value_train_config.json
GEMMA_DIR=${ROOT}/Evo-RL/local_models/gemma-3-270m
SIGLIP_DIR=${ROOT}/Evo-RL/local_models/siglip2-so400m-patch14-224-fixed
VALUE_PY=${ROOT}/dexjoco/.conda/openpi/bin/python
LOCAL_TMP=/tmp/${USER}/recap_value_${SLURM_JOB_ID}

mkdir -p "${OUT}/logs" "${OUT}/cache" "${LOCAL_TMP}"
trap 'rm -rf "${LOCAL_TMP}"' EXIT
test -f "${OUT}/DATA_COMPLETE"
test -f "${OUT}/VALUE_GEMMA3_SIGLIP2_8GPU_SMOKE_COMPLETE"
test -f "${DATA_ROOT}/meta/info.json"
test -f "${GEMMA_DIR}/model.safetensors"
test -f "${SIGLIP_DIR}/model.safetensors"
test -x "${VALUE_PY}"
test -s "${RESUME_CKPT}/pretrained_model/model.safetensors"
test -s "${RESUME_CKPT}/training_state/optimizer_state.safetensors"
test -s "${RESUME_CONFIG}"
test ! -e "${INFER_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${LOCAL_TMP}
export PYTHONPATH="${RECAP_ROOT}/pydeps:${RECAP_ROOT}/lerobot_src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "formal_resume_start=$(date --iso-8601=seconds)"
echo "github_baseline=95f8b6cd3da782b84c41112c91db5c73fa60f705"
echo "critic=Gemma-3-270M+SigLIP2-So400m-patch14-224 full_train=true"
echo "world_size=8 per_gpu_batch=4 global_batch=32 steps=10000 seed=1000"
echo "optimizer=AdamW lr=5e-5 weight_decay=1e-5 warmup=500 decay_steps=8000 decay_lr=1e-6"
echo "save_freq=2000 acp_n_step=50 acp_positive_ratio=0.3"
echo "resume_checkpoint=${RESUME_CKPT} num_workers=0 tmpdir=${TMPDIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${VALUE_PY}" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_value_train \
  --config_path="${RESUME_CONFIG}" \
  --resume=true \
  --output_dir="${VALUE_DIR}" \
  --steps=10000 \
  --num_workers=0 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --wandb.enable=false

export CUDA_VISIBLE_DEVICES=0
export LEROBOT_VALUE_INFER_DRY_RUN=1
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
    --inference.checkpoint_path="${VALUE_DIR}/checkpoints/${step_ref}/pretrained_model" \
    --inference.checkpoint_ref=last \
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
    --job_name="click_mouse_gemma3_siglip2_infer_${step_ref}"
done

test -s "${INFER_ROOT}/010000/value/acp_annotations.npz"
touch "${OUT}/VALUE_GEMMA3_SIGLIP2_10K_B32_8GPU_ACP30_COMPLETE"
echo "formal_checkpoint=${VALUE_DIR}/checkpoints/010000/pretrained_model"
echo "formal_sidecar=${INFER_ROOT}/010000/value/acp_annotations.npz"
echo "formal_complete=$(date --iso-8601=seconds)"
