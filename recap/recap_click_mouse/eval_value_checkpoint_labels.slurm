#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_value_audit
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-label-audit-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/value-label-audit-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40|L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 0-03:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
DATA=${OUT}/data/click_mouse
VALUE_RUN=${OUT}/value_pistar06_64k_b8_seed27_v5
VALUE_PY=/nfs_global/S/yangrongzheng/pi05/recap_workspace/.venv_recap311/bin/python
AUDIT_ROOT=${OUT}/value_checkpoint_label_audit

mkdir -p "${AUDIT_ROOT}" "${OUT}/cache" "${OUT}/tmp"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export TMPDIR=${OUT}/tmp
export PYTHONPATH="${RECAP_ROOT}/pydeps:${RECAP_ROOT}/lerobot_src"
export LEROBOT_VALUE_INFER_DRY_RUN=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

for STEP in 008000 016000 024000 032000 040000 048000 056000; do
  CHECKPOINT=${VALUE_RUN}/checkpoints/${STEP}/pretrained_model
  RESULT=${AUDIT_ROOT}/step_${STEP}
  test -s "${CHECKPOINT}/model.safetensors"
  "${VALUE_PY}" -m lerobot.scripts.lerobot_value_infer \
    --dataset.repo_id=local_repo \
    --dataset.root="${DATA}" \
    --dataset.revision=v3.0 \
    --dataset.download_videos=false \
    --dataset.success_field=episode_success \
    --dataset.default_success=failure \
    --inference.checkpoint_path="${CHECKPOINT}" \
    --runtime.device=cuda \
    --runtime.batch_size=32 \
    --runtime.num_workers=2 \
    --acp.enable=true \
    --acp.n_step=50 \
    --acp.positive_ratio=0.3 \
    --acp.c_fail_coef=1.0 \
    --viz.enable=false \
    --output_dir="${RESULT}" \
    --job_name="label_audit_${STEP}"
done

echo "audit_complete=${AUDIT_ROOT}"
