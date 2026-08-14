#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J watch_recap30k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/watch-recap30k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/watch-recap30k-%j.err
#SBATCH -p r8cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 0-04:00:00
#SBATCH --no-requeue

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
OUT=${ROOT}/outputs/recap-click-mouse
CHECKPOINT=${OUT}/actor_full_pytorch_checkpoints/click_mouse_recap_full/recap_click_mouse_full_pytorch_mergedbase_value8k_acp30_b32_30k_seed27/30000
EVAL_SCRIPT=${ROOT}/recap_click_mouse/eval_actor_corrected_step30000_seed27_100ep.slurm
JOB_ID_FILE=${OUT}/eval/corrected_mergedbase_value8k_step30000_seed27_100ep_job_id.txt

mkdir -p "${OUT}/logs" "${OUT}/eval"
echo "watch_start=$(date --iso-8601=seconds)"
echo "checkpoint=${CHECKPOINT}"

while [[ ! -s "${CHECKPOINT}/model.safetensors" || ! -s "${CHECKPOINT}/assets/local_repo/norm_stats.json" ]]; do
  sleep 30
done

echo "checkpoint_ready=$(date --iso-8601=seconds)"
EVAL_JOB_ID=$(sbatch --parsable "${EVAL_SCRIPT}")
printf '%s\n' "${EVAL_JOB_ID}" >"${JOB_ID_FILE}"
echo "evaluation_job_id=${EVAL_JOB_ID}"
echo "watch_end=$(date --iso-8601=seconds)"
