#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J watch_recap_jax20k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/watch-recap-jax20k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/watch-recap-jax20k-%j.err
#SBATCH -p cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 0-02:00:00
#SBATCH --no-requeue

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
OUT=${ROOT}/outputs/recap-click-mouse
CHECKPOINT=${OUT}/actor_jax_oldcfg_checkpoints/click_mouse/recap_click_mouse_gemma3value10k_100ep_oldcfg/20000
EVAL_SCRIPT=${ROOT}/recap_click_mouse/eval_actor_jax_oldcfg_final_seed27_100ep.slurm
JOB_ID_FILE=${OUT}/eval/jax_oldcfg_step20000_seed27_100ep_job_id.txt

mkdir -p "${OUT}/logs" "${OUT}/eval"
echo "watch_start=$(date --iso-8601=seconds)"
echo "checkpoint=${CHECKPOINT}"

while [[ ! -s "${CHECKPOINT}/params/_METADATA" || ! -s "${CHECKPOINT}/params/manifest.ocdbt" ]]; do
  sleep 30
done

echo "checkpoint_ready=$(date --iso-8601=seconds)"
EVAL_JOB_ID=$(sbatch --parsable --export=ALL,RECAP_EVAL_STEP=20000 "${EVAL_SCRIPT}")
printf '%s\n' "${EVAL_JOB_ID}" >"${JOB_ID_FILE}"
echo "evaluation_job_id=${EVAL_JOB_ID}"
echo "watch_end=$(date --iso-8601=seconds)"
