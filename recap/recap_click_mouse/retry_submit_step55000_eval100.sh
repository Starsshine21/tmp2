#!/bin/bash
set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
OUT=${ROOT}/outputs/recap-click-mouse
SCRIPT=${ROOT}/recap_click_mouse/eval_actor_corrected_step55000_seed27_100ep.slurm
JOB_FILE=${OUT}/eval/corrected_mergedbase_value8k_step55000_seed27_100ep_job_id.txt
LOG=${OUT}/logs/retry-submit-step55000-eval100.log

mkdir -p "${OUT}/logs" "${OUT}/eval"
if [[ -s "${JOB_FILE}" ]]; then
  exit 0
fi

echo "retry_start=$(date --iso-8601=seconds)" >>"${LOG}"
while true; do
  if JOB_ID=$(sbatch --parsable "${SCRIPT}" 2>>"${LOG}"); then
    printf '%s\n' "${JOB_ID}" >"${JOB_FILE}"
    echo "evaluation_job_id=${JOB_ID} submitted=$(date --iso-8601=seconds)" >>"${LOG}"
    exit 0
  fi
  sleep 30
done
