#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/nfs_global/S/yangrongzheng/evo-RL/dexjoco}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${ROOT}/outputs/ogpo/checkpoints/click_mouse_pi05_jax_flash_td10_tgr_b8_8gpu_accel_resume450_10000/step_3000}
CHECKPOINT=${CHECKPOINT:-${CHECKPOINT_DIR}/click_mouse_pi05_jax_flash_td10_tgr_b8_8gpu_accel_resume450_10000_final.pt}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/scripts/pi05_jax_ogpo_eval_100.slurm}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-${ROOT}/outputs/pi05_rollouts/click_mouse_tgr_8gpu_step3000_seed27_100ep}
STATE_DIR=${STATE_DIR:-${ROOT}/outputs/ogpo/watchdog}
STATE_FILE=${STATE_DIR}/pi05_tgr_8gpu_step3000_seed27_eval100.submitted
LOCK_FILE=${STATE_DIR}/pi05_tgr_8gpu_step3000_seed27_eval100.lock
POLL_SECONDS=${POLL_SECONDS:-30}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-172800}

mkdir -p "${STATE_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another watchdog holds ${LOCK_FILE}; exiting."
  exit 0
fi

if [[ -s "${STATE_FILE}" ]]; then
  echo "Evaluation was already submitted: $(cat "${STATE_FILE}")"
  exit 0
fi

checkpoint_is_complete() {
  [[ -s "${CHECKPOINT}" ]] \
    && [[ -s "${CHECKPOINT}.jax/_CHECKPOINT_METADATA" ]] \
    && [[ -s "${CHECKPOINT}.jax/manifest.ocdbt" ]]
}

started_at=$(date +%s)
echo "Watching for complete step-3000 checkpoint: ${CHECKPOINT}"
while ! checkpoint_is_complete; do
  now=$(date +%s)
  if (( now - started_at >= TIMEOUT_SECONDS )); then
    echo "Timed out after ${TIMEOUT_SECONDS}s waiting for step 3000." >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

echo "Submitting 100-episode seed-27 evaluation for ${CHECKPOINT}."
eval_job_id=$(
  sbatch --parsable \
    --job-name=pi05_tgr3000_eval100 \
    --export="ALL,OGPO_CHECKPOINT=${CHECKPOINT},PI05_EPISODES=100,PI05_SEED=27,PI05_EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR},PI05_STRICT_REPRODUCIBILITY=true" \
    "${EVAL_SCRIPT}"
)
printf '%s\n' "${eval_job_id}" >"${STATE_FILE}"

echo "Submitted evaluation job ${eval_job_id}."
echo "Evaluation output: ${EVAL_OUTPUT_DIR}"
