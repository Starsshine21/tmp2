#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/nfs_global/S/yangrongzheng/evo-RL/dexjoco}
TRAIN_JOB_ID=${TRAIN_JOB_ID:-787418}
TRAIN_LOG=${TRAIN_LOG:-${ROOT}/outputs/ogpo/logs/pi05-flash-train-${TRAIN_JOB_ID}.out}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${ROOT}/outputs/ogpo/click_mouse_pi05_jax_flash_100ep_fast.pt}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-${ROOT}/outputs/ogpo/checkpoints/click_mouse_pi05_jax_flash_fast_step_0050}
SNAPSHOT_CHECKPOINT=${SNAPSHOT_DIR}/$(basename "${SOURCE_CHECKPOINT}")
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/scripts/pi05_jax_ogpo_eval_100.slurm}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-${ROOT}/outputs/pi05_rollouts/click_mouse_jax_ogpo_step0050_100ep}
EVAL_SERVER_PORT=${EVAL_SERVER_PORT:-19650}
STATE_DIR=${STATE_DIR:-${ROOT}/outputs/ogpo/watchdog}
POLL_SECONDS=${POLL_SECONDS:-30}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-43200}

# Training steps are zero-indexed in the log. step=49 is the checkpoint after
# 50 optimizer updates.
READY_PATTERN='[flash] periodic checkpoint saved at step=49'
STATE_FILE=${STATE_DIR}/pi05_step0050_eval100.submitted
LOCK_DIR=${STATE_DIR}/pi05_step0050_eval100.lock

mkdir -p "${STATE_DIR}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another step-50 watchdog holds ${LOCK_DIR}; exiting."
  exit 0
fi
cleanup_lock() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup_lock EXIT

if [[ -s "${STATE_FILE}" ]]; then
  echo "Evaluation was already submitted: $(cat "${STATE_FILE}")"
  exit 0
fi

checkpoint_is_complete() {
  [[ -s "${SOURCE_CHECKPOINT}" ]] \
    && [[ -s "${SOURCE_CHECKPOINT}.jax/_CHECKPOINT_METADATA" ]] \
    && [[ -s "${SOURCE_CHECKPOINT}.jax/manifest.ocdbt" ]]
}

started_at=$(date +%s)
echo "Watching ${TRAIN_LOG} for the completed 50-update checkpoint."
while ! grep -Fq "${READY_PATTERN}" "${TRAIN_LOG}" 2>/dev/null || ! checkpoint_is_complete; do
  now=$(date +%s)
  if (( now - started_at >= TIMEOUT_SECONDS )); then
    echo "Timed out after ${TIMEOUT_SECONDS}s waiting for step 50." >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

if [[ ! -s "${SNAPSHOT_CHECKPOINT}" \
   || ! -s "${SNAPSHOT_CHECKPOINT}.jax/_CHECKPOINT_METADATA" \
   || ! -s "${SNAPSHOT_CHECKPOINT}.jax/manifest.ocdbt" ]]; then
  snapshot_parent=$(dirname "${SNAPSHOT_DIR}")
  snapshot_name=$(basename "${SNAPSHOT_DIR}")
  temporary_dir=${snapshot_parent}/.${snapshot_name}.tmp.$$
  mkdir -p "${temporary_dir}"

  echo "Freezing step 50 at ${SNAPSHOT_DIR}."
  rsync -a -- "${SOURCE_CHECKPOINT}" "${temporary_dir}/$(basename "${SOURCE_CHECKPOINT}")"
  rsync -a -- "${SOURCE_CHECKPOINT}.jax/" "${temporary_dir}/$(basename "${SOURCE_CHECKPOINT}").jax/"

  test -s "${temporary_dir}/$(basename "${SOURCE_CHECKPOINT}")"
  test -s "${temporary_dir}/$(basename "${SOURCE_CHECKPOINT}").jax/_CHECKPOINT_METADATA"
  test -s "${temporary_dir}/$(basename "${SOURCE_CHECKPOINT}").jax/manifest.ocdbt"
  mv "${temporary_dir}" "${SNAPSHOT_DIR}"
  trap cleanup_lock EXIT
else
  echo "Using existing complete snapshot ${SNAPSHOT_DIR}."
fi

echo "Submitting 100-episode evaluation for ${SNAPSHOT_CHECKPOINT}."
eval_job_id=$(
  sbatch --parsable \
    --export="ALL,OGPO_CHECKPOINT=${SNAPSHOT_CHECKPOINT},PI05_EPISODES=100,PI05_EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR},PI05_SERVER_PORT=${EVAL_SERVER_PORT}" \
    "${EVAL_SCRIPT}"
)
printf '%s\n' "${eval_job_id}" >"${STATE_FILE}"

echo "Submitted evaluation job ${eval_job_id}."
echo "State file: ${STATE_FILE}"
echo "Evaluation output: ${EVAL_OUTPUT_DIR}"
echo "Server port: ${EVAL_SERVER_PORT}"
