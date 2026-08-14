#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_click_100ep
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-click-100ep-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-click-100ep-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH -t 0-04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
DEXJOCO_ENV=${DEXJOCO_ENV:-${ROOT}/.conda/dexjoco}
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}
DATASET_ENV=${DATASET_ENV:-/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final}
SOURCE_CHECKPOINT=${PI05_CHECKPOINT:-/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse}
EPISODES=${ROLLOUT_EPISODES:-100}
SEED=${PI05_SEED:-27}
PORT=${PI05_SERVER_PORT:-$((18000 + SLURM_JOB_ID % 1000))}
ROLLOUT_DIR=${ROLLOUT_OUTPUT_DIR:-${ROOT}/outputs/pi05_rollouts/click_mouse_jax_${EPISODES}ep_${SLURM_JOB_ID}}
LOCAL_CHECKPOINT=${SLURM_TMPDIR:-/tmp}/pi05-jax-click-mouse-${SLURM_JOB_ID}
SERVER_LOG=${ROOT}/outputs/ogpo/logs/pi05-click-100ep-server-${SLURM_JOB_ID}.log
DATASET_CONFIG=${OGPO_DATASET_CONFIG:-configs/ogpo/pi05_dataset_100ep.yaml}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_file "${DEXJOCO_ENV}/bin/python"
require_file "${OPENPI_ENV}/bin/python"
require_file "${DATASET_ENV}/bin/python"
require_file "${ROOT}/${DATASET_CONFIG}"
require_dir "${SOURCE_CHECKPOINT}/params"

mkdir -p "${ROOT}/outputs/ogpo/logs" "${ROLLOUT_DIR}" "${LOCAL_CHECKPOINT}"
rsync -a "${SOURCE_CHECKPOINT}/" "${LOCAL_CHECKPOINT}/"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "collection_start=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID} host=$(hostname) episodes=${EPISODES} seed=${SEED}"
echo "prompt=Move the mouse to the purple mouse pad and click the left mouse button."
echo "rollout_dir=${ROLLOUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${ROOT}/openpi"
PYTHONPATH="${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src" \
CUDA_VISIBLE_DEVICES=0 \
"${OPENPI_ENV}/bin/python" "${ROOT}/scripts/serve_policy_orbax_compat.py" \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.config=click_mouse \
  --policy.dir="${LOCAL_CHECKPOINT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "PI0.5 policy server exited before becoming ready" >&2
    tail -100 "${SERVER_LOG}" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "${READY}" != "1" ]]; then
  echo "PI0.5 policy server did not become ready" >&2
  tail -100 "${SERVER_LOG}" >&2
  exit 1
fi

cd "${ROOT}"
PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/packages/openpi-client/src" \
"${DEXJOCO_ENV}/bin/python" -m dexjoco_openpi_client.eval_dexjoco_openpi \
  --config="${ROOT}/configs/rand_obj/click_mouse.yaml" \
  --seed="${SEED}" \
  --port="${PORT}" \
  --host=127.0.0.1 \
  --output="${ROLLOUT_DIR}" \
  --episodes="${EPISODES}" \
  --replan-ratio=0.01 \
  --save-replay-zarr

cleanup
trap - EXIT

PYTHONPATH="${ROOT}/dexjoco" \
"${DATASET_ENV}/bin/python" scripts/build_offline_dataset.py \
  --config "${DATASET_CONFIG}" \
  --source zarr \
  --input-zarr-root "${ROLLOUT_DIR}"

echo "dataset=${ROOT}/outputs/ogpo/click_mouse_pi05_replay_100ep.pt"
echo "collection_end=$(date --iso-8601=seconds)"
