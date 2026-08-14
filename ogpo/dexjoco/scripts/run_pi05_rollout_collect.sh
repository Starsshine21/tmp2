#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

conda activate "${DEXJOCO_ENV_PREFIX}"
cd "${DEXJOCO_ROOT}"

ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-${DEXJOCO_ROOT}/outputs/pi05_rollouts/${PI05_TASK}_${PI05_CONFIG_SET}_seed${PI05_SEED}}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-20}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_RECORD_PASSWORD="${ROLLOUT_RECORD_PASSWORD:-0}"
ROLLOUT_RANDOMIZE_DYNAMICS="${ROLLOUT_RANDOMIZE_DYNAMICS:-0}"
ROLLOUT_SAVE_REPLAY_ZARR="${ROLLOUT_SAVE_REPLAY_ZARR:-1}"
ROLLOUT_REPLAN_RATIO="${ROLLOUT_REPLAN_RATIO:-0.8}"

if [[ "${ROLLOUT_HOST}" == "127.0.0.1" || "${ROLLOUT_HOST}" == "localhost" ]]; then
  unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY ws_proxy WS_PROXY wss_proxy WSS_PROXY
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
  export no_proxy="${no_proxy:-127.0.0.1,localhost}"
fi

ARGS=(
  --config="${PI05_TASK_CONFIG}"
  --seed="${PI05_SEED}"
  --port="${PI05_SERVER_PORT}"
  --episodes="${ROLLOUT_EPISODES}"
  --output="${ROLLOUT_OUTPUT_DIR}"
  --host="${ROLLOUT_HOST}"
  --replan-ratio="${ROLLOUT_REPLAN_RATIO}"
)

if [[ "${PI05_CONFIG_SET}" == "rand_full" ]]; then
  ARGS+=(--rand-full)
fi

if [[ "${ROLLOUT_RECORD_PASSWORD}" == "1" ]]; then
  ARGS+=(--record-pressed-digits)
fi

if [[ "${ROLLOUT_RANDOMIZE_DYNAMICS}" == "1" ]]; then
  ARGS+=(--randomize-dynamics)
fi

if [[ "${ROLLOUT_SAVE_REPLAY_ZARR}" == "1" ]]; then
  ARGS+=(--save-replay-zarr)
fi

echo "[rollout] config=${PI05_TASK_CONFIG}"
echo "[rollout] output=${ROLLOUT_OUTPUT_DIR}"
echo "[rollout] episodes=${ROLLOUT_EPISODES}"
echo "[rollout] host=${ROLLOUT_HOST} port=${PI05_SERVER_PORT}"
echo "[rollout] replan_ratio=${ROLLOUT_REPLAN_RATIO}"
echo "[rollout] save_replay_zarr=${ROLLOUT_SAVE_REPLAY_ZARR}"

python -m dexjoco_openpi_client.eval_dexjoco_openpi "${ARGS[@]}"
