#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

conda activate "${DEXJOCO_ENV_PREFIX}"
cd "${DEXJOCO_ROOT}"

ARGS=(
  --config="${PI05_TASK_CONFIG}"
  --seed="${PI05_SEED}"
  --port="${PI05_SERVER_PORT}"
  --episodes="${PI05_EPISODES}"
)

if [[ "${PI05_CONFIG_SET}" == "rand_full" ]]; then
  ARGS+=(--rand-full)
fi

echo "[eval] config=${PI05_TASK_CONFIG}"
echo "[eval] episodes=${PI05_EPISODES}"

dexjoco-openpi-eval "${ARGS[@]}"
