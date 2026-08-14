#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

conda activate "${DEXJOCO_ENV_PREFIX}"
cd "${DEXJOCO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${DEXJOCO_ROOT}/outputs/recordings/${PI05_TASK}_${PI05_CONFIG_SET}_seed${PI05_SEED}}"
SUCCESS_NEEDED="${SUCCESS_NEEDED:-20}"
RENDER_MODE="${RENDER_MODE:-human}"
RANDOMIZE="${RANDOMIZE:-0}"

ARGS=(
  --exp_name="${PI05_TASK}"
  --out_dir="${OUTPUT_DIR}"
  --successes_needed="${SUCCESS_NEEDED}"
  --render_mode="${RENDER_MODE}"
)

if [[ "${PI05_CONFIG_SET}" == "rand_full" || "${RANDOMIZE}" == "1" ]]; then
  ARGS+=(--randomize)
fi

echo "[collect] config=${PI05_TASK_CONFIG}"
echo "[collect] output=${OUTPUT_DIR}"
echo "[collect] successes=${SUCCESS_NEEDED}"

python ./scripts/record_demos_zarr.py "${ARGS[@]}"
