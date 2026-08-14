#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

conda activate "${PI05_EFFECTIVE_OPENPI_ENV_PREFIX}"
cd "${DEXJOCO_ROOT}/openpi"

if [[ ! -d "${PI05_POLICY_DIR}" ]]; then
  echo "[serve] missing policy dir: ${PI05_POLICY_DIR}" >&2
  echo "[serve] if you only want to smoke-test the base environment, set PI05_POLICY_DIR to a valid checkpoint dir once download finishes." >&2
  exit 1
fi

if [[ "${DEXJOCO_STRIP_PYTHONPATH}" == "1" ]]; then
  export PYTHONPATH="${DEXJOCO_CLEAN_PYTHONPATH_DEFAULT}"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${DEXJOCO_ROOT}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

echo "[serve] env=${PI05_EFFECTIVE_OPENPI_ENV_PREFIX}"
echo "[serve] config=${PI05_POLICY_CONFIG}"
echo "[serve] checkpoint=${PI05_POLICY_DIR}"
echo "[serve] pythonpath=${PYTHONPATH:-<unset>}"
echo "[serve] port=${PI05_SERVER_PORT}"

XLA_PYTHON_CLIENT_MEM_FRACTION="${PI05_XLA_MEM_FRACTION}" \
CUDA_VISIBLE_DEVICES="${PI05_CUDA_VISIBLE_DEVICES}" \
python ./scripts/serve_policy.py \
  --port="${PI05_SERVER_PORT}" \
  policy:checkpoint \
  --policy.config="${PI05_POLICY_CONFIG}" \
  --policy.dir="${PI05_POLICY_DIR}"
