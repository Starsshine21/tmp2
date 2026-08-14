#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="robosuite"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found: ${CONDA_SH}" >&2
  return 1 2>/dev/null || exit 1
fi

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "Activated ${ENV_NAME} for ${REPO_ROOT}"
echo "NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT}"
