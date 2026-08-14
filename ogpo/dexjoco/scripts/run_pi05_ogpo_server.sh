#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

conda activate "${PI05_EFFECTIVE_OPENPI_ENV_PREFIX}"
cd "${DEXJOCO_ROOT}"

: "${PI05_POLICY_DIR:=${PI05_PYTORCH_POLICY_DIR:-}}"
: "${PI05_POLICY_DIR:?set PI05_POLICY_DIR to the native JAX or converted PyTorch checkpoint}"
: "${OGPO_CHECKPOINT:?set OGPO_CHECKPOINT to the trained residual checkpoint}"

export PYTHONPATH="${DEXJOCO_ROOT}/dexjoco:${DEXJOCO_OPENPI_SRC}:${DEXJOCO_OPENPI_CLIENT_SRC}"

echo "[serve-ogpo] env=${PI05_EFFECTIVE_OPENPI_ENV_PREFIX}"
echo "[serve-ogpo] config=${PI05_POLICY_CONFIG}"
echo "[serve-ogpo] pi05=${PI05_POLICY_DIR}"
echo "[serve-ogpo] ogpo=${OGPO_CHECKPOINT}"
echo "[serve-ogpo] port=${PI05_SERVER_PORT}"

CUDA_VISIBLE_DEVICES="${PI05_CUDA_VISIBLE_DEVICES}" \
python scripts/serve_ogpo_policy.py \
  --port="${PI05_SERVER_PORT}" \
  --pi05-checkpoint="${PI05_POLICY_DIR}" \
  --train-config="${PI05_POLICY_CONFIG}" \
  --ogpo-checkpoint="${OGPO_CHECKPOINT}" \
  --device="cuda"
