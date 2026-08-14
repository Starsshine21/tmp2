#!/bin/bash
#SBATCH -J pi05_dex_eval
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-dex-eval-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-dex-eval-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH -t 0-01:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${ROOT}/checkpoints/pi05_pytorch/click_mouse}
LOCAL_CHECKPOINT=${SLURM_TMPDIR:-/tmp}/pi05-pytorch-click-mouse-${SLURM_JOB_ID}
SERVER_LOG=${ROOT}/outputs/ogpo/logs/pi05-dex-server-${SLURM_JOB_ID}.log

mkdir -p "${ROOT}/outputs/ogpo/logs" "${LOCAL_CHECKPOINT}"
rsync -a "${SOURCE_CHECKPOINT}/" "${LOCAL_CHECKPOINT}/"

export PI05_USE_CLEAN_OPENPI_ENV=1
export PI05_TASK=click_mouse
export PI05_CONFIG_SET=rand_obj
export PI05_SEED=0
export PI05_POLICY_DIR="${LOCAL_CHECKPOINT}"
export PI05_PYTORCH_POLICY_DIR="${LOCAL_CHECKPOINT}"
export PI05_SERVER_PORT="${PI05_SERVER_PORT:-$((18000 + SLURM_JOB_ID % 1000))}"
export PI05_CUDA_VISIBLE_DEVICES=0
export ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-1}"
export ROLLOUT_HOST=127.0.0.1
export ROLLOUT_REPLAN_RATIO="${ROLLOUT_REPLAN_RATIO:-0.01}"
export ROLLOUT_SAVE_REPLAY_ZARR=1
export ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-${ROOT}/outputs/pi05_rollouts/click_mouse_pytorch_${ROLLOUT_EPISODES}ep_${SLURM_JOB_ID}}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "evaluation_start=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ -n "${OGPO_CHECKPOINT:-}" ]]; then
  bash "${ROOT}/scripts/run_pi05_ogpo_server.sh" >"${SERVER_LOG}" 2>&1 &
else
  bash "${ROOT}/scripts/run_pi05_server.sh" >"${SERVER_LOG}" 2>&1 &
fi
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "policy server exited before becoming ready" >&2
    tail -100 "${SERVER_LOG}" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PI05_SERVER_PORT}/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "${READY}" != "1" ]]; then
  echo "policy server did not become ready" >&2
  tail -100 "${SERVER_LOG}" >&2
  exit 1
fi

bash "${ROOT}/scripts/run_pi05_rollout_collect.sh"

echo "server_log=${SERVER_LOG}"
echo "rollout_output=${ROLLOUT_OUTPUT_DIR}"
echo "evaluation_end=$(date --iso-8601=seconds)"
