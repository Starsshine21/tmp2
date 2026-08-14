#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_click_collect
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/collect-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/collect-%j.err
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

ROOT=/nfs_global/S/yangrongzheng/evo-RL
DEXJOCO_ROOT=${ROOT}/dexjoco
RECAP_ROOT=${ROOT}/recap_click_mouse
OUT=${ROOT}/outputs/recap-click-mouse
DEXJOCO_PY=${DEXJOCO_ROOT}/.conda/dexjoco/bin/python
OPENPI_PY=${DEXJOCO_ROOT}/.conda/openpi/bin/python
VALUE_PY=/nfs_global/S/yangrongzheng/pi05/recap_workspace/.venv_recap311/bin/python
SOURCE_CHECKPOINT=${ROOT}/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse

MODE=${RECAP_COLLECTION_MODE:-formal}
EPISODES=${RECAP_COLLECTION_EPISODES:-100}
SEED=${RECAP_COLLECTION_SEED:-27}
if [[ "${MODE}" != "formal" && "${MODE}" != "smoke" ]]; then
  echo "RECAP_COLLECTION_MODE must be formal or smoke, got ${MODE}" >&2
  exit 2
fi
if [[ "${MODE}" == "formal" ]]; then
  DATA_ROOT=${OUT}/data/click_mouse
  SUMMARY=${OUT}/data/click_mouse_dataset_summary.json
  COMPLETE=${OUT}/DATA_COMPLETE
else
  DATA_ROOT=${OUT}/data_smoke/click_mouse
  SUMMARY=${OUT}/data_smoke/click_mouse_dataset_summary.json
  COMPLETE=${OUT}/DATA_SMOKE_COMPLETE
fi

PORT=$((19000 + SLURM_JOB_ID % 1000))
ROLLOUT_DIR=${OUT}/raw_rollouts/${MODE}_seed${SEED}_${EPISODES}ep_${SLURM_JOB_ID}
LOCAL_CHECKPOINT=${SLURM_TMPDIR:-/tmp}/recap-click-mouse-${MODE}-${SLURM_JOB_ID}
SERVER_LOG=${OUT}/logs/${MODE}-base-server-${SLURM_JOB_ID}.log

mkdir -p "${OUT}/logs" "${OUT}/raw_rollouts" "${OUT}/cache" "${OUT}/tmp" "${LOCAL_CHECKPOINT}"
test -x "${DEXJOCO_PY}"
test -x "${OPENPI_PY}"
test -x "${VALUE_PY}"
test -d "${SOURCE_CHECKPOINT}/params"
rsync -a "${SOURCE_CHECKPOINT}/" "${LOCAL_CHECKPOINT}/"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export MPLCONFIGDIR=${OUT}/cache/matplotlib
export TMPDIR=${OUT}/tmp
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "collection_start=$(date --iso-8601=seconds)"
echo "mode=${MODE} episodes=${EPISODES} seed=${SEED} rollout_dir=${ROLLOUT_DIR}"
echo "base_checkpoint=${SOURCE_CHECKPOINT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${DEXJOCO_ROOT}/openpi"
PYTHONPATH="${DEXJOCO_ROOT}/openpi/src:${DEXJOCO_ROOT}/openpi/packages/openpi-client/src" \
CUDA_VISIBLE_DEVICES=0 \
"${OPENPI_PY}" "${DEXJOCO_ROOT}/scripts/serve_policy_orbax_compat.py" \
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
if [[ "${READY}" != 1 ]]; then
  echo "PI0.5 policy server did not become ready" >&2
  tail -100 "${SERVER_LOG}" >&2
  exit 1
fi

cd "${DEXJOCO_ROOT}"
PYTHONPATH="${DEXJOCO_ROOT}/dexjoco:${DEXJOCO_ROOT}/openpi/packages/openpi-client/src" \
"${DEXJOCO_PY}" -m dexjoco_openpi_client.eval_dexjoco_openpi \
  --config="${DEXJOCO_ROOT}/configs/rand_obj/click_mouse.yaml" \
  --seed="${SEED}" \
  --port="${PORT}" \
  --host=127.0.0.1 \
  --output="${ROLLOUT_DIR}" \
  --episodes="${EPISODES}" \
  --replan-ratio=0.01 \
  --save-replay-zarr

cleanup
trap - EXIT

export PYTHONPATH="${RECAP_ROOT}/pydeps:${RECAP_ROOT}/lerobot_src"
"${VALUE_PY}" "${RECAP_ROOT}/scripts/dexjoco_zarr_to_lerobot.py" \
  --input-root "${ROLLOUT_DIR}" \
  --output-root "${DATA_ROOT}" \
  --summary-output "${SUMMARY}" \
  --repo-id local_repo \
  --expected-episodes "${EPISODES}" \
  --overwrite

printf '%s\n' "${ROLLOUT_DIR}" >"${OUT}/${MODE^^}_RAW_ROLLOUT_DIR"
touch "${COMPLETE}"
echo "dataset=${DATA_ROOT}"
echo "collection_end=$(date --iso-8601=seconds)"
