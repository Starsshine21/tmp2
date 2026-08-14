#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_click_eval100
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/eval100-seed27-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/eval100-seed27-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 0-06:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
DEXJOCO_ROOT=${ROOT}/dexjoco
RECAP_ROOT=${ROOT}/recap_click_mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OUT=${ROOT}/outputs/recap-click-mouse
DEXJOCO_PY=${DEXJOCO_ROOT}/.conda/dexjoco/bin/python
OPENPI_PY=${DEXJOCO_ROOT}/.conda/openpi/bin/python
CHECKPOINT=${OUT}/actor_checkpoints/click_mouse/recap_click_mouse_pistar06_acp_2k/1999
EVAL_CONFIG=${RECAP_ROOT}/configs/click_mouse_acp_positive.yaml
EPISODES=100
SEED=27
PORT=$((21000 + SLURM_JOB_ID % 1000))
EVAL_DIR=${OUT}/eval/step1999_acp_positive_seed27_100ep_${SLURM_JOB_ID}
LOCAL_CHECKPOINT=${SLURM_TMPDIR:-/tmp}/recap-click-mouse-step1999-${SLURM_JOB_ID}
SERVER_LOG=${OUT}/logs/eval100-seed27-server-${SLURM_JOB_ID}.log

mkdir -p "${OUT}/logs" "${OUT}/eval" "${EVAL_DIR}" "${LOCAL_CHECKPOINT}"
test -x "${DEXJOCO_PY}"
test -x "${OPENPI_PY}"
test -d "${CHECKPOINT}/params"
test -f "${EVAL_CONFIG}"
rsync -a "${CHECKPOINT}/" "${LOCAL_CHECKPOINT}/"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DEXJOCO_RECAP_DATASET_ROOT=${OUT}/data
export DEXJOCO_RECAP_CKPTS_ROOT=${OUT}/actor_checkpoints
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "evaluation_start=$(date --iso-8601=seconds)"
echo "checkpoint=${CHECKPOINT}"
echo "episodes=${EPISODES} seed=${SEED} replan_ratio=0.8"
echo "prompt=Move the mouse to the purple mouse pad and click the left mouse button. | Advantage: positive"
echo "output=${EVAL_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${OPENPI_ROOT}"
CUDA_VISIBLE_DEVICES=0 \
"${OPENPI_PY}" scripts/serve_policy_orbax_compat.py \
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
for _ in $(seq 1 300); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ReCap policy server exited before becoming ready" >&2
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
  echo "ReCap policy server did not become ready" >&2
  tail -100 "${SERVER_LOG}" >&2
  exit 1
fi

cd "${DEXJOCO_ROOT}"
PYTHONPATH="${DEXJOCO_ROOT}/dexjoco:${DEXJOCO_ROOT}/openpi/packages/openpi-client/src" \
"${DEXJOCO_PY}" -m dexjoco_openpi_client.eval_dexjoco_openpi \
  --config="${EVAL_CONFIG}" \
  --seed="${SEED}" \
  --port="${PORT}" \
  --host=127.0.0.1 \
  --output="${EVAL_DIR}" \
  --episodes="${EPISODES}" \
  --replan-ratio=0.8

cleanup
trap - EXIT

SUCCESSES=$(find "${EVAL_DIR}" -maxdepth 1 -type d -name 'episode_*_success' | wc -l)
FAILURES=$(find "${EVAL_DIR}" -maxdepth 1 -type d -name 'episode_*_failure' | wc -l)
TOTAL=$((SUCCESSES + FAILURES))
if [[ "${TOTAL}" -ne "${EPISODES}" ]]; then
  echo "Expected ${EPISODES} completed episodes, got ${TOTAL}" >&2
  exit 1
fi
printf 'checkpoint\tseed\tepisodes\tsuccesses\tfailures\tsuccess_rate\tprompt_mode\n' >"${EVAL_DIR}/summary.tsv"
printf '1999\t%s\t%s\t%s\t%s\t%s\tacp_positive\n' \
  "${SEED}" "${TOTAL}" "${SUCCESSES}" "${FAILURES}" "$((100 * SUCCESSES / TOTAL))%" \
  >>"${EVAL_DIR}/summary.tsv"
ln -sfn "${EVAL_DIR}" "${OUT}/eval/latest_seed27_100ep"
touch "${OUT}/EVAL_SEED27_100EP_COMPLETE"
echo "successes=${SUCCESSES} failures=${FAILURES} total=${TOTAL}"
echo "evaluation_end=$(date --iso-8601=seconds)"
