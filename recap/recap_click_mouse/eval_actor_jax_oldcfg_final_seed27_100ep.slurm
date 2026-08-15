#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_eval_g3s2
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/eval-jax-oldcfg-final-seed27-100ep-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/eval-jax-oldcfg-final-seed27-100ep-%j.err
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
REFERENCE_ROOT=${RECAP_ROOT}/reference_evorl_95f8b6c
OPENPI_ROOT=${REFERENCE_ROOT}/dexjoco/openpi
OUT=${ROOT}/outputs/recap-click-mouse
DEXJOCO_PY=${DEXJOCO_ROOT}/.conda/dexjoco/bin/python
OPENPI_PY=${DEXJOCO_ROOT}/.conda/openpi/bin/python
EVAL_STEP=${RECAP_EVAL_STEP:-29999}
CHECKPOINT=${OUT}/actor_jax_oldcfg_checkpoints/click_mouse/recap_click_mouse_gemma3value10k_100ep_oldcfg/${EVAL_STEP}
EVAL_CONFIG=${RECAP_ROOT}/configs/click_mouse_acp_positive.yaml
EPISODES=100
SEED=27
PORT=$((22000 + SLURM_JOB_ID % 1000))
EVAL_DIR=${OUT}/eval/jax_oldcfg_gemma3value10k_step${EVAL_STEP}_acp_positive_seed27_100ep_${SLURM_JOB_ID}
LOCAL_CHECKPOINT=${SLURM_TMPDIR:-/tmp}/recap-jax-oldcfg-step${EVAL_STEP}-${SLURM_JOB_ID}
SERVER_LOG=${OUT}/logs/eval-jax-oldcfg-step${EVAL_STEP}-server-${SLURM_JOB_ID}.log

mkdir -p "${OUT}/logs" "${OUT}/eval" "${EVAL_DIR}" "${LOCAL_CHECKPOINT}"
test -x "${DEXJOCO_PY}"
test -x "${OPENPI_PY}"
test -d "${CHECKPOINT}/params"
test -f "${EVAL_CONFIG}"
grep -Fxq '  Advantage: positive' "${EVAL_CONFIG}"
rsync -a "${CHECKPOINT}/" "${LOCAL_CHECKPOINT}/"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DEFAULT_PROMPT=$'Move the mouse to the purple mouse pad and click the left mouse button.\nAdvantage: positive'
export OPENPI_ACTION_CHUNK_TIMESTAMP_MODE=obs_plus_1
export OPENPI_DATASET_ROOT=${OUT}/data
export OPENPI_CKPTS_ROOT=${OUT}/actor_jax_oldcfg_checkpoints
export PYTHONPATH="${REFERENCE_ROOT}/src:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "evaluation_start=$(date --iso-8601=seconds)"
echo "checkpoint=${CHECKPOINT}"
echo "episodes=${EPISODES} seed=${SEED} replan_ratio=0.8"
echo "prompt=Move the mouse to the purple mouse pad and click the left mouse button. | Advantage: positive"
echo "output=${EVAL_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${OPENPI_ROOT}"
CUDA_VISIBLE_DEVICES=0 \
"${OPENPI_PY}" scripts/serve_policy.py \
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
printf '%s\t%s\t%s\t%s\t%s\t%s\tacp_positive\n' \
  "${EVAL_STEP}" \
  "${SEED}" "${TOTAL}" "${SUCCESSES}" "${FAILURES}" "$((100 * SUCCESSES / TOTAL))%" \
  >>"${EVAL_DIR}/summary.tsv"
ln -sfn "${EVAL_DIR}" "${OUT}/eval/latest_jax_oldcfg_step${EVAL_STEP}_seed27_100ep"
touch "${OUT}/EVAL_JAX_OLDCFG_GEMMA3VALUE10K_STEP${EVAL_STEP}_SEED27_100EP_COMPLETE"
echo "successes=${SUCCESSES} failures=${FAILURES} total=${TOTAL}"
echo "evaluation_end=$(date --iso-8601=seconds)"
