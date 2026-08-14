#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_full5k_eval100
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/full5k-eval100-seed27-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/full5k-eval100-seed27-%j.err
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
CHECKPOINT=${OUT}/actor_full_pytorch_checkpoints/click_mouse_recap_full/recap_click_mouse_full_pytorch_b32_30000s_seed27/5000
EVAL_CONFIG=${RECAP_ROOT}/configs/click_mouse_acp_positive.yaml
EPISODES=100
SEED=27
PORT=$((22000 + SLURM_JOB_ID % 10000))
EVAL_DIR=${OUT}/eval/full_pytorch_step5000_acp_positive_seed27_100ep_${SLURM_JOB_ID}
SERVER_LOG=${OUT}/logs/full5k-eval100-seed27-server-${SLURM_JOB_ID}.log

mkdir -p "${OUT}/logs" "${OUT}/eval" "${EVAL_DIR}" "${OUT}/cache/openpi"
test -x "${DEXJOCO_PY}"
test -x "${OPENPI_PY}"
test -s "${CHECKPOINT}/model.safetensors"
test -s "${CHECKPOINT}/assets/local_repo/norm_stats.json"
test -f "${EVAL_CONFIG}"

"${DEXJOCO_PY}" - "${EVAL_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    prompt = yaml.safe_load(stream)["prompt"]
expected = "Move the mouse to the purple mouse pad and click the left mouse button.\nAdvantage: positive"
if prompt != expected:
    raise RuntimeError(f"ACP evaluation prompt mismatch: {prompt!r}")
print(f"verified_acp_prompt={prompt!r}")
PY

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export OPENPI_DATA_HOME=${OUT}/cache/openpi
export DEXJOCO_RECAP_DATASET_ROOT=${OUT}/data
export DEXJOCO_RECAP_FULL_CKPTS_ROOT=${OUT}/actor_full_pytorch_checkpoints
export DEXJOCO_RECAP_FULL_MAX_TOKEN_LEN=128
export DEXJOCO_RECAP_PYTORCH_COMPILE_MODE=none
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
  --policy.config=click_mouse_recap_full \
  --policy.dir="${CHECKPOINT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 300); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ReCap full PyTorch policy server exited before becoming ready" >&2
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
  echo "ReCap full PyTorch policy server did not become ready" >&2
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
printf '5000\t%s\t%s\t%s\t%s\t%s\tacp_positive\n' \
  "${SEED}" "${TOTAL}" "${SUCCESSES}" "${FAILURES}" "$((100 * SUCCESSES / TOTAL))%" \
  >>"${EVAL_DIR}/summary.tsv"
ln -sfn "${EVAL_DIR}" "${OUT}/eval/latest_full_step5000_seed27_100ep"
touch "${OUT}/EVAL_FULL_STEP5000_SEED27_100EP_COMPLETE"
echo "successes=${SUCCESSES} failures=${FAILURES} total=${TOTAL}"
echo "evaluation_end=$(date --iso-8601=seconds)"
