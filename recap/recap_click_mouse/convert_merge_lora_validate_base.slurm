#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_merge_base
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/merge-base-validate-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/merge-base-validate-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40|L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 0-04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
DEXJOCO_ROOT=${ROOT}/dexjoco
RECAP_ROOT=${ROOT}/recap_click_mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OUT=${ROOT}/outputs/recap-click-mouse
OPENPI_PY=${DEXJOCO_ROOT}/.conda/openpi/bin/python
DEXJOCO_PY=${DEXJOCO_ROOT}/.conda/dexjoco/bin/python
JAX_CKPT=${ROOT}/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse
MERGED=${DEXJOCO_ROOT}/checkpoints/pi05_pytorch/click_mouse_lora_merged
EPISODES=10
SEED=27
PORT=$((27000 + SLURM_JOB_ID % 10000))
EVAL_DIR=${OUT}/eval/merged_base_seed${SEED}_${EPISODES}ep_${SLURM_JOB_ID}
SERVER_LOG=${OUT}/logs/merged-base-server-${SLURM_JOB_ID}.log

mkdir -p "${OUT}/logs" "${OUT}/cache/openpi" "${EVAL_DIR}"
rm -rf "${MERGED}.tmp"
mkdir -p "${MERGED}.tmp"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${OUT}/cache/huggingface
export HF_DATASETS_CACHE=${OUT}/cache/datasets
export OPENPI_DATA_HOME=${OUT}/cache/openpi
export DEXJOCO_RECAP_DATASET_ROOT=${OUT}/data
export DEXJOCO_RECAP_FULL_CKPTS_ROOT=${OUT}/actor_full_pytorch_checkpoints
export DEXJOCO_RECAP_PYTORCH_COMPILE_MODE=none
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${OPENPI_ROOT}"
"${OPENPI_PY}" examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir="${JAX_CKPT}" \
  --config-name=click_mouse \
  --output-path="${MERGED}.tmp" \
  --precision=bfloat16
rm -rf "${MERGED}"
mv "${MERGED}.tmp" "${MERGED}"

CUDA_VISIBLE_DEVICES=0 "${OPENPI_PY}" scripts/serve_policy_orbax_compat.py \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.config=click_mouse_recap_full \
  --policy.dir="${MERGED}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 300); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -100 "${SERVER_LOG}" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
test "${READY}" = 1

cd "${DEXJOCO_ROOT}"
PYTHONPATH="${DEXJOCO_ROOT}/dexjoco:${DEXJOCO_ROOT}/openpi/packages/openpi-client/src" \
"${DEXJOCO_PY}" -m dexjoco_openpi_client.eval_dexjoco_openpi \
  --config="${DEXJOCO_ROOT}/configs/rand_obj/click_mouse.yaml" \
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
echo "merged_base successes=${SUCCESSES} failures=${FAILURES}"
if (( SUCCESSES < 3 )); then
  echo "Merged base validation failed: expected at least 3/10 successes" >&2
  exit 1
fi
touch "${OUT}/MERGED_BASE_VALIDATED"

