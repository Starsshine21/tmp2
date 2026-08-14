#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_base_norm
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/base-norm-ablation-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/base-norm-ablation-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40|L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 0-04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
DEXJOCO_ROOT=${ROOT}/dexjoco
RECAP_ROOT=${ROOT}/recap_click_mouse
OPENPI_ROOT=${RECAP_ROOT}/runtime/dexjoco/openpi
OUT=${ROOT}/outputs/recap-click-mouse
DEXJOCO_PY=${DEXJOCO_ROOT}/.conda/dexjoco/bin/python
OPENPI_PY=${DEXJOCO_ROOT}/.conda/openpi/bin/python
MODEL=${DEXJOCO_ROOT}/checkpoints/pi05_pytorch/click_mouse/model.safetensors
NORM_MODE=${NORM_MODE:?NORM_MODE must be base or recap}
EPISODES=${EPISODES:-50}
SEED=${SEED:-27}
PORT=$((25000 + SLURM_JOB_ID % 10000))

case "${NORM_MODE}" in
  base)
    NORM=${DEXJOCO_ROOT}/checkpoints/pi05_pytorch/click_mouse/assets/local_repo/norm_stats.json
    ;;
  recap)
    NORM=${OPENPI_ROOT}/assets/click_mouse_recap_full/local_repo/norm_stats.json
    ;;
  *)
    echo "Unsupported NORM_MODE=${NORM_MODE}" >&2
    exit 1
    ;;
esac

POLICY_DIR=${SLURM_TMPDIR:-/tmp}/recap-base-norm-${NORM_MODE}-${SLURM_JOB_ID}
EVAL_DIR=${OUT}/eval/pytorch_base_${NORM_MODE}_norm_seed${SEED}_${EPISODES}ep_${SLURM_JOB_ID}
SERVER_LOG=${OUT}/logs/base-${NORM_MODE}-norm-server-${SLURM_JOB_ID}.log

mkdir -p "${POLICY_DIR}/assets/local_repo" "${EVAL_DIR}" "${OUT}/logs" "${OUT}/cache/openpi"
test -s "${MODEL}"
test -s "${NORM}"
ln -s "${MODEL}" "${POLICY_DIR}/model.safetensors"
ln -s "${NORM}" "${POLICY_DIR}/assets/local_repo/norm_stats.json"

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
export DEXJOCO_RECAP_FULL_MAX_TOKEN_LEN=128
export DEXJOCO_RECAP_PYTORCH_COMPILE_MODE=none
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "evaluation_start=$(date --iso-8601=seconds)"
echo "model=${MODEL} norm_mode=${NORM_MODE} norm=${NORM} episodes=${EPISODES} seed=${SEED}"

cd "${OPENPI_ROOT}"
CUDA_VISIBLE_DEVICES=0 \
"${OPENPI_PY}" scripts/serve_policy_orbax_compat.py \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.config=click_mouse_recap_full \
  --policy.dir="${POLICY_DIR}" >"${SERVER_LOG}" 2>&1 &
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
printf 'norm_mode\tseed\tepisodes\tsuccesses\tfailures\n%s\t%s\t%s\t%s\t%s\n' \
  "${NORM_MODE}" "${SEED}" "${EPISODES}" "${SUCCESSES}" "${FAILURES}" >"${EVAL_DIR}/summary.tsv"
echo "norm_mode=${NORM_MODE} successes=${SUCCESSES} failures=${FAILURES}"
