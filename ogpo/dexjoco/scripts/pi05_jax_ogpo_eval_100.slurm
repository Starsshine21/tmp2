#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_ogpo_eval100
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-ogpo-eval100-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-ogpo-eval100-%j.err
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

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
DEXJOCO_ENV=${DEXJOCO_ENV:-${ROOT}/.conda/dexjoco}
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}
PI05_CHECKPOINT=${PI05_CHECKPOINT:-/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse}
OGPO_CHECKPOINT=${OGPO_CHECKPOINT:-${ROOT}/outputs/ogpo/checkpoints/click_mouse_pi05_jax_flash_fast_step_0025/click_mouse_pi05_jax_flash_100ep_fast.pt}
EPISODES=${PI05_EPISODES:-100}
SEED=${PI05_SEED:-27}
EVAL_PROTOCOL=${PI05_EVAL_PROTOCOL:-strict}
STRICT_REPRODUCIBILITY=${PI05_STRICT_REPRODUCIBILITY:-true}
DETERMINISTIC_POLICY_NOISE=${PI05_DETERMINISTIC_POLICY_NOISE:-${STRICT_REPRODUCIBILITY}}
BLOCK_WHEN_EMPTY=${PI05_BLOCK_ON_INFERENCE_WHEN_EMPTY:-${STRICT_REPRODUCIBILITY}}
BLOCK_AFTER_REPLAN=${PI05_BLOCK_AFTER_REPLAN:-${STRICT_REPRODUCIBILITY}}
VERIFY_POLICY_REPEATABILITY=${PI05_VERIFY_POLICY_REPEATABILITY:-false}
DETERMINISTIC_XLA=${PI05_DETERMINISTIC_XLA:-${STRICT_REPRODUCIBILITY}}
INFERENCE_DYNAMICS=${PI05_EVAL_INFERENCE_DYNAMICS:-native_ode}
INFERENCE_NUM_STEPS=${PI05_EVAL_NUM_STEPS:-10}
REQUIRE_MODEL_FLOW_DIM=${PI05_EVAL_REQUIRE_MODEL_FLOW_DIM:-true}
LEGACY_EVALUATION=false
FAST_POLICY_SERVER=${PI05_FAST_POLICY_SERVER:-true}
if [[ "${EVAL_PROTOCOL}" == "legacy" ]]; then
  STRICT_REPRODUCIBILITY=false
  DETERMINISTIC_POLICY_NOISE=false
  BLOCK_WHEN_EMPTY=false
  BLOCK_AFTER_REPLAN=false
  DETERMINISTIC_XLA=false
  LEGACY_EVALUATION=true
  FAST_POLICY_SERVER=false
elif [[ "${EVAL_PROTOCOL}" != "strict" && "${EVAL_PROTOCOL}" != "custom" ]]; then
  echo "Unsupported PI05_EVAL_PROTOCOL=${EVAL_PROTOCOL}; expected strict, legacy, or custom" >&2
  exit 2
fi
PORT=${PI05_SERVER_PORT:-$((19000 + SLURM_JOB_ID % 1000))}
OUTPUT_DIR=${PI05_EVAL_OUTPUT_DIR:-${ROOT}/outputs/pi05_rollouts/click_mouse_jax_ogpo_step0025_${EPISODES}ep_${SLURM_JOB_ID}}
SERVER_LOG=${ROOT}/outputs/ogpo/logs/pi05-ogpo-eval100-server-${SLURM_JOB_ID}.log

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_file "${DEXJOCO_ENV}/bin/python"
require_file "${OPENPI_ENV}/bin/python"
require_file "${OGPO_CHECKPOINT}"
require_dir "${OGPO_CHECKPOINT}.jax"
require_dir "${PI05_CHECKPOINT}/params"

mkdir -p "${ROOT}/outputs/ogpo/logs" "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  FIRST_SLURM_GPU=${SLURM_JOB_GPUS%%,*}
  FIRST_CUDA_GPU=${CUDA_VISIBLE_DEVICES%%,*}
  if [[ "${FIRST_SLURM_GPU}" =~ ^[0-9]+$ ]]; then
    export MUJOCO_EGL_DEVICE_ID=${FIRST_SLURM_GPU}
  elif [[ "${FIRST_CUDA_GPU}" =~ ^[0-9]+$ ]]; then
    export MUJOCO_EGL_DEVICE_ID=${FIRST_CUDA_GPU}
  else
    export MUJOCO_EGL_DEVICE_ID=0
  fi
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.65}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ "${STRICT_REPRODUCIBILITY}" == "true" ]]; then
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
fi
if [[ "${DETERMINISTIC_XLA}" == "true" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0 --xla_gpu_enable_latency_hiding_scheduler=false"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "evaluation_start=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID} host=$(hostname)"
echo "checkpoint=${OGPO_CHECKPOINT}"
echo "episodes=${EPISODES} seed=${SEED} output=${OUTPUT_DIR}"
echo "eval_protocol=${EVAL_PROTOCOL} strict_reproducibility=${STRICT_REPRODUCIBILITY} deterministic_policy_noise=${DETERMINISTIC_POLICY_NOISE} block_when_empty=${BLOCK_WHEN_EMPTY} block_after_replan=${BLOCK_AFTER_REPLAN} fast_policy_server=${FAST_POLICY_SERVER}"
echo "inference_dynamics=${INFERENCE_DYNAMICS} inference_num_steps=${INFERENCE_NUM_STEPS} require_model_flow_dim=${REQUIRE_MODEL_FLOW_DIM}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} slurm_job_gpus=${SLURM_JOB_GPUS:-unset} mujoco_egl_device_id=${MUJOCO_EGL_DEVICE_ID}"
echo "prompt=Move the mouse to the purple mouse pad and click the left mouse button."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

SERVER_BOOL_FLAGS=()
if [[ "${FAST_POLICY_SERVER}" == "true" ]]; then
  SERVER_BOOL_FLAGS+=(--skip-reference-policy --skip-critic)
fi
if [[ "${REQUIRE_MODEL_FLOW_DIM}" == "true" ]]; then
  SERVER_BOOL_FLAGS+=(--require-model-flow-dim)
fi

cd "${ROOT}"
PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src" \
"${OPENPI_ENV}/bin/python" scripts/serve_ogpo_policy.py \
  --port="${PORT}" \
  --pi05-checkpoint="${PI05_CHECKPOINT}" \
  --train-config=click_mouse \
  --ogpo-checkpoint="${OGPO_CHECKPOINT}" \
  --device=cuda \
  --inference-dynamics="${INFERENCE_DYNAMICS}" \
  --inference-num-steps="${INFERENCE_NUM_STEPS}" \
  "${SERVER_BOOL_FLAGS[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 300); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "OGPO policy server exited before becoming ready" >&2
    tail -100 "${SERVER_LOG}" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "${READY}" != "1" ]]; then
  echo "OGPO policy server did not become ready" >&2
  tail -100 "${SERVER_LOG}" >&2
  exit 1
fi

EVAL_BOOL_FLAGS=()
if [[ "${BLOCK_WHEN_EMPTY}" == "true" ]]; then
  EVAL_BOOL_FLAGS+=(--block-on-inference-when-empty)
fi
if [[ "${DETERMINISTIC_POLICY_NOISE}" == "true" ]]; then
  EVAL_BOOL_FLAGS+=(--deterministic-policy-noise)
fi
if [[ "${BLOCK_AFTER_REPLAN}" == "true" ]]; then
  EVAL_BOOL_FLAGS+=(--block-on-inference-after-replan)
fi
if [[ "${VERIFY_POLICY_REPEATABILITY}" == "true" ]]; then
  EVAL_BOOL_FLAGS+=(--verify-policy-repeatability)
fi
if [[ "${STRICT_REPRODUCIBILITY}" == "true" ]]; then
  EVAL_BOOL_FLAGS+=(--strict-reproducibility)
fi
if [[ "${LEGACY_EVALUATION}" == "true" ]]; then
  EVAL_BOOL_FLAGS+=(--legacy-evaluation)
fi

PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/packages/openpi-client/src" \
"${DEXJOCO_ENV}/bin/python" -m dexjoco_openpi_client.eval_dexjoco_openpi \
  --config="${ROOT}/configs/rand_obj/click_mouse.yaml" \
  --seed="${SEED}" \
  --port="${PORT}" \
  --host=127.0.0.1 \
  --output="${OUTPUT_DIR}" \
  --episodes="${EPISODES}" \
  --replan-ratio=0.8 \
  --policy-noise-dim=32 \
  "${EVAL_BOOL_FLAGS[@]}"

cleanup
trap - EXIT

echo "metrics=${OUTPUT_DIR}/evaluation_metrics.jsonl"
echo "evaluation_end=$(date --iso-8601=seconds)"
