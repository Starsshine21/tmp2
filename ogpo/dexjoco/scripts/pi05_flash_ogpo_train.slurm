#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_flash_ogpo
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-flash-train-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-flash-train-%j.err
#SBATCH -p r8nv-gpu-hw-80g
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 1-06:00:00
#SBATCH --exclude=r8a100-d05

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
OPENPI_ENV=${OPENPI_ENV:-${ROOT}/.conda/openpi}
CONFIG=${OGPO_FLASH_CONFIG:-configs/ogpo/pi05_flash_ogpo.yaml}
TRAIN_REPLAY=${OGPO_TRAIN_REPLAY:-${ROOT}/outputs/ogpo/click_mouse_pi05_replay_train.pt}
VALIDATION_REPLAY=${OGPO_VALIDATION_REPLAY:-${ROOT}/outputs/ogpo/click_mouse_pi05_replay_validation.pt}
PI05_CHECKPOINT=${OGPO_PI05_CHECKPOINT:-${ROOT}/checkpoints/pi05_pytorch/click_mouse/model.safetensors}
GEMMA_DIR=${OGPO_GEMMA_DIR:-/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/gemma-3-270m}
SIGLIP_DIR=${OGPO_SIGLIP_DIR:-/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/siglip2-so400m-patch14-224-fixed}
CALIBRATED_CHECKPOINT=${OGPO_CALIBRATED_CHECKPOINT:-${ROOT}/outputs/ogpo/click_mouse_gemma_udivl_calibrated.pt}
CONFIG_PATH=${CONFIG}
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT}/${CONFIG_PATH}"
fi

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

require_checkpoint() {
  # Accept either a converted PyTorch checkpoint (model.safetensors file) or a
  # native JAX Orbax checkpoint (a directory containing params/).
  if [[ -d "$1" && -d "$1/params" ]]; then
    return 0
  fi
  if [[ -s "$1" ]]; then
    return 0
  fi
  echo "Missing required checkpoint (JAX dir with params/ or a safetensors file): $1" >&2
  exit 1
}

require_file "${OPENPI_ENV}/bin/python"
require_file "${CONFIG_PATH}"
require_file "${TRAIN_REPLAY}"
require_file "${VALIDATION_REPLAY}"
require_checkpoint "${PI05_CHECKPOINT}"
require_file "${CALIBRATED_CHECKPOINT}"
require_dir "${GEMMA_DIR}"
require_dir "${SIGLIP_DIR}"

mkdir -p "${ROOT}/outputs/ogpo/logs" "${ROOT}/outputs/ogpo/tensorboard"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
# expandable_segments reports a driver OOM on the A100-80GB PCIe nodes even
# with more than 78GB free. Use the portable caching allocator there.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
# On A100-80G, reserve roughly 28GB for the full Gemma+SigLIP critic and
# PyTorch CUDA runtime. JAX still receives about 52GB, more than the
# previously validated 90% allocation on a 48GB L40S.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.65}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# Compute nodes allow unlimited virtual address space, but the default login
# cgroup caps it near 192 GiB. The platform allocator's per-buffer cudaMalloc
# reservations grow with the replica count and can exceed that cap, raising
# RESOURCE_EXHAUSTED mid-run. Lift it so multi-GPU layouts keep running.
ulimit -v unlimited 2>/dev/null || true

echo "train_start=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID:-manual} host=$(hostname) config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${ROOT}"
CMD=("${OPENPI_ENV}/bin/python" scripts/train_flash_ogpo.py --config "${CONFIG_PATH}")
if [[ -n "${OGPO_FLASH_RESUME:-}" ]]; then
  require_file "${OGPO_FLASH_RESUME}"
  CMD+=(--resume "${OGPO_FLASH_RESUME}")
fi
if [[ -n "${OGPO_CONFIG_OVERLAY:-}" ]]; then
  require_file "${OGPO_CONFIG_OVERLAY}"
  CMD+=(--overlay "${OGPO_CONFIG_OVERLAY}")
fi
printf 'train_command:'
printf ' %q' "${CMD[@]}"
printf '\n'
ACTOR_GPUS_PER_TASK=${OGPO_ACTOR_GPUS_PER_TASK:-1}
if (( ACTOR_GPUS_PER_TASK > 1 )); then
  GPU_BIND_ARGS=()
else
  GPU_BIND_ARGS=(--gpu-bind=closest)
fi
# The 80GB hardware partition allocates GPUs in blocks of four, while this
# mixed JAX/PyTorch trainer defaults to one GPU. Four-device data parallel
# runs opt in with OGPO_ACTOR_GPUS_PER_TASK=4 and expose the complete block.
srun \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --gpus-per-task="${ACTOR_GPUS_PER_TASK}" \
  "${GPU_BIND_ARGS[@]}" \
  bash -c '
    if [[ -n "${OGPO_CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]]; then
      export CUDA_VISIBLE_DEVICES="${OGPO_CUDA_VISIBLE_DEVICES_OVERRIDE}"
      echo "step_cuda_visible_devices_override=${CUDA_VISIBLE_DEVICES}"
    fi
    echo "step_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "step_slurm_step_gpus=${SLURM_STEP_GPUS:-unset}"
    nvidia-smi \
      --query-gpu=index,memory.used,memory.free \
      --format=csv,noheader
    if [[ "${OGPO_GPU_MEMORY_MONITOR_INTERVAL:-0}" -gt 0 ]]; then
      (
        while true; do
          echo "[gpu-memory] time=$(date --iso-8601=seconds)"
          nvidia-smi \
            --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
          sleep "${OGPO_GPU_MEMORY_MONITOR_INTERVAL}"
        done
      ) &
      monitor_pid=$!
      trap "kill ${monitor_pid} 2>/dev/null || true" EXIT
      "$@"
    else
      exec "$@"
    fi
  ' bash "${CMD[@]}"
echo "train_end=$(date --iso-8601=seconds)"
