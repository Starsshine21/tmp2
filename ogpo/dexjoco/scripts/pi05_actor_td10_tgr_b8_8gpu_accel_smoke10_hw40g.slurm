#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_8g_acc_40g
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-8gpu-accel-40g-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-8gpu-accel-40g-smoke-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --nodelist=r8a100-a05
#SBATCH --constraint=A100&40G
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH -t 04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
export OGPO_ACTOR_GPUS_PER_TASK=8
export OGPO_FLASH_CONFIG=configs/ogpo/pi05_jax_flash_ogpo_td10_tgr_b8_8gpu_accel_resume250_smoke10.yaml
export OGPO_TRAIN_REPLAY=${ROOT}/outputs/ogpo/click_mouse_pi05_replay_100ep_train.pt
export OGPO_VALIDATION_REPLAY=${ROOT}/outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt
export OGPO_PI05_CHECKPOINT=${ROOT}/checkpoints/pi05_pytorch/click_mouse/model.safetensors
export OGPO_CALIBRATED_CHECKPOINT=${ROOT}/outputs/ogpo/click_mouse_gemma_udivl_100ep_td10_balanced_30k_b32_best_calibrated.pt

# Reserve about 4.8GB per GPU for PyTorch/CUDA and cross-framework runtime.
# The measured 8-GPU JAX working set is below this 35.2GB cap.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.88
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=false"
export NCCL_BUFFSIZE=4194304
export OGPO_GPU_MEMORY_MONITOR_INTERVAL=10
ulimit -v unlimited 2>/dev/null || true

export JAX_COMPILATION_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ogpo-jax-cache-${SLURM_JOB_ID:-manual}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

exec bash "${ROOT}/scripts/pi05_flash_ogpo_train.slurm"
