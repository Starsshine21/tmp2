#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_tp2_gn200_10k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-trueppo2-gn200-resume3000-10000-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-trueppo2-gn200-resume3000-10000-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --nodelist=r8a100-a05
#SBATCH --constraint=A100&40G
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH -t 3-23:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
export OGPO_ACTOR_GPUS_PER_TASK=8
export OGPO_FLASH_CONFIG=configs/ogpo/pi05_jax_flash_ogpo_td10_tgr_b8_8gpu_trueppo2_gn200_resume3000_10000.yaml
export OGPO_TRAIN_REPLAY=${ROOT}/outputs/ogpo/click_mouse_pi05_replay_100ep_train.pt
export OGPO_VALIDATION_REPLAY=${ROOT}/outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt
export OGPO_CALIBRATED_CHECKPOINT=${ROOT}/outputs/ogpo/click_mouse_gemma_udivl_100ep_td10_balanced_30k_b32_best_calibrated.pt
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.88
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=false"
export NCCL_BUFFSIZE=4194304
export OGPO_GPU_MEMORY_MONITOR_INTERVAL=30
ulimit -v unlimited 2>/dev/null || true

export JAX_COMPILATION_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ogpo-jax-cache-${SLURM_JOB_ID:-manual}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

exec bash "${ROOT}/scripts/pi05_flash_ogpo_train.slurm"
