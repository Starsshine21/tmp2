#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_ogpo_4gpu
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-flash-4gpu-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-flash-4gpu-%j.err
#SBATCH -p r8nv-gpu-hw-80g
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH -t 1-06:00:00
#SBATCH --exclude=r8a100-d05

set -euo pipefail

export OGPO_ACTOR_GPUS_PER_TASK=4
export OGPO_FLASH_CONFIG=${OGPO_FLASH_CONFIG:-configs/ogpo/pi05_jax_flash_ogpo_td10_balanced_b8_4gpu_smoke10.yaml}
export OGPO_TRAIN_REPLAY=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_pi05_replay_train.pt
export OGPO_VALIDATION_REPLAY=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_pi05_replay_validation.pt
export OGPO_PI05_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/checkpoints/pi05_pytorch/click_mouse/model.safetensors
export OGPO_CALIBRATED_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_gemma_udivl_100ep_td10_balanced_30k_b32_best_calibrated.pt
# Replicated PI0.5 states and an 8-candidate local backward need more than the
# old single-GPU 65% cap. GPU0 retains roughly 12GB for the PyTorch critic.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# Keep scheduling deterministic. Gradient aggregation is point-to-point on
# GPU0, so no multi-GiB NCCL collective workspace is required.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=false"
export NCCL_BUFFSIZE=4194304
export OGPO_GPU_MEMORY_MONITOR_INTERVAL=10

exec bash /nfs_global/S/yangrongzheng/evo-RL/dexjoco/scripts/pi05_flash_ogpo_train.slurm
