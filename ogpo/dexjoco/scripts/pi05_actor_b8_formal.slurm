#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_actor_b8
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-actor-b8-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-actor-b8-%j.err
#SBATCH -p r8nv-gpu-hw-80g
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 1-06:00:00
#SBATCH --exclude=r8a100-d05

set -euo pipefail

export OGPO_FLASH_CONFIG=configs/ogpo/pi05_jax_flash_ogpo_100ep_b8.yaml
export OGPO_TRAIN_REPLAY=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_pi05_replay_100ep_train.pt
export OGPO_VALIDATION_REPLAY=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt
export OGPO_PI05_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse
export OGPO_CALIBRATED_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_gemma_udivl_100ep_warmup_b32_calibrated.pt

exec bash /nfs_global/S/yangrongzheng/evo-RL/dexjoco/scripts/pi05_flash_ogpo_train.slurm
