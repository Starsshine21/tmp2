#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_critic_warm_b32
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-critic-warm-b32-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-critic-warm-b32-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 1-00:00:00

set -euo pipefail

export OGPO_CRITIC_CONFIG=configs/ogpo/pi05_gemma_udivl_critic_100ep_warmup_b32.yaml
export OGPO_TRAIN_REPLAY=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_pi05_replay_100ep_train.pt
export OGPO_VALIDATION_REPLAY=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt
export OGPO_CRITIC_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_gemma_udivl_100ep_warmup_b32.pt
export OGPO_CALIBRATED_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_gemma_udivl_100ep_warmup_b32_calibrated.pt
export OGPO_CALIBRATION_OUTPUT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_gemma_udivl_100ep_warmup_b32_calibration.json
export OGPO_MIN_RANKING_ACCURACY=0.55

exec bash /nfs_global/S/yangrongzheng/evo-RL/dexjoco/scripts/pi05_udivl_critic_train.slurm
