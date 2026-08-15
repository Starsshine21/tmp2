#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J catq_b2_scratch
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-catq-b2-scratch-30k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-catq-b2-scratch-30k-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 4-00:00:00

set -euo pipefail

export OGPO_CRITIC_CONFIG=configs/ogpo/pi05_gemma_udivl_critic_100ep_td10_catq_b2_scratch_30k_b32.yaml
export OGPO_TRAIN_REPLAY=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_pi05_replay_100ep_train.pt
export OGPO_VALIDATION_REPLAY=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt
export OGPO_CRITIC_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_gemma_udivl_100ep_td10_catq_b2_scratch_30k_b32_final.pt
export OGPO_CALIBRATED_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_gemma_udivl_100ep_td10_catq_b2_scratch_30k_b32_final_calibrated.pt
export OGPO_CALIBRATION_OUTPUT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_gemma_udivl_100ep_td10_catq_b2_scratch_30k_b32_final_calibration.json

exec bash /nfs_global/S/yangrongzheng/evo-RL/dexjoco/scripts/pi05_udivl_critic_train.slurm
