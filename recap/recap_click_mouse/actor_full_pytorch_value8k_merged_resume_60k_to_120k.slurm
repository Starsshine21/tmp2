#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_v8k_r120k
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-full-v8k-resume120k-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/actor-full-v8k-resume120k-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --constraint=L40S&48G
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH -t 4-23:00:00

set -euo pipefail

export RECAP_RESUME_START_STEP=60000
export RECAP_RESUME_TARGET_STEP=120000
exec bash /nfs_global/S/yangrongzheng/evo-RL/recap_click_mouse/actor_full_pytorch_value8k_merged_resume_30k_to_60k.slurm
