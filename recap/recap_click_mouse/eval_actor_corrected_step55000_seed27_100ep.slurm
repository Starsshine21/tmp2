#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_fix55k_eval100
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/fix55k-eval100-seed27-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/fix55k-eval100-seed27-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40|L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 0-06:00:00

set -euo pipefail

export RECAP_EVAL_STEP=55000
exec bash /nfs_global/S/yangrongzheng/evo-RL/recap_click_mouse/eval_actor_corrected_step30000_seed27_100ep.slurm
