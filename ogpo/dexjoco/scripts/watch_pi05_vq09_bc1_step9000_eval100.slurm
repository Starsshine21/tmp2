#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J watch_vq09_9000
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/watchdog/pi05_vq09_bc1_step9000_seed27_eval100-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/watchdog/pi05_vq09_bc1_step9000_seed27_eval100-%j.err
#SBATCH -p cpu
#SBATCH --qos=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 2-01:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
exec env \
  ROOT="${ROOT}" \
  TIMEOUT_SECONDS=172800 \
  POLL_SECONDS=30 \
  EXISTING_EVAL_JOB_ID="${EXISTING_EVAL_JOB_ID:-}" \
  "${ROOT}/scripts/watch_pi05_vq09_bc1_step9000_eval100.sh"
