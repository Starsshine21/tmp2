#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J watch_tgr3000
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/watchdog/pi05_tgr_8gpu_step3000_seed27_eval100-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/watchdog/pi05_tgr_8gpu_step3000_seed27_eval100-%j.err
#SBATCH -p cpu
#SBATCH --qos=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 2-01:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
exec env ROOT="${ROOT}" TIMEOUT_SECONDS=172800 POLL_SECONDS=30 \
  "${ROOT}/scripts/watch_pi05_tgr_8gpu_step3000_eval100.sh"
