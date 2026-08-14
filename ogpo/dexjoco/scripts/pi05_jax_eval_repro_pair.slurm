#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J pi05_eval_repro
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-eval-repro-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-eval-repro-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 0-01:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
CHECKPOINT=${OGPO_CHECKPOINT:?OGPO_CHECKPOINT must name the actor checkpoint}
EPISODES=${PI05_EPISODES:-1}
SEED=${PI05_SEED:-27}
PAIR_ROOT=${PI05_REPRO_OUTPUT_ROOT:-${ROOT}/outputs/pi05_rollouts/repro_pair_${SLURM_JOB_ID}}

for RUN in 1 2; do
  PI05_EPISODES=${EPISODES} \
  PI05_SEED=${SEED} \
  PI05_STRICT_REPRODUCIBILITY=true \
  PI05_VERIFY_POLICY_REPEATABILITY=true \
  OGPO_CHECKPOINT=${CHECKPOINT} \
  PI05_EVAL_OUTPUT_DIR=${PAIR_ROOT}/run${RUN} \
  bash "${ROOT}/scripts/pi05_jax_ogpo_eval_100.slurm"
done

"${ROOT}/.conda/dexjoco/bin/python" \
  "${ROOT}/scripts/compare_eval_reproducibility.py" \
  "${PAIR_ROOT}/run1" \
  "${PAIR_ROOT}/run2"
