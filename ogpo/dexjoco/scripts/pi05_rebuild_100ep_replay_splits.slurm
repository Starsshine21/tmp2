#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J rebuild_pi05_data
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-rebuild-replay-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-rebuild-replay-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH -t 0-04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
DATASET_ENV=/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
cd "${ROOT}"

"${DATASET_ENV}/bin/python" scripts/build_offline_dataset.py \
  --config configs/ogpo/pi05_dataset_100ep_rebuild_splits.yaml \
  --source zarr \
  --input-zarr-root outputs/pi05_rollouts/click_mouse_jax_100ep_786734 \
  --save-only-splits train validation

test -s outputs/ogpo/click_mouse_pi05_replay_train.pt
test -s outputs/ogpo/click_mouse_pi05_replay_validation.pt
ls -lh \
  outputs/ogpo/click_mouse_pi05_replay_train.pt \
  outputs/ogpo/click_mouse_pi05_replay_validation.pt
