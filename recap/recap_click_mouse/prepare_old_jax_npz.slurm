#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_npz_oldref
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/prepare-old-npz-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/prepare-old-npz-%j.err
#SBATCH -p cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
OUT=${ROOT}/outputs/recap-click-mouse
INPUT=${OUT}/raw_rollouts/formal_seed27_100ep_796461
OUTPUT=${OUT}/jax_oldcfg_raw_npz_100ep
MARKER=${OUT}/JAX_OLDCFG_RAW_NPZ_100EP_COMPLETE
PY=${ROOT}/dexjoco/.conda/dexjoco/bin/python

test -d "${INPUT}"
test ! -e "${OUTPUT}"
test ! -e "${MARKER}"

"${PY}" "${ROOT}/recap_click_mouse/scripts/dexjoco_zarr_to_old_jax_npz.py" \
  --input-root "${INPUT}" \
  --output-root "${OUTPUT}" \
  --expected-episodes 100 \
  --expected-frames 66347

test "$(find "${OUTPUT}" -maxdepth 1 -name 'episode_*.npz' | wc -l)" -eq 100
touch "${MARKER}"
echo "old_reference_npz_ready=${OUTPUT}"
