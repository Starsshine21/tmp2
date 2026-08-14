#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J critic_fullval_cmp
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-critic-fullval-cmp-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/logs/pi05-critic-fullval-cmp-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-02:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
PYTHON=${ROOT}/.conda/openpi/bin/python
VALIDATION=${ROOT}/outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt
CONFIG=configs/ogpo/pi05_gemma_udivl_critic_100ep.yaml
OUTPUT=${ROOT}/outputs/ogpo/comparisons/critic_fullval_2257

mkdir -p "${OUTPUT}" "${ROOT}/outputs/ogpo/logs"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/dexjoco:${ROOT}/openpi/src:${ROOT}/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

names=(
  legacy_15k
  warmup_best
  td1_30k
  td10_4k
  td10_8k
  td10_12k
)
checkpoints=(
  outputs/ogpo/click_mouse_gemma_udivl_100ep_calibrated.pt
  outputs/ogpo/click_mouse_gemma_udivl_100ep_warmup_b32_calibrated.pt
  outputs/ogpo/click_mouse_gemma_udivl_100ep_td_30k_b32_calibrated.pt
  outputs/ogpo/checkpoints/click_mouse_divl_td10_balanced/milestones/critic_step_04000.pt
  outputs/ogpo/checkpoints/click_mouse_divl_td10_balanced/milestones/critic_step_08000.pt
  outputs/ogpo/checkpoints/click_mouse_divl_td10_balanced/milestones/critic_step_12000.pt
)

cd "${ROOT}"
for index in "${!names[@]}"; do
  name=${names[$index]}
  checkpoint=${checkpoints[$index]}
  if [[ ! -s "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  echo "[fullval] start name=${name} checkpoint=${checkpoint}"
  "${PYTHON}" scripts/evaluate_ogpo.py \
    --checkpoint "${checkpoint}" \
    --config "${CONFIG}" \
    --validation-replay "${VALIDATION}" \
    --calibration-output "${OUTPUT}/${name}.json"
  echo "[fullval] done name=${name}"
done

"${PYTHON}" - "${OUTPUT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.json")):
    metrics = json.loads(path.read_text())
    rows.append({"name": path.stem, **metrics})
keys = [
    "name",
    "pairwise_ranking_accuracy",
    "q_rank_correlation",
    "q_rmse",
    "q_exploitation_gap",
    "interval_coverage",
    "categorical_saturation",
    "q_mean",
]
with (root / "comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows([{key: row.get(key) for key in keys} for row in rows])
print(json.dumps([{key: row.get(key) for key in keys} for row in rows], indent=2))
PY
