#!/bin/bash
set -uo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
SCRIPT=${ROOT}/recap_click_mouse/actor_jax_oldcfg_qwen10k_100ep.slurm
LOG=${ROOT}/outputs/recap-click-mouse/logs/actor-formal-submit-watchdog.log

while true; do
  if squeue -h -u "${USER}" -n recap_jax_g3s2_8g | grep -q .; then
    printf '%s actor job already present; watchdog exiting\n' "$(date --iso-8601=seconds)" >> "${LOG}"
    exit 0
  fi

  output=$(sbatch --dependency=afterok:803847 "${SCRIPT}" 2>&1)
  status=$?
  printf '%s status=%s %s\n' "$(date --iso-8601=seconds)" "${status}" "${output}" >> "${LOG}"
  if [[ ${status} -eq 0 ]]; then
    exit 0
  fi
  if [[ "${output}" != *QOSMaxSubmitJobPerUserLimit* ]]; then
    exit "${status}"
  fi
  sleep 60
done
