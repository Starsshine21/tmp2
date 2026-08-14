#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J recap_dl_oldcfg
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/download-oldcfg-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/download-oldcfg-%j.err
#SBATCH -p cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/evo-RL
MODEL_ROOT=${ROOT}/Evo-RL/local_models
OUT=${ROOT}/outputs/recap-click-mouse
VALUE_PY=/nfs_global/S/yangrongzheng/pi05/recap_workspace/.venv_recap311/bin/python
QWEN_DIR=${MODEL_ROOT}/qwen2.5-0.5b
SIGLIP_DIR=${MODEL_ROOT}/siglip-so400m-patch14-384

mkdir -p "${OUT}/logs"
test -f "${QWEN_DIR}/config.json"
test -f "${SIGLIP_DIR}/config.json"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "download_oldcfg_start=$(date --iso-8601=seconds)"
download_until_complete() {
  local url=$1
  local output=$2
  local expected_size=$3
  while [[ "$(stat -c '%s' "${output}" 2>/dev/null || echo 0)" -lt "${expected_size}" ]]; do
    curl -L --fail --silent --show-error -C - "${url}" -o "${output}" || true
    sleep 5
  done
  test "$(stat -c '%s' "${output}")" -eq "${expected_size}"
}

download_until_complete \
  https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B/resolve/master/model.safetensors \
  "${QWEN_DIR}/model.safetensors" 988097824 &
qwen_pid=$!
download_until_complete \
  https://www.modelscope.cn/models/google/siglip-so400m-patch14-384/resolve/master/model.safetensors \
  "${SIGLIP_DIR}/model.safetensors" 3511950624 &
siglip_pid=$!
wait "${qwen_pid}"
wait "${siglip_pid}"

"${VALUE_PY}" - "${QWEN_DIR}" "${SIGLIP_DIR}" <<'PY'
import gc
import sys
from transformers import AutoModel

for path in sys.argv[1:]:
    print(f"load_validation_start={path}", flush=True)
    model = AutoModel.from_pretrained(path, local_files_only=True, low_cpu_mem_usage=True)
    print(f"load_validation_ok={path} class={type(model).__name__}", flush=True)
    del model
    gc.collect()
PY

touch "${OUT}/OLDCFG_BACKBONES_COMPLETE"
ls -lh "${QWEN_DIR}/model.safetensors" "${SIGLIP_DIR}/model.safetensors"
echo "download_oldcfg_end=$(date --iso-8601=seconds)"
