#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEXJOCO_ENV_PREFIX="${DEXJOCO_ENV_PREFIX:-${ROOT_DIR}/.conda/dexjoco}"
OPENPI_ENV_PREFIX="${OPENPI_ENV_PREFIX:-${ROOT_DIR}/.conda/openpi}"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[setup] repo root: ${ROOT_DIR}"
echo "[setup] dexjoco env prefix: ${DEXJOCO_ENV_PREFIX}"
echo "[setup] openpi env prefix: ${OPENPI_ENV_PREFIX}"

mkdir -p "${ROOT_DIR}/.conda"

echo "[setup] writing openpi/config.yaml from local path defaults"
cat > "${ROOT_DIR}/openpi/config.yaml" <<EOF
pretrained_model_path: "${OPENPI_PRETRAINED_MODEL_PATH:-${ROOT_DIR}/checkpoints/pi05_base/params}"
pretrained_model_action_dim_44_path: "${OPENPI_PRETRAINED_MODEL_ACTION_DIM_44_PATH:-${ROOT_DIR}/checkpoints/pi05_base_action_dim_44/params}"
dataset_root: "${OPENPI_DATASET_ROOT:-${ROOT_DIR}/datasets/dexjoco_lerobot_datasets}"
rand_full_dataset_root: "${OPENPI_RAND_FULL_DATASET_ROOT:-${ROOT_DIR}/datasets/dexjoco_lerobot_datasets_rand_full}"
ckpts_root: "${OPENPI_CKPTS_ROOT:-${ROOT_DIR}/checkpoints/pi05_ckpts}"
rand_full_ckpts_root: "${OPENPI_RAND_FULL_CKPTS_ROOT:-${ROOT_DIR}/checkpoints/pi05_rand_full_ckpts}"
wandb_enabled: true

batch_size: 32
single_arm_steps: 30000
dual_arm_steps: 60000
EOF

if [[ ! -d "${DEXJOCO_ENV_PREFIX}" ]]; then
  echo "[setup] creating DexJoCo env"
  conda env create -p "${DEXJOCO_ENV_PREFIX}" -f "${ROOT_DIR}/environment-dexjoco.yaml"
else
  echo "[setup] DexJoCo env already exists, skip create"
fi

if [[ ! -d "${OPENPI_ENV_PREFIX}" ]]; then
  echo "[setup] creating OpenPI env"
  conda env create -p "${OPENPI_ENV_PREFIX}" -f "${ROOT_DIR}/openpi/environment-openpi.yaml"
else
  echo "[setup] OpenPI env already exists, skip create"
fi

echo "[setup] installing openpi python packages"
conda run -p "${OPENPI_ENV_PREFIX}" pip install lerobot --no-deps
conda run -p "${OPENPI_ENV_PREFIX}" pip install -e "${ROOT_DIR}/openpi"
conda run -p "${OPENPI_ENV_PREFIX}" pip install -e "${ROOT_DIR}/openpi/packages/openpi-client"

TRANSFORMERS_DIR="$(conda run -p "${OPENPI_ENV_PREFIX}" python -c \
  'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')"
echo "[setup] installing OpenPI transformers compatibility files into ${TRANSFORMERS_DIR}"
rsync -a \
  "${ROOT_DIR}/openpi/src/openpi/models_pytorch/transformers_replace/" \
  "${TRANSFORMERS_DIR}/"
conda run -p "${OPENPI_ENV_PREFIX}" python -c \
  'from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly()'

echo "[setup] done"
echo "[setup] activate DexJoCo with: conda activate ${DEXJOCO_ENV_PREFIX}"
echo "[setup] activate OpenPI with: conda activate ${OPENPI_ENV_PREFIX}"
