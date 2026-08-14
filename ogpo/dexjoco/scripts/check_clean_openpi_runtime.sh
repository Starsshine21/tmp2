#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

conda activate "${PI05_EFFECTIVE_OPENPI_ENV_PREFIX}"

if [[ "${DEXJOCO_STRIP_PYTHONPATH}" == "1" ]]; then
  export PYTHONPATH="${DEXJOCO_CLEAN_PYTHONPATH_DEFAULT}"
fi

python - <<'PY'
import os
import sys
print('python =', sys.executable)
print('PYTHONPATH =', os.environ.get('PYTHONPATH'))
for name in ['openpi', 'lerobot', 'tyro']:
    try:
        mod = __import__(name)
        print(f'{name} = {getattr(mod, "__file__", "<namespace>")}')
    except Exception as exc:
        print(f'{name} IMPORT FAILED = {exc!r}')
print('sys.path[:10] =')
for p in sys.path[:10]:
    print(' ', p)
PY
