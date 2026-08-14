#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J jax_nccl_8g_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/jax-nccl-8gpu-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/jax-nccl-8gpu-smoke-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH -t 00:10:00

set -euo pipefail
ROOT=/nfs_global/S/yangrongzheng/evo-RL
mkdir -p "${ROOT}/outputs/recap-click-mouse/logs"
export LD_LIBRARY_PATH="${ROOT}/recap_click_mouse/runtime/nccl_2_26_5/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${ROOT}/recap_click_mouse/runtime/nccl_2_26_5/nvidia/nccl/lib/libnccl.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
echo "shm=$(df -h /dev/shm | tail -1)"
echo "memlock=$(ulimit -l)"
export NCCL_DEBUG=INFO
export NCCL_COMM_BLOCKING=1
export NCCL_LAUNCH_MODE=PARALLEL
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"${ROOT}/dexjoco/.conda/openpi/bin/python" - <<'PY'
import torch
ts = [torch.ones(1, device=f"cuda:{i}") for i in range(torch.cuda.device_count())]
torch.cuda.nccl.all_reduce(ts)
torch.cuda.synchronize()
print("torch_result", [t.item() for t in ts])

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

n = jax.device_count()
mesh = jax.make_mesh((n,), ("fsdp",))
s = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp"))
x = jax.device_put(jnp.arange(n, dtype=jnp.float32), s)
f = jax.jit(shard_map(lambda a: jax.lax.psum(a, "fsdp"), mesh=mesh, in_specs=P("fsdp"), out_specs=P("fsdp")))
y = f(x)
print("devices", jax.devices())
print("result", jax.device_get(y))
PY
