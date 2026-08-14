#!/bin/bash
#SBATCH -A yangrongzheng
#SBATCH -J jax_mp_8g_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/jax-mp-8gpu-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/evo-RL/outputs/recap-click-mouse/logs/jax-mp-8gpu-smoke-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --constraint=A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH -t 00:10:00

set -euo pipefail
ROOT=/nfs_global/S/yangrongzheng/evo-RL
mkdir -p "${ROOT}/outputs/recap-click-mouse/logs"
export JAX_COORDINATOR_ADDRESS="$(hostname):$((25000 + SLURM_JOB_ID % 20000))"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_DEBUG=WARN

srun --ntasks=8 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3,4,5,6,7 \
  "${ROOT}/dexjoco/.conda/openpi/bin/python" - <<'PY'
import os
import numpy as np
import jax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

rank = int(os.environ["SLURM_PROCID"])
world = int(os.environ["SLURM_NTASKS"])
jax.distributed.initialize(
    coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
    num_processes=world,
    process_id=rank,
    local_device_ids=[0],
)
mesh = jax.make_mesh((world,), ("fsdp",))
s = jax.sharding.NamedSharding(mesh, P("fsdp"))
x = jax.device_put(np.arange(world, dtype=np.float32), s)
f = jax.jit(shard_map(lambda a: jax.lax.psum(a, "fsdp"), mesh=mesh, in_specs=P("fsdp"), out_specs=P("fsdp")))
y = f(x)
print(
    f"rank={rank} process_count={jax.process_count()} local_devices={jax.local_devices()} "
    f"local_result={np.asarray(y.addressable_data(0))}",
    flush=True,
)
PY
