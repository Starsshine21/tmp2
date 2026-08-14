# DexJoCo + π0.5 本地环境配置说明

这个说明面向当前仓库目录下的使用方式：

- `DexJoCo` 仓库：`/nfs_global/S/yangrongzheng/evo-RL/dexjoco`
- 目标：在 `dexjoco` 自带的 `openpi` 里直接启动 `π0.5`，随后在 `dexjoco` 环境里做评测或采集
- 优先不依赖 `../pi05` 的运行时代码；但如果 `../pi05` 已经有可用 `openpi` 环境和 base 权重，也可以直接复用

## 1. 已补充的脚本

- 环境初始化：`dexjoco/scripts/setup_local_envs.sh`
- 统一变量：`dexjoco/scripts/common_env.sh`
- 启动 π0.5 server：`dexjoco/scripts/run_pi05_server.sh`
- 运行评测：`dexjoco/scripts/run_pi05_eval.sh`
- 采集 demo：`dexjoco/scripts/run_pi05_collect.sh`

这些脚本默认：

- `dexjoco` 侧使用本仓库下的 `dexjoco/.conda/dexjoco`
- `openpi` 侧优先复用 `../pi05/.conda-pi05-openpi-final`

这样可以避开当前机器上默认 `conda env create` 没有可写 env 目录的问题。

## 2. 推荐目录约定

默认路径都通过 `dexjoco/scripts/common_env.sh` 管理：

- 默认复用的基础权重：`/nfs_global/S/yangrongzheng/pi05/openpi_official/checkpoints/pi05_droid/params`
- 双臂 44 维权重：`dexjoco/checkpoints/pi05_base_action_dim_44/params`
- rand-obj 数据集：`dexjoco/datasets/dexjoco_lerobot_datasets`
- rand-full 数据集：`dexjoco/datasets/dexjoco_lerobot_datasets_rand_full`
- rand-obj finetune ckpt：`dexjoco/checkpoints/pi05_ckpts`
- rand-full finetune ckpt：`dexjoco/checkpoints/pi05_rand_full_ckpts`

如果你要复用外部权重或外部 `openpi` 环境，不需要挪文件，只要在启动前覆盖环境变量即可，例如：

```bash
export OPENPI_PRETRAINED_MODEL_PATH=/path/to/pi05_base/params
export OPENPI_PRETRAINED_MODEL_ACTION_DIM_44_PATH=/path/to/pi05_base_action_dim_44/params
export OPENPI_CKPTS_ROOT=/path/to/your/pi05_ckpts
export OPENPI_ENV_PREFIX=/path/to/your/openpi_env
```

## 3. 创建环境

在仓库根目录执行：

```bash
cd /nfs_global/S/yangrongzheng/evo-RL/dexjoco
bash scripts/setup_local_envs.sh
```

如果你已经复用 `../pi05/.conda-pi05-openpi-final`，这一步主要是补 `dexjoco` 自己的本地环境。

## 4. 启动 π0.5 policy server

例子：启动 `water_plant` 的 `rand_obj` 策略。

```bash
cd /nfs_global/S/yangrongzheng/evo-RL/dexjoco
export PI05_TASK=water_plant
export PI05_CONFIG_SET=rand_obj
export PI05_POLICY_DIR=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/checkpoints/pi05_ckpts/water_plant/<exp_name>/<step>
bash scripts/run_pi05_server.sh
```

如果是 `rand_full`：

```bash
export PI05_CONFIG_SET=rand_full
export PI05_POLICY_DIR=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/checkpoints/pi05_rand_full_ckpts/water_plant_rand_full/<exp_name>/<step>
bash scripts/run_pi05_server.sh

如果你现在还在下载 DexJoCo 任务权重，这一步可以先不跑；
当前默认已经指向 `../pi05` 里的基础 `π0.5` 权重和 `openpi` 环境，等你把 DexJoCo checkpoint 下载完后，
只需要把 `PI05_POLICY_DIR` 指到新目录即可。
```

## 5. 评测 π0.5

另开一个终端：

```bash
cd /nfs_global/S/yangrongzheng/evo-RL/dexjoco
export PI05_TASK=water_plant
export PI05_CONFIG_SET=rand_obj
export PI05_EPISODES=50
bash scripts/run_pi05_eval.sh
```

## 6. 采集 demo

当前仓库里的 `record_demos_zarr.py` 是 `DexJoCo` 自带的 teleoperation 采集脚本，
会把成功 episode 存成 zarr + mp4。

它不是通过 `π0.5 policy server` 自动 rollout 采集，而是用于交互式示教采集。

如果你要的是“先用人控/teleop 采数据，再用 `π0.5` 训练”，用这个脚本即可。

```bash
cd /nfs_global/S/yangrongzheng/evo-RL/dexjoco
export PI05_TASK=water_plant
export PI05_CONFIG_SET=rand_obj
export SUCCESS_NEEDED=20
export OUTPUT_DIR=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/recordings/water_plant_seed0
bash scripts/run_pi05_collect.sh
```

如果你要的是“用已经训好的 `π0.5` policy 自动 rollout 并保存 episode”，则需要额外补一层
基于 `dexjoco-openpi-eval` 的 rollout 保存逻辑；这部分当前仓库还没有现成脚本。

## 7. 你当前场景下的建议

你之前 `../pi05` 主要是 `robotwin` 链路；现在切到 `dexjoco` 时建议分开：

- `dexjoco` 仿真、评测、采集：都走当前仓库下的 `dexjoco` + `openpi`
- 只复用 `π0.5` 基础权重或你已有 finetune 权重
- 不再耦合 `../pi05` 里的启动脚本，避免环境变量和依赖互相污染

## 8. 还缺什么

要真正跑起来，还需要你准备至少一项：

1. `π0.5 base` 权重目录
2. 或者某个 DexJoCo 任务的 finetune checkpoint 目录

如果你下一步是“先采数据再训”，那至少先要有一个可推理的 policy：

- 可以用 DexJoCo 官方提供的 `DexJoCo-Pi05` checkpoint
- 或者你自己的 DexJoCo finetune checkpoint

如果你要，我下一步可以继续帮你：

- 直接把基础权重/任务权重下载到这里
- 先选一个任务做 smoke test
- 再补一个把采集数据转成你后续训练目录结构的脚本
