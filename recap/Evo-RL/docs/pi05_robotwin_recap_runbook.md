# PI05 + RoboTwin + Recap 运行说明

## 目标

在当前 `Evo-RL` 仓库环境较干净、但外部已经存在 `pi05` 权重、`pi05-robotwin` finetune 权重和 `RoboTwin` 环境的前提下，复用外部目录：

- `/nfs_global/S/yangrongzheng/pi05`

跑通 `pi05` 在 `robotwin` 下的 `recap` 链路。

---

## 外部资源位置

外部工作目录：

- `/nfs_global/S/yangrongzheng/pi05`

其中已经包含：

- `RoboTwin` 环境：`/nfs_global/S/yangrongzheng/pi05/external/RoboTwin`
- `pi05` robotwin 权重：`/nfs_global/S/yangrongzheng/pi05/model_robotwin`
- `recap` 工作区：`/nfs_global/S/yangrongzheng/pi05/recap_workspace`

---

## 已确认跑通的两段链路

### 1. RoboTwin rollout 已跑通

现成脚本：

- `/nfs_global/S/yangrongzheng/pi05/scripts/robotwin_pi05_adjust_bottle_smoke.slurm`

它会在：

- `/nfs_global/S/yangrongzheng/pi05/external/RoboTwin`

执行：

- `python script/eval_policy.py --config policy/pi05/deploy_policy.yml ...`

使用参数：

- `TASK_NAME=adjust_bottle`
- `TRAIN_CONFIG=pi05_aloha_full_base`
- `MODEL_NAME=model_robotwin`

已存在真实运行证据：

- 日志：`/nfs_global/S/yangrongzheng/pi05/logs/rt-pi05-adj1-763500.err:1`
- 文档记录：`/nfs_global/S/yangrongzheng/pi05/README_ROBOTWIN_PI05.md:59`

说明：`pi05` 在 `RoboTwin adjust_bottle` rollout 侧已经不是纸面配置，而是已经真实进入仿真执行。

---

### 2. Recap value 训练已进入真实 checkpoint 保存

现成脚本：

- `/nfs_global/S/yangrongzheng/pi05/recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm`

对应配置：

- `/nfs_global/S/yangrongzheng/pi05/recap_workspace/configs/local_value_sft_adjust_bottle_100.yaml`

训练数据：

- `/nfs_global/S/yangrongzheng/pi05/data/lerobot_adjust_bottle_recap_100`

已存在真实训练产物：

- `/nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap_value_757680/value_sft_local_pi05/checkpoints/global_step_5472/actor/model_state_dict/full_weights.pt`

同时存在 DCP checkpoint：

- `/nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap_value_757680/value_sft_local_pi05/checkpoints/global_step_5472/actor/dcp_checkpoint/.metadata`

这说明 `recap value` 链路并不只是完成 import，而是已经真实进入过训练并保存 checkpoint。

---

## 最小复现路径

### A. 复现 RoboTwin rollout

```bash
cd /nfs_global/S/yangrongzheng/pi05
sbatch scripts/robotwin_pi05_adjust_bottle_smoke.slurm
```

关注产物：

- `logs/rt-pi05-adj1-<jobid>.out`
- `logs/rt-pi05-adj1-<jobid>.err`
- `external/RoboTwin/eval_result/adjust_bottle/pi05/demo_clean/model_robotwin`

---

### B. 复现 recap value 训练

```bash
cd /nfs_global/S/yangrongzheng/pi05
sbatch recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm
```

关注产物：

- `recap_workspace/logs/recap-adj100-<jobid>.out`
- `recap_workspace/logs/recap-adj100-<jobid>.err`
- `recap_workspace/logs/recap_value_<jobid>/`

重点检查：

- `worker_logs/ActorGroup/rank_0.log`
- `tensorboard/all/events.out.tfevents.*`
- `value_sft_local_pi05/checkpoints/global_step_*/actor/model_state_dict/full_weights.pt`

---

## 运行这条链路依赖的关键环境

`recap` slurm 里已经处理好了这些核心项：

- conda 环境：`/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final`
- `use_local_openpi_env.sh`
- `REPO_PATH=${ROOT}/vendor/rlinf-recap`
- 独立 `RAY_TMPDIR`
- 显式本地 `ray start --head`
- `RAY_WORKER_PRELOAD_MODULES=recap_workspace.ray_worker_preload`

因此当前最推荐做法是：

- **直接在外部 `pi05` 仓库里用现成 slurm 提交**
- 而不是先试图把整套 `robotwin + recap` 重新搬进当前 `Evo-RL` 仓库

---

## 当前结论

如果把“跑通 pi05 在 robotwin 下的 recap”拆成证据链，当前已经具备：

1. `pi05 + RoboTwin rollout` 已有真实成功运行记录。
2. `recap value train` 已有真实 checkpoint 保存产物。
3. `adjust_bottle recap` 数据与配置文件都已落地。

所以当前最实用的操作不是重新实现，而是：

- 直接复用 `/nfs_global/S/yangrongzheng/pi05` 里的现成脚本
- 重新提交一次 smoke / recap 任务
- 以新的 job 日志作为这次复现的最新证据

