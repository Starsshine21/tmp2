# PI0.5 JAX OGPO 运行手册

更新时间：2026-07-27

实现状态见 `docs/ogpo_implementation_zh.md`，公式见
`docs/ogpo_paper_derivation_zh.md`。

## 1. 固定路径

```bash
ROOT=/nfs_global/S/yangrongzheng/evo-RL/dexjoco
OPENPI_ENV=${ROOT}/.conda/openpi
PI05_JAX_CKPT=/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse
CRITIC_CKPT=${ROOT}/outputs/ogpo/click_mouse_gemma_udivl_calibrated.pt
```

本地 VLM：

```text
/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/gemma-3-270m
/nfs_global/S/yangrongzheng/evo-RL/Evo-RL/local_models/siglip2-so400m-patch14-224-fixed
```

## 2. 本地验证

```bash
cd /nfs_global/S/yangrongzheng/evo-RL/dexjoco
PYTHONPATH=dexjoco:openpi/src:openpi/packages/openpi-client/src \
  .conda/openpi/bin/python -m pytest tests/ogpo -q
```

只跑 JAX actor：

```bash
PYTHONPATH=dexjoco:openpi/src:openpi/packages/openpi-client/src \
  .conda/openpi/bin/python -m pytest \
  tests/ogpo/test_pi05_jax_actor_path.py -q
```

## 3. Adafactor GPU Smoke

2026-07-27 当前验证任务：

```text
Job ID: 786557
stdout: outputs/ogpo/logs/pi05-jax-smoke-786557.out
stderr: outputs/ogpo/logs/pi05-jax-smoke-786557.err
```

结果：PI0.5 全参 backward 通过，随后 Adafactor update 为梯度平方临时量申请
2.25 GiB 时 OOM。已将 optimizer 与参数更新融合并复用 gradient buffer。

修复后重试：

```text
Job ID: 786571
stdout: outputs/ogpo/logs/pi05-jax-smoke-786571.out
stderr: outputs/ogpo/logs/pi05-jax-smoke-786571.err
```

结果：`COMPLETED`，耗时 7 分 18 秒，`actor_loss=0.0228`。产物：

```text
outputs/ogpo/click_mouse_pi05_jax_flash_gpu_smoke.pt
outputs/ogpo/click_mouse_pi05_jax_flash_gpu_smoke.pt.jax/
```

真实 checkpoint 恢复与推理验证：

```text
Job ID: 786576
stdout: outputs/ogpo/logs/pi05-jax-restore-786576.out
stderr: outputs/ogpo/logs/pi05-jax-restore-786576.err
```

结果：首次恢复在读取原始 JAX `params/` 时发现 inference 入口遗漏 Orbax/JAX
`record_scalar` 兼容初始化，未进入 sidecar restore；该入口已修复。

修复后重试：

```text
Job ID: 786578
stdout: outputs/ogpo/logs/pi05-jax-restore-786578.out
stderr: outputs/ogpo/logs/pi05-jax-restore-786578.err
```

结果：base 与 finetuned sidecar 均恢复成功；验证脚本对巨型参数 leaf 整体做
bf16 -> fp32 差异统计时额外申请 4.50 GiB OOM。已改为原 dtype 融合比较，
只对变化 leaf 的稀疏样本升 fp32。

低显存验证重试：

```text
Job ID: 786580
stdout: outputs/ogpo/logs/pi05-jax-restore-786580.out
stderr: outputs/ogpo/logs/pi05-jax-restore-786580.err
```

结果：确认 finetuned PI0.5 backend 参数相对 SFT reference 已改变；随后发现真实
DexJoCo client 只发送 actor 使用的 `base/wrist`，而 critic 使用 replay 侧的
`image_base/image_wrist`。Inference policy 已通过 `flow.image_mapping` 自动
完成在线键到 critic 键的逆映射。

映射修复后重试：Job `786582`，日志
`outputs/ogpo/logs/pi05-jax-restore-786582.{out,err}`。

结果：`COMPLETED`，耗时 1 分 37 秒：

```text
checkpoint_restore_ok ... sample_max_abs_difference=3.0517578e-05
inference_ok action_shape=(30, 22)
reference_divergence=6.6834326e-06
predicted_q=-0.036078759
```

Critic 非零 PPO signal 隔离验证（生产配置仍使用 two-sided sign consensus）：

```text
Job ID: 786584, 786589
config: configs/ogpo/pi05_jax_flash_ogpo_critic_signal_smoke.yaml
logs: outputs/ogpo/logs/pi05-jax-smoke-786584.{out,err}
```

该诊断临时使用 LCB advantage，并关闭 KL/FM/success；若 actor gradient 非零，
只能来自 critic-derived PPO objective。

首次结果：critic-driven PPO backward 和 Adafactor update 已通过，随后 eager
EMA old-policy 同步因 2.25 GiB 临时量 OOM。EMA 已改为融合 JIT；`ema=0`
的完整 JAX backend hard-sync 语义也已修复。

`786589` 完成 fused EMA，但初始 ratio 为 `0.025`，PPO 进入常数 clipped
分支，actor gradient 为零。Flash/Full 现均在同一个 loss trace 内重算
old/current likelihood，避免 scan 与 standalone bf16 数值差被 660 维求和放大。
修复后复验 Job `786596`：`COMPLETED`，6 分 25 秒。关键 metrics：

```text
importance_ratio_mean=1.0
conservative_advantage_abs_mean=0.0815001
flash_ppo_loss=0.0741816
actor_grad_norm=760.6247
fm_anchor_loss=0
reference_kl_beta=0
success_buffer_loss=0
```

该隔离配置中非零 actor gradient 只能来自 critic-derived PPO。

生产正则组合显存 smoke：Job `786597`，配置
`configs/ogpo/pi05_jax_flash_ogpo_regularization_smoke.yaml`。

结果：8 分 03 秒完成，`ratio=1.0`、FM `0.2280`、success `0.7327`、
actor grad `2.1275`，EMA 和 checkpoint 保存通过。

提交：

```bash
cd /nfs_global/S/yangrongzheng/evo-RL/dexjoco
sbatch scripts/pi05_jax_flash_ogpo_gpu_smoke.slurm
```

配置：

```text
configs/ogpo/pi05_jax_flash_ogpo_gpu_smoke.yaml
```

预期完整经过：

```text
load calibrated critic
-> critic_update
-> JAX old-policy rollout
-> endpoint critic advantage
-> JAX PPO + fixed KL + FM value_and_grad
-> Adafactor update
-> EMA old-policy sync
-> .pt + Orbax .pt.jax/ checkpoint
```

日志：

```text
outputs/ogpo/logs/pi05-jax-smoke-<job>.out
outputs/ogpo/logs/pi05-jax-smoke-<job>.err
```

检查：

```bash
sacct -j <job> --format=JobID,State,Elapsed,ExitCode,NodeList
tail -n 100 outputs/ogpo/logs/pi05-jax-smoke-<job>.out
tail -n 100 outputs/ogpo/logs/pi05-jax-smoke-<job>.err
```

成功产物必须同时存在：

```text
outputs/ogpo/click_mouse_pi05_jax_flash_gpu_smoke.pt
outputs/ogpo/click_mouse_pi05_jax_flash_gpu_smoke.pt.jax/
```

## 4. 正式训练

### 4.0 从零提交

先提交 critic，再让 actor 只在 critic 成功退出后启动：

```bash
cd /nfs_global/S/yangrongzheng/evo-RL/dexjoco

critic_job=$(sbatch --parsable scripts/pi05_udivl_critic_train.slurm)
actor_job=$(sbatch --parsable \
  --dependency="afterok:${critic_job}" \
  --export=ALL,\
OGPO_FLASH_CONFIG=configs/ogpo/pi05_jax_flash_ogpo.yaml,\
OGPO_PI05_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse,\
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
  scripts/pi05_flash_ogpo_train.slurm)

printf 'critic=%s actor=%s\n' "${critic_job}" "${actor_job}"
```

提交 actor 前，应先检查 critic 校准日志中的
`pairwise_ranking_accuracy` 是否达到
`configs/ogpo/pi05_gemma_udivl_critic.yaml` 的
`critic.min_ranking_accuracy`。`afterok` 只保证进程成功，不保证 critic
质量门槛通过；actor 还会在运行时执行这层质量门控。

### 4.1 当前正式依赖链

2026-07-27：

```text
critic: 786719, COMPLETED in 00:19:43, exit code 0:0
actor:  786720, CANCELLED after 00:30:14 because every attempted actor step was gated
```

Critic 校准结果中
`pairwise_ranking_accuracy=0.521791`，低于正式配置的
`critic.min_ranking_accuracy=0.55`。因此 actor 虽然已经正常启动，但截至
step 6 均被 `critic_ranking_accuracy_below_min` 门控跳过，没有执行 PI0.5
参数更新。这说明作业依赖和训练链路正常，不代表当前 critic 已达到可驱动正式
actor 实验的质量。

### 4.2 100-episode 正式链路

根据旧 replay 只有 5 个 episode、critic 排序能力接近随机的问题，已提交新的
串行依赖链：

```text
collection:   786734, 100 click_mouse episodes, FAILED only during replay conversion
replay build: 787097, reuse 786734 Zarr, r8cpu, time limit 01:00:00
critic:       787098, afterok:787097, 15000 steps, batch size 8, time limit 04:00:00
actor:        787099, afterok:787098, 500 full-JAX actor steps, time limit 23:30:00
```

采集 instruction：

```text
Move the mouse to the purple mouse pad and click the left mouse button.
```

对应文件：

```text
scripts/pi05_click_mouse_collect_100.slurm
configs/ogpo/pi05_dataset_100ep.yaml
configs/ogpo/pi05_gemma_udivl_critic_100ep.yaml
configs/ogpo/pi05_jax_flash_ogpo_100ep.yaml
```

采集完成后，同一个 Job 会自动把 episode Zarr 转换为：

```text
outputs/ogpo/click_mouse_pi05_replay_100ep.pt
outputs/ogpo/click_mouse_pi05_replay_100ep_train.pt
outputs/ogpo/click_mouse_pi05_replay_100ep_validation.pt
outputs/ogpo/click_mouse_pi05_replay_100ep_heldout.pt
outputs/ogpo/click_mouse_pi05_replay_100ep_success.pt
outputs/ogpo/click_mouse_pi05_replay_100ep_failure.pt
```

Critic 使用单阶段 `full_td`：共享 Gemma3+SigLIP VLM 全量微调，三个 Q/V
heads 只在末端分叉。关闭 early stopping，最多训练 15000 steps；训练 batch
size 为 8，validation batch size 为 32。Value-learning 主干对齐 LWD：

```text
1-step TD
gamma = 0.9999
Q loss = MSE
optimizer = Adam, lr = 5e-4, cosine decay
target EMA update rate = 0.005
TD bootstrap = mean of three DIVL quantiles
categorical support = 201 atoms over [-0.1, 1.1]
offline tau = clip(0.6 - 0.3 * normalized_entropy, 0.5, 0.6)
quantile = first support atom whose CDF reaches tau
```

真实 A100-40G smoke：

```text
786740: full Gemma3+SigLIP backward, batch 8, 1 step,  COMPLETED in 77s
786752: full Gemma3+SigLIP backward, batch 8, 20 steps, COMPLETED in 87s
```

扣除共同启动/保存/校准开销后约 0.53 秒/step，15000 steps 训练主体预计约
2.2 小时，仍在 4 小时时限内。

校准后 Slurm 会强制检查：

```text
pairwise_ranking_accuracy >= 0.55
```

只有该检查通过，`afterok:787098` 才会释放 actor Job `787099`。这既保留
actor 内部质量门控，也避免低质量 critic 启动后长期占用 GPU。

Actor 使用 `batch_size=1`，从 100-episode replay 有放回采样，10 个 flow
timestep 均匀选择。500 steps 使每个 flow timestep 期望获得约 50 次全参数
更新；相比旧 100-step 配置覆盖更充分，同时限制小数据上的 policy drift。
Actor 阶段设置 `critic.steps_per_actor_step=0`：critic 仍负责候选动作评分、
three-head two-sided advantage 和质量门控，但在固定 replay 上不再继续 TD
更新，避免 15000-step critic 进一步过拟合并降低 JAX/PyTorch 共卡显存压力。

首次采集 Job `786728` 暴露出标准 OpenPI 服务入口未初始化 Orbax/JAX
`record_scalar` 兼容层，在加载 checkpoint 时失败，未产生 episode；其依赖
Job `786729/786730` 已取消。新增
`scripts/serve_policy_orbax_compat.py` 后重提 `786734`，已确认原生 JAX
checkpoint 恢复、WebSocket 连接和全部 100 个 episode 均正常。最终成功率为
`59/100`。采集完成后的 replay 转换最初错误使用不含 `zarr` 的
`.conda/openpi`，导致 Job `786734` 非零退出；原始 5.4 GB Zarr 完整保留。
采集脚本现改用同时含 `torch` 和 `zarr` 的
`/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final`。Job `787097`
只复用既有 Zarr 补建 replay，不重新采集。

状态和日志：

```bash
squeue -j 787097,787098,787099 \
  -o '%.18i %.26j %.10T %.12M %.12l %R'

tail -f outputs/ogpo/logs/pi05-build-100ep-787097.out
tail -f outputs/ogpo/logs/pi05-gemma-udivl-787098.out
tail -f outputs/ogpo/logs/pi05-flash-train-787099.out
```

新实验产物：

```text
outputs/ogpo/click_mouse_gemma_udivl_100ep.pt
outputs/ogpo/click_mouse_gemma_udivl_100ep_calibrated.pt
outputs/ogpo/click_mouse_gemma_calibration_100ep.json
outputs/ogpo/click_mouse_pi05_jax_flash_100ep.pt
outputs/ogpo/click_mouse_pi05_jax_flash_100ep.pt.jax/
outputs/ogpo/click_mouse_pi05_jax_flash_100ep_metrics.jsonl
```

日志：

```text
outputs/ogpo/logs/pi05-gemma-udivl-786719.out
outputs/ogpo/logs/pi05-gemma-udivl-786719.err
outputs/ogpo/logs/pi05-flash-train-786720.out
outputs/ogpo/logs/pi05-flash-train-786720.err
```

检查：

```bash
squeue -j 786719,786720 \
  -o '%.18i %.24j %.2t %.10M %.10l %R'
sacct -j 786719,786720 \
  --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode
```

Actor 使用 `afterok:786719`，critic 失败时不会误用不完整 checkpoint 启动。
还应检查 actor 日志和 metrics 中的 `actor_skipped`；仅看到 actor 作业
`RUNNING` 不能证明 PI0.5 正在更新。

正式配置：

```text
configs/ogpo/pi05_jax_flash_ogpo.yaml
```

当前单卡预算：500 actor steps、`batch_size=1`、`group_size=1`、每 25 步
checkpoint，Slurm 最长 23 小时 30 分。正式脚本默认
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`。

提交通用 Slurm 时必须把校验路径指向原生 JAX checkpoint：

```bash
sbatch \
  --export=ALL,\
OGPO_FLASH_CONFIG=configs/ogpo/pi05_jax_flash_ogpo.yaml,\
OGPO_PI05_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse,\
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
  scripts/pi05_flash_ogpo_train.slurm
```

正式产物：

```text
outputs/ogpo/click_mouse_pi05_jax_flash.pt
outputs/ogpo/click_mouse_pi05_jax_flash.pt.jax/
outputs/ogpo/click_mouse_pi05_jax_flash_metrics.jsonl
```

## 5. Resume

Resume 需要 `.pt` 和同名 `.pt.jax/`：

```bash
sbatch \
  --export=ALL,\
OGPO_FLASH_CONFIG=configs/ogpo/pi05_jax_flash_ogpo.yaml,\
OGPO_PI05_CHECKPOINT=/nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse,\
OGPO_FLASH_RESUME=/nfs_global/S/yangrongzheng/evo-RL/dexjoco/outputs/ogpo/click_mouse_pi05_jax_flash.pt,\
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
  scripts/pi05_flash_ogpo_train.slurm
```

缺失 sidecar 时不得把 checkpoint 当作全量微调模型。

## 6. 推理服务

```bash
cd /nfs_global/S/yangrongzheng/evo-RL/dexjoco
PYTHONPATH=dexjoco:openpi/src:openpi/packages/openpi-client/src \
  .conda/openpi/bin/python scripts/serve_ogpo_policy.py \
  --pi05-checkpoint \
    /nfs_global/S/yangrongzheng/evo-RL/click_mouse_ckpt/pi05_dexjoco_ckpt/click_mouse \
  --train-config click_mouse \
  --ogpo-checkpoint outputs/ogpo/click_mouse_pi05_jax_flash.pt \
  --device cuda \
  --port 8000
```

Loader 会从原始 JAX checkpoint 创建结构，再恢复
`click_mouse_pi05_jax_flash.pt.jax/` 中的 finetuned actor。

## 7. 必看指标

Actor：

- `actor_loss`
- `flash_ppo_loss`
- `reference_kl`
- `reference_kl_beta`
- `fm_anchor_loss`
- `success_buffer_loss`
- `actor_grad_norm`
- `importance_ratio_mean/std`
- `ppo_clip_fraction`
- `old_policy_lag`

Critic gate：

- `pairwise_ranking_accuracy`
- `interval_coverage`
- `categorical_entropy`
- `candidate_ensemble_disagreement`
- `support_distance_mean`
- `stop_reason`

任何 non-finite、KL 超阈值、ranking/coverage gate 失败都不能算有效 actor
训练步。
