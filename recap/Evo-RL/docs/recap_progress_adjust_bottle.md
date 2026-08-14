# Recap 复现进度文档（adjust bottle）

## 当前目标

围绕 `adjust bottle` 任务，已经完成一条用于复现 `recap` 数据准备链路的关键前置流程：

1. 构建本地 value 数据集。
2. 训练 value model。
3. 对数据集执行 value inference，产出 `value / advantage / acp_indicator`。
4. 基于 `acp_indicator == 1` 导出 filtered 数据集，作为后续 policy 训练输入。

当前状态可以概括为：**value model 已实现，ACP 正样本 filtered 数据集已导出并验证可加载。**

---

## 已完成内容

### 1. value 数据集构建脚本

已具备脚本：

- `scripts/prepare_adjust_bottle_value_dataset.py:1`

用途：

- 生成本地 `LeRobotDataset` 格式的 value 数据集。
- 供后续 `lerobot_value_train` 与 `lerobot_value_infer` 使用。

关联环境激活脚本：

- `scripts/activate_recap_env.sh:1`

用途：

- 激活 `robosuite` conda 环境。
- 设置 `PYTHONPATH=$PWD/src` 等复现所需环境变量。

---

### 2. value model 训练脚本

已具备 slurm 脚本：

- `scripts/value_train_adjust_bottle.slurm:1`

用途：

- 基于本地数据集 `local_data/lerobot_adjust_bottle_value` 训练 value model。
- 训练入口为 `python -m lerobot.scripts.lerobot_value_train`。

当前脚本内关键信息：

- 数据集根目录：`local_data/lerobot_adjust_bottle_value`
- value 模型类型：`pistar06`
- 视觉模型目录：`local_models/siglip2-so400m-patch14-224-fixed`
- 语言模型目录：`local_models/gemma-3-270m`
- 默认训练步数：`VALUE_STEPS=200000`
- 默认 batch size：`VALUE_BATCH_SIZE=8`

说明：value model 训练链路已经打通。

---

### 3. value inference 脚本

已具备 slurm 脚本：

- `scripts/value_infer_adjust_bottle_100k.slurm:1`

用途：

- 对 `local_data/lerobot_adjust_bottle_value` 执行 value inference。
- 产出带有 value 相关补充字段的数据。

当前脚本内关键信息：

- 推理入口：`python -m lerobot.scripts.lerobot_value_infer`
- 数据集根目录：`local_data/lerobot_adjust_bottle_value`
- 默认 checkpoint：`outputs/value_train_adjust_bottle_763533/checkpoints/100000`
- 打开 ACP：`--acp.enable=true`

这一步的目标是为样本打上如下字段：

- `complementary_info.value`
- `complementary_info.advantage`
- `complementary_info.acp_indicator`

说明：value inference 链路已经准备好，并且从当前交付结果看，数据中这些字段已经存在，说明这条链路已经被实际跑通过至少一次。

---

### 4. filtered 数据集导出脚本

已完成脚本：

- `scripts/export_filtered_dataset.py:1`

用途：

- 读取：`local_data/lerobot_adjust_bottle_value`
- 按字段：`complementary_info.acp_indicator == 1`
- 导出新的 filtered 数据集到：`local_data/lerobot_adjust_bottle_filtered`

当前筛选规则：

- 仅保留 `complementary_info.acp_indicator == 1` 的帧。
- 且每个 episode 至少保留 `4` 帧。

这意味着当前版本已经具备一个**基于 ACP 正样本帧的 filtered 数据集导出器**。

---

## 实际运行结果

### filtered 数据导出结果

源数据：

- `local_data/lerobot_adjust_bottle_value`

导出结果：

- `local_data/lerobot_adjust_bottle_filtered`

保留下来的数据量：

- `kept_episodes = 19`
- `kept_frames = 108`

摘要文件：

- `local_data/lerobot_adjust_bottle_filtered/filter_summary.json`

---

## 数据验证结果

导出后的 filtered 数据已经实际验证可以被 `LeRobotDataset` 正常加载。

验证结果：

- `num_frames = 108`
- `num_episodes = 19`

数据列正常存在：

- `observation.images.front`
- `observation.images.wrist`
- `observation.state`
- `task`
- `timestamp`
- `frame_index`
- `episode_index`
- `index`
- `task_index`
- `complementary_info.value`
- `complementary_info.advantage`
- `complementary_info.acp_indicator`

这说明 filtered 数据集不仅导出成功，而且元数据与 parquet 结构也是自洽的。

---

## 当前产物清单

脚本：

- `scripts/activate_recap_env.sh:1`
- `scripts/prepare_adjust_bottle_value_dataset.py:1`
- `scripts/value_train_adjust_bottle.slurm:1`
- `scripts/value_infer_adjust_bottle_100k.slurm:1`
- `scripts/export_filtered_dataset.py:1`

filtered 数据目录：

- `local_data/lerobot_adjust_bottle_filtered`

关键文件：

- `local_data/lerobot_adjust_bottle_filtered/filter_summary.json`
- `local_data/lerobot_adjust_bottle_filtered/meta/info.json`
- `local_data/lerobot_adjust_bottle_filtered/meta/tasks.parquet`
- `local_data/lerobot_adjust_bottle_filtered/meta/stats.json`
- `local_data/lerobot_adjust_bottle_filtered/meta/episodes/chunk-000/file-000.parquet`
- `local_data/lerobot_adjust_bottle_filtered/data/chunk-000/file-000.parquet`

---

## 当前已经完成到什么程度

如果把“复现 recap”拆成一个最小闭环，目前已经完成了前半段中最关键的部分：

- 已有 value dataset。
- 已有 value training 脚本。
- 已有 value inference 脚本。
- 已有 filtered dataset 导出脚本。
- 已成功导出并验证 filtered 数据集。

因此，当前进度可以定义为：

**已经完成 recap 复现中的“value 打分 + ACP 过滤数据构建”阶段。**

---

## 如果现在要继续复现 recap，下一步做什么

要继续复现 `recap`，下一步最直接的是进入 **policy training / policy evaluation** 阶段，也就是：

### Step 1：确认 recap 的目标 policy 是什么

需要先明确你想复现的是哪一种 policy 训练方式：

- 用 `local_data/lerobot_adjust_bottle_filtered` 直接做 imitation / finetune。
- 继续沿用当前工程里的某个 policy（例如 `pi05`）做训练。
- 还是要严格对齐你之前的 `recap` 实验配置。

如果不先固定 policy 配方，后面虽然能训练，但不一定叫“复现同一个 recap 实验”。

### Step 2：写一个针对 filtered 数据集的 policy 训练脚本

这是当前最缺的关键环节。

已有数据：

- `local_data/lerobot_adjust_bottle_filtered`

下一步需要新增：

- 一个训练 policy 的 slurm 脚本。

这个脚本至少要明确：

- 用哪个 policy 类型。
- 数据集根目录指向哪个路径。
- 训练输出目录。
- batch size / steps / save freq。
- 是否多卡。

也就是说，**现在最推荐的直接下一步是：补一个“用 filtered 数据训练 policy”的 slurm 脚本并跑通。**

### Step 3：跑 policy 训练，拿到 checkpoint

拿到 policy checkpoint 之后，才进入真正意义上的“recap 结果复现”后半段。

你至少需要沉淀：

- policy checkpoint
- 训练日志
- 训练配置

### Step 4：跑 evaluation / rollout

如果你的“复现 recap”目标包含最终任务表现，那么还需要：

- 一个 evaluation 脚本或 slurm。
- 固定评测环境与 checkpoint。
- 输出 success rate / episode return / 轨迹录像等。

没有这一步，当前只能算“复现了数据构造流程”，还不能算“复现了 recap 的最终实验结果”。

---

## 建议的最小复现路径

建议按下面顺序推进：

1. 确认 recap 对应的 policy 类型和训练入口。
2. 新建一个使用 `local_data/lerobot_adjust_bottle_filtered` 的 policy 训练 slurm。
3. 先跑 smoke test（极少步数）验证数据和训练链路。
4. 再跑正式训练。
5. 产出 checkpoint 后补 evaluation 脚本。
6. 最后汇总训练与评测结果，形成真正完整的 recap 复现记录。

---

## 当前可直接复现的命令

### 环境激活

```bash
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate robosuite
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

### 导出 filtered 数据集

```bash
python scripts/export_filtered_dataset.py
```

---

## 当前结论

截至目前，`adjust bottle` 相关 recap 复现工作已经完成了：

- value model 相关实现
- ACP 正样本筛选逻辑
- filtered 数据集导出
- filtered 数据集加载验证

**如果现在继续往前推进，最优先的下一步不是再改 value，而是开始补 policy training + evaluation 这两个环节。**
