# PI0.5 + DIVL-OGPO 训练进展（2026-08-05）

## 1. 一句话总结

当前已经完成 **100 条 Click Mouse 轨迹采集、三组 Q-V Critic 训练、PI0.5 JAX
全量参数更新、checkpoint 保存与 100-episode 评测链路**。Critic 已具备明显高于随机
的排序能力；Actor 已确认能够被 Critic 信号推动并稳定保存，但截至目前的评测尚未证明
成功率高于初始 PI0.5。

当前任务：

- Critic：训练完成，Actor 固定使用训练步数为 12k 的 calibrated checkpoint；
- Actor：已有 step 50 和 step 100 完整 checkpoint；
- 最新 5-step 显存修复 smoke `795734` 已完成 step 100--104 并正常保存；
- 正式续训任务 `795735` 已由 smoke 依赖自动释放，正在从 step 100 继续到 step 500。

## 2. 任务与数据

任务指令：

```text
Move the mouse to the purple mouse pad and click the left mouse button.
```

数据由初始 PI0.5 在 DexJoCo Click Mouse 环境中采集：

| 项目 | 数量 |
|---|---:|
| Episode | 100 |
| 成功 Episode | 59 |
| 失败 Episode | 41 |
| Transition / action chunk | 17,478 |
| 成功样本 | 5,465 |
| 失败样本 | 12,013 |
| Train | 13,551 |
| Validation | 2,257 |
| Held-out | 1,670 |

每个动作 chunk 包含 30 个时间步，每步动作维数为 22；环境实际执行前 4 步后重新规划。

## 3. 整体流程

```text
100 条 PI0.5 环境轨迹
        |
        v
构造 10-step 外层 TD 样本
        |
        v
训练共享 Gemma3 + SigLIP 的三组 Q-V Critic
        |
        v
固定并校准 Critic
        |
        v
每个 replay state 用 PI0.5 生成 4 个候选动作
        |
        v
三个 Q-V 对进行 two-sided 同号判断
        |
        v
Flash-OGPO：每个状态抽一个去噪步骤反向
        |
        v
KL + success-buffer BC 约束下全量更新 PI0.5 JAX 参数
        |
        v
保存 .pt 元数据和 .pt.jax 完整 Actor 参数
        |
        v
DexJoCo 100-episode 评测
```

## 4. Critic 方法

### 4.1 架构

Critic 使用共享的 Gemma3 + SigLIP 多模态 backbone，并连接三组独立的 Q/V head：

```text
两路 RGB + 语言 + 机器人状态
              |
              v
共享 Gemma3 + SigLIP backbone（全量微调）
              |
      +-------+-------+
      |       |       |
    Q1,V1   Q2,V2   Q3,V3
```

三个 Q-V 对不是三套 VLM；它们共享视觉语言特征，只在末端 head 上独立。

### 4.2 10-step TD target

对于长度为 10 的外层状态转移，目标值为：

$$
y_t = \sum_{k=0}^{9}\gamma^k r_{t+k}
      + \gamma^{10}(1-d_{t:t+9})\,\overline V^{-}(s_{t+10}),
$$

其中当前采用三个 target V 的平均值：

$$
\overline V^{-}(s') = \frac{1}{3}\sum_{m=1}^{3}V_m^{-}(s').
$$

这样一次 TD 更新能看到更长的真实奖励区间，同时避免原先使用最小 V 导致 target 系统性偏低。

### 4.3 Q 与分布式 V 的训练

Q 使用 MSE：

$$
\mathcal L_Q = \frac{1}{3}\sum_{m=1}^{3}
\left(Q_m(s_t,a_t)-y_t\right)^2.
$$

V 使用 DIVL 的分位数分布表示。设固定 atoms 为 $z_1,\ldots,z_K$，V head 输出概率
$p_{m,k}(s)$：

$$
V_m(s)=\sum_{k=1}^{K}p_{m,k}(s)z_k.
$$

训练时将 TD target 投影到 atoms 上，并使用交叉熵学习目标分布。总 Critic loss 为：

$$
\mathcal L_{critic}=\mathcal L_Q+\lambda_Z\mathcal L_{DIVL}.
$$

主要训练设置：batch size 32、microbatch 8、Adam、head 学习率 $5\times10^{-4}$、
backbone 学习率 $2\times10^{-5}$、target EMA 系数 0.005、无 early stopping，完整训练
30k steps，同时保存 4k/8k/12k 里程碑。

### 4.4 Critic 结果

在完整 validation 集上：

| Critic | Pairwise ranking | Rank correlation | Q RMSE | Q mean |
|---|---:|---:|---:|---:|
| 旧 15k | 0.6308 | -0.0091 | 0.2838 | 0.1066 |
| TD10 4k | 0.6892 | 0.2317 | 0.3695 | 0.3370 |
| TD10 8k | 0.7718 | 0.3344 | 0.3447 | 0.3075 |
| **TD10 12k** | **0.7803** | **0.4493** | **0.3442** | **0.3053** |
| TD1 30k | 0.4093 | -0.0733 | 0.2944 | 0.0062 |

结论：TD10 12k 在排序准确率和 rank correlation 上最好，因此 Actor 使用该版本。Q
mean 约 0.305 不是要求每个状态都接近 1；它反映当前 replay 中大量非终止、失败和远离
成功的状态。对 Actor 更关键的是候选动作之间的排序方向，而不是所有 Q 的绝对值都为 1。

成功/失败轨迹可视化显示：

- 成功轨迹 Q 从约 0.30 开始，在最终完成点击时升至约 1.04；
- 失败轨迹整体约 0.22--0.33，但中间出现过约 0.83 的峰值；
- 失败峰值说明失败轨迹中仍包含“移动到鼠标垫附近”等有价值片段；
- 当前 Critic 能识别最终成功，但仍存在中间状态过高估计和全轨迹校准误差。

## 5. Actor 方法

### 5.1 候选动作和保守 advantage

一个训练 step 采样 8 个 replay state；每个 state 由 PI0.5 生成 4 个候选动作，因此
每轮共有 32 个候选 flow trajectory。

对第 $i$ 个状态、第 $j$ 个候选和第 $m$ 个 Critic：

$$
A_{ijm}=Q_m(s_i,a_{ij})-V_m(s_i).
$$

使用 two-sided conservative advantage：

$$
A_{ij}^{CA}=
\begin{cases}
\min_m A_{ijm}, & \text{三个 head 都认为更好},\\
\max_m A_{ijm}, & \text{三个 head 都认为更差},\\
0, & \text{三个 head 对方向有分歧}.
\end{cases}
$$

这使 Critic 不确定的候选不会推动 Actor，同时保留正 advantage 和负 advantage。

### 5.2 Flash-OGPO / TGR

完整 OGPO 会对整条去噪链计算 likelihood ratio。当前为了降低 PI0.5 全量微调成本，
每个 replay state 只抽一个去噪步骤 $k$：

$$
r_{ij,k}(\theta)=
\frac{\pi_\theta(x_{k-1}^{ij}\mid s_i,x_k^{ij})}
     {\pi_{old}(x_{k-1}^{ij}\mid s_i,x_k^{ij})}.
$$

Flash surrogate 为：

$$
\mathcal L_{flash}=-\mathbb E\left[
w_k^{TGR}\min\left(
r_{ij,k}A_{ij}^{CA},
\operatorname{clip}(r_{ij,k},1-\epsilon,1+\epsilon)A_{ij}^{CA}
\right)\right].
$$

$w_k^{TGR}$ 用于校正不同去噪时刻的梯度贡献。batch 中 8 个状态按 stratified
sampling 覆盖全部 8 个 flow steps。

需要如实说明：当前 `actor_epochs_per_rollout=1`，并且每轮同步 old policy，因此更新前
ratio 通常为 1、clip fraction 为 0。当前实现更准确地说是 **Flash-OGPO/GRPO 风格的
一次策略梯度，加 KL 和成功轨迹 BC**，不是多 epoch PPO。

### 5.3 当前内层与 Flash-GRPO 的关系

当前内层不是 Flash-GRPO 原样复现，而是 **Flash-GRPO 的低成本单步反向方法 +
OGPO-DIVL 的 Critic 驱动方法**。

与 Flash-GRPO 相同的部分：

- 完整生成最终结果，但只保存并反向一个随机去噪转移；
- 同一状态的 4 个候选共享同一个去噪时间步，避免组内比较被时间步难度混淆；
- 使用 TGR 校正不同去噪时间步的梯度尺度；
- 使用旧策略转移概率构造 ratio，并保留 clipped surrogate 形式；
- 每批数据只训练一轮。Flash-GRPO 官方配置同样使用 `num_inner_epochs=1`。

与 Flash-GRPO 不同的部分：

- Flash-GRPO 用视频奖励模型评分，再按同 prompt 的奖励均值和标准差构造 group-relative
  advantage；本工作用三组 `Q_m-V_m` 和 two-sided 同号规则，不做组均值 baseline，
  也不做组内标准差归一化；
- Flash-GRPO 在线生成视频并即时计算奖励；本工作从固定的 100-episode replay 中抽状态，
  在线生成候选 action chunk，再由冻结 TD10 Critic 评分，不执行新的环境 rollout；
- 本工作额外使用固定 PI0.5 reference KL、KL 超限回滚和 success-buffer BC，这些属于
  OGPO/OGPO+ 的稳定化设计，不是 Flash-GRPO 的核心机制；
- Flash-GRPO 官方训练默认使用 LoRA，公开的视频实验配置中 KL loss 和 SFT 均关闭；
  本工作全量更新 PI0.5 JAX 参数，并实际启用 reference KL 与 success-buffer BC；
- Flash-GRPO 视频实验的 PPO clip range 为 `0.001`，本工作在关闭 `ustate` 自适应后固定为
  `0.2`。不过双方都只训练一个 inner epoch，采样后第一次计算的 ratio 接近 1，因此当前
  两边的 clipping 通常都不是主要更新约束；
- Flash-GRPO 在视频 flow 的推导和离散时间表上实现 TGR；本工作根据 PI0.5 的
  OGPO-corrected SDE 方差重新推导权重，再做 batch-mean normalization 和 `[0.25,4.0]`
  截断，因此是面向 PI0.5 action flow 的适配，不是逐行照搬；
- 当前 batch 的 8 个状态用 `stratified_uniform` 恰好覆盖 8 个 flow steps；官方实现是按
  prompt 随机选同一时间步，并不要求每个 batch 覆盖所有时间步。

### 5.4 KL 与成功轨迹 BC

Actor 总梯度对应：

$$
\mathcal L_{actor}=\mathcal L_{flash}
+\beta_{KL}\mathcal L_{KL}
+\lambda_{succ}\mathcal L_{BC}(\mathcal D_{succ}).
$$

success-buffer BC 现在每个 Actor step 都启用，与 OGPO+ 的更新频率一致。实现上 PPO、
KL 和 BC 分开计算梯度以控制峰值显存，梯度相加后只执行一次 Adafactor 参数更新。

当前日志中 post-update KL 通常约 0.08--0.13，硬阈值为 0.15。这个 KL 是单个抽样
去噪转移、约 660 个 action-chunk 维度上的求和，不等于最终动作已经与 base 相差很大。
例如总 KL 为 0.10 时，平均每维约为 $1.5\times10^{-4}$。

## 6. Actor 工程进度

已完成：

1. PI0.5 JAX 全量参数 `value_and_grad` 与 Adafactor 更新；
2. Flash 单去噪步骤反向和 TGR；
3. 三组 Q-V two-sided advantage；
4. KL 超限原子回滚；
5. success-buffer BC；
6. `.pt` 训练元数据与 `.pt.jax` Orbax 全量参数保存/恢复；
7. 四卡并行处理 32 个候选；
8. old/reference 固定统计只在 GPU 0 微批计算，避免两套完整 PI0.5 广播到四卡；
9. 每两步清理 JAX executable cache，避免第三步连续分配因碎片化 OOM。

现有正式 checkpoint：

| Checkpoint | 状态 |
|---|---|
| step 50 | 完整 `.pt + .pt.jax`，已完成 100-episode 评测 |
| step 100 | 完整 `.pt + .pt.jax`，当前正式续训起点 |
| smoke completed step 105 | `795734` 工程验证产物，不作为正式模型 |

最新 smoke `795734` 已连续完成 step 100--104，所有更新均通过 KL gate，并成功保存
completed step 105。正式任务 `795735` 将从正式 step 100 checkpoint 重新训练到 step 500。

截至文档更新时，`795734` 状态为 `COMPLETED (0:0)`，`795735` 已在
`r8a100-c01` 正式运行。

## 7. Actor 评测结果

| 模型 | Seed | Episodes | 成功率 |
|---|---:|---:|---:|
| PI0.5 base（历史） | 27 | 100 | 68% |
| PI0.5 base（最新复测 `795687`） | 27 | 100 | 65% |
| 早期 TD10 uniform-timestep smoke step 10 | 27 | 100 | 66% |
| 早期 Actor 旧实现 step 100 | 27 | 100 | 65% |
| 当前 TGR step 50 | 27 | 100 | 61% |

这里的“早期 step 10”特指配置
`configs/ogpo/pi05_jax_flash_ogpo_td10_balanced_b8_4gpu_smoke10_gc.yaml` 的产物。它和
当前路线使用同一个 TD10 12k Critic、相同的 8 state x 4 candidate、单去噪步反向、
analytic TGR 和 three-head two-sided advantage。差别主要是：早期版本对 8 个状态分别
独立均匀抽时间步，当前版本每批分层覆盖全部 8 步；成功轨迹 BC 从每 4 步一次改为每步
一次；old/current 固定概率改为同一 JAX trace 内计算，消除了 bf16 累加造成的伪 ratio
偏移；当前路线还加入了更稳定的显存清理，并以 500 step 正式训练为目标。因此 66% 与
61% 不是只改变一个变量的严格消融结果。

当前结论：

- Actor 没有崩塌，仍保持约 60%--65% 成功率；
- 目前没有证据证明 Actor 优于 base；
- 同一 base 的两次结果为 68% 和 65%，说明 100 episode 仍有数个百分点波动；
- KL 非零只说明局部去噪转移发生变化，不代表最终动作或成功率一定显著变化；
- 下一步需要做同状态、同 flow noise 的 base/Actor 最终 action chunk 配对比较，并完成
  step 100/200/500 的多 seed 评测。

## 8. 当前进度判断

| 部分 | 进度 | 判断 |
|---|---:|---|
| 数据采集与 replay | 100% | 已完成 |
| Critic 训练与选择 | 100% | TD10 12k 已固定使用 |
| Actor 算法链路 | 90% | 梯度、KL、BC、保存恢复均已接通 |
| Actor 500-step 正式训练 | 20% | 当前正式 checkpoint 为 step 100/500 |
| Actor 效果验证 | 30% | 已有 base/step50 评测，尚无提升证据 |

## 9. 关键文件

Critic：

```text
outputs/ogpo/click_mouse_gemma_udivl_100ep_td10_balanced_30k_b32_best_calibrated.pt
```

Actor step 50：

```text
outputs/ogpo/checkpoints/click_mouse_pi05_jax_flash_td10_tgr_b8_4gpu_500/step_0050/click_mouse_pi05_jax_flash_td10_tgr_b8_4gpu_500_final.pt
```

Actor step 100：

```text
outputs/ogpo/checkpoints/click_mouse_pi05_jax_flash_td10_tgr_b8_4gpu_500/step_0100/click_mouse_pi05_jax_flash_td10_tgr_b8_4gpu_500_final.pt
```

详细算法文档：

```text
docs/ogpo_paper_derivation_zh.md
docs/ogpo_implementation_zh.md
docs/ogpo_origin_vs_divl_zh.md
0729_meeting_report_zh.md
```

## 10. 可视化图片

TD10 12k 成功/失败轨迹 Q-V 曲线：

```text
outputs/ogpo/visualizations/click_mouse_divl_td10_balanced_step12000/trajectory_q_v_scores.png
```

TD10 12k 成功/失败轨迹关键帧：

```text
outputs/ogpo/visualizations/click_mouse_divl_td10_balanced_step12000/trajectory_keyframes.png
```

TD10 12k Q 峰值对应关键帧：

```text
outputs/ogpo/visualizations/click_mouse_divl_td10_balanced_step12000/trajectory_peak_keyframes.png
```

旧 15k Critic 对照 Q-V 曲线：

```text
outputs/ogpo/visualizations/click_mouse_critic_15k/trajectory_q_v_scores.png
```

旧 15k Critic 对照关键帧：

```text
outputs/ogpo/visualizations/click_mouse_critic_15k/trajectory_keyframes.png
```
