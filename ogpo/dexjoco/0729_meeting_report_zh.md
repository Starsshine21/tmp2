# PI0.5 + U-DIVL + OGPO 组会汇报

> 更新时间：2026-07-29  
> 当前任务：让机械臂把鼠标移动到紫色鼠标垫，并点击鼠标左键。  
> 指令：`Move the mouse to the purple mouse pad and click the left mouse button.`

## 1. 一句话概括

本工作先训练一个能够判断“当前状态和候选动作好不好”的 Critic，再让 Critic
指导 PI0.5 策略更新。Critic 借鉴 LWD 的分布式价值学习方法，Actor 借鉴
OGPO 的 PPO 更新方法；最终目标是在不破坏 PI0.5 原有能力的前提下，提高成功动作
的概率并压低失败动作的概率。

当前已经完成：

1. 采集并整理 100 条 `click_mouse` 轨迹；
2. 完成共享 Gemma 3 + SigLIP 主干的三组 Q-V Critic；
3. 完成 Critic 的 15,000 步训练、校准和验证；
4. 完成 PI0.5 JAX 全参数训练链路，包括 PPO、KL 约束、成功轨迹模仿、
   显存优化、保存与恢复；
5. 正在进行 PI0.5 Actor 的 100 步正式训练。

## 2. 数据

| 项目 | 数量 |
|---|---:|
| 轨迹总数 | 100 |
| 成功轨迹 | 59 |
| 失败轨迹 | 41 |
| 状态转移样本 | 17,478 |

训练集和验证集固定划分。当前是小数据离线实验：Critic 和 Actor 都使用这批已有
数据，没有在训练过程中继续与环境交互。

## 3. 整体流程

```text
100 条真实轨迹
        |
        v
构造 (图像、语言、机器人状态、动作、奖励、下一状态)
        |
        v
训练三组 Q-V Critic
        |
        +--> Q：判断某个候选动作的长期收益
        |
        +--> V：判断当前状态下“合理动作”应达到的价值门槛
        |
        v
对同一状态由 PI0.5 生成 4 个候选动作
        |
        v
三个 Q-V 对共同判断候选动作是更好、更差，还是不确定
        |
        v
使用 PPO 更新 PI0.5 全部 JAX 参数
        |
        +--> KL：限制策略不要偏离原始 PI0.5 太远
        |
        +--> 成功轨迹模仿：防止忘记已经成功的行为
```

## 4. Critic：如何给动作评分

### 4.1 网络结构

Critic 的输入包括：

- 基座相机图像；
- 腕部相机图像；
- 英文任务指令；
- 机器人自身状态；
- 候选动作序列。

图像先经过 SigLIP，文字和图像信息再由 Gemma 3 融合，得到状态表示：

$$
h_s = \operatorname{Gemma3}
\left(
\operatorname{SigLIP}(I_{\mathrm{base}}),
\operatorname{SigLIP}(I_{\mathrm{wrist}}),
\ell,
p
\right).
$$

其中，$I$ 是图像，$\ell$ 是语言指令，$p$ 是机器人状态，$h_s$ 是融合后的状态表示。

动作序列通过一个时间编码器得到动作表示 $h_a$。只有实际执行的动作前缀会进入
Critic，尚未执行的动作不会影响评分。

三个 Q-V 对共享同一个 Gemma 3 + SigLIP 主干，仅最后的输出头不同：

$$
Q_m(s,a)=f_m^Q(h_s,h_a), \qquad m=1,2,3,
$$

$$
Z_m(s)=f_m^V(h_s), \qquad m=1,2,3.
$$

这样做比训练三个完整大模型节省很多显存，同时三个独立输出头仍能表达预测分歧。

### 4.2 Q 和 V 分别表示什么

$Q_m(s,a)$ 表示：在状态 $s$ 执行动作 $a$ 后，预计能得到多大长期收益。

V 不直接输出一个数，而是在 201 个位置上输出概率分布：

$$
\zeta_c \in [-0.1,1.1], \qquad c=1,\ldots,201,
$$

$$
p_{m,c}(s)
=
\frac{\exp(z_{m,c})}
{\sum_{c'=1}^{201}\exp(z_{m,c'})}.
$$

从这个分布中选取一个分位数作为状态价值：

$$
V_m(s)=F^{-1}_{Z_m(s)}\left(\alpha(s)\right).
$$

直观地说，V 不是只猜一个平均分，而是先给出“可能得到哪些分数以及各自概率”，
再根据当前预测是否确定，选择一个相对保守的价值门槛。

### 4.3 Critic 的训练目标

下一状态的价值采用三个 V 分位数的平均值：

$$
V_{\mathrm{next}}(s')
=
\frac{1}{3}\sum_{m=1}^{3}V_m^{-}(s').
$$

这里上标 $-$ 表示缓慢更新的目标网络。一步 TD 目标为：

$$
y
=
r+\gamma(1-d)V_{\mathrm{next}}(s'),
\qquad \gamma=0.9999.
$$

$r$ 是当前奖励，$d$ 表示任务是否结束。Q 使用平方误差训练：

$$
\mathcal L_Q
=
\frac{1}{3B}
\sum_{m=1}^{3}\sum_{i=1}^{B}
\left(Q_m(s_i,a_i)-y_i\right)^2.
$$

同时，把目标 Q 投影到 201 个价值位置上，训练 V 的完整分布：

$$
\mathcal L_V
=
-\frac{1}{3B}
\sum_{m=1}^{3}\sum_{i=1}^{B}\sum_{c=1}^{201}
\Phi\left(Q_m^{-}(s_i,a_i)\right)_c
\log p_{m,c}(s_i).
$$

最终 Critic 损失为：

$$
\mathcal L_{\mathrm{Critic}}
=
\mathcal L_Q+\mathcal L_V.
$$

当前 Critic 对 Gemma 3、SigLIP 和三个 Q-V 输出头进行全参数训练；batch size
为 8，共训练 15,000 步。

## 5. 三个 Critic 如何保守地指导 Actor

对同一个状态，PI0.5 并行生成 4 个候选动作。每个 Q-V 对分别计算：

$$
\Delta_{m,j}
=
Q_m(s,a_j)-V_m(s).
$$

其中 $m$ 表示第几个 Q-V 对，$j$ 表示第几个候选动作。

只有三个 Q-V 对判断方向完全相同时，才更新 Actor。

三个判断都为正时：

$$
\Delta_{1,j}>0,\quad \Delta_{2,j}>0,\quad \Delta_{3,j}>0,
$$

$$
A_j=\min_m\Delta_{m,j}.
$$

三个判断都为负时：

$$
\Delta_{1,j}<0,\quad \Delta_{2,j}<0,\quad \Delta_{3,j}<0,
$$

$$
A_j=\max_m\Delta_{m,j}.
$$

三个判断方向存在分歧时：

$$
A_j=0.
$$

含义是：

- 三个判断都认为动作更好：提高该动作的概率，但采用最保守的正分数；
- 三个判断都认为动作更差：降低该动作的概率，但采用最保守的负分数；
- 三个判断意见不一致：本轮不使用这个候选动作更新 Actor。

这就是当前保留的“双侧保守判断”。它既防止错误强化坏动作，也防止错误删除好动作。

这里没有使用候选组平均值作为基线，也没有用组内标准差强行缩放。基线来自
V 的价值分布，因此可以保留不同状态之间真实的价值差异。

## 6. Actor：如何训练 PI0.5

### 6.1 从确定性生成变成可计算概率的生成

PI0.5 原本逐步把噪声动作变成真实动作。为了使用 PPO，需要让每个生成步骤都有
可计算的概率，因此在 flow 过程中加入很小的噪声：

$$
x_{k-1}
=
\mu_\theta(s,x_k,t_k)+\sigma(t_k)\epsilon_k,
\qquad
\epsilon_k \sim \mathcal N(0,I).
$$

噪声会随着动作逐渐生成而减小：

$$
\sigma(t)=\sigma_0\sqrt{t}.
$$

这样既能计算“新策略生成该动作的概率”，又尽量保持原始 PI0.5 的动作分布。
当前每条动作生成链使用 8 个 flow step。

### 6.2 PPO 更新

当前采用 Flash 版本：从 8 个生成步骤中选择一个有效步骤反向传播，避免每次更新
都对完整 PI0.5 生成链做反向传播。

新旧策略在该步骤上的概率比为：

$$
\rho(\theta)
=
\frac{
\pi_\theta(x_{k-1}\mid s,x_k,t_k)
}{
\pi_{\mathrm{old}}(x_{k-1}\mid s,x_k,t_k)
}.
$$

PPO 损失为：

$$
\mathcal L_{\mathrm{PPO}}
=
-\mathbb E
\left[
\min
\left(
\rho A,
\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A
\right)
w_{\mathrm{time}}
\right].
$$

$A$ 是上一节的三头保守分数，$w_{\mathrm{time}}$ 用于平衡不同生成时间步。

当 $A>0$ 时，训练会提高好动作的生成概率；当 $A<0$ 时，训练会降低坏动作的
生成概率；当三个 Critic 意见不一致、$A=0$ 时，本轮 PPO 为 0。这种 0 是保守
机制主动跳过，不代表程序没有梯度或训练出错。

### 6.3 防止 Actor 偏离原始 PI0.5

使用固定的原始 PI0.5 作为参考：

$$
\mathcal L_{\mathrm{KL}}
=
D_{\mathrm{KL}}
\left(
\pi_\theta \,\|\, \pi_{\mathrm{ref}}
\right).
$$

更新后如果 KL 超过 0.5，会同时撤销 Actor 参数和 Adafactor 优化器状态，避免一次
过大的更新破坏模型。

### 6.4 保留已经成功的行为

只使用 59 条成功轨迹进行模仿，不模仿 41 条失败轨迹。成功轨迹损失为：

$$
x_t=t\epsilon+(1-t)a,
$$

$$
v^\star=\epsilon-a,
$$

$$
\mathcal L_{\mathrm{success}}
=
\mathbb E
\left[
\left\|
v_\theta(s,x_t,t)-v^\star
\right\|_2^2
\right].
$$

当前总目标可以写为：

$$
\mathcal L_{\mathrm{Actor}}
=
\mathcal L_{\mathrm{PPO}}
+0.01\mathcal L_{\mathrm{KL}}
+0.02\mathcal L_{\mathrm{success}}.
$$

为了节省时间，成功轨迹模仿每 4 个 Actor step 执行一次，而不是每一步都执行。
普通全数据模仿已经关闭，因为全数据中包含 41 条失败轨迹。

### 6.5 U-state 当前做什么

U-state 表示当前状态本身的不确定程度。三个 V head 都输出离散价值分布，先计算
每个分布的归一化熵，再取三个 head 的平均：

$$
u_{\mathrm{state}}(s)
=
\frac{1}{3}
\sum_{m=1}^{3}
H_m(s).
$$

当前状态权重为：

$$
w_{\mathrm{state}}(s)
=
\exp\left(-0.5u_{\mathrm{state}}(s)\right).
$$

状态越确定，$w_{\mathrm{state}}$ 越接近 1；状态越不确定，该状态产生的 Actor
梯度越小。它仍然会缩放最终 advantage，但当前配置为：

```yaml
adapt_ppo_clip: false
adapt_kl_beta: false
```

因此 U-state **不会**改变 PPO 的截断范围，也**不会**改变 KL 系数。这样可以把
“样本权重”和“更新力度”分开，便于解释和做消融实验。

### 6.6 $r_n$：校准 Critic 的不确定程度

$r_n$ 不是直接乘在 PPO loss 上的权重。它只在 Critic 校准阶段使用，用于判断
“三个 Q 的分歧是否真实反映预测误差”。

对第 $n$ 个校准样本，先计算三个 Q 的均值和标准差：

$$
\mu_{Q,n}
=
\frac{1}{3}
\sum_{m=1}^{3}Q_m(s_n,a_n),
$$

$$
\sigma_{Q,n}
=
\operatorname{Std}_{m=1,2,3}
\left[Q_m(s_n,a_n)\right].
$$

然后计算非一致性分数：

$$
r_n
=
\frac{
\left|\mu_{Q,n}-G_n\right|
}{
\max\left(\sigma_{Q,n},\varepsilon\right)
},
$$

其中 $G_n$ 是数据中的真实回报。取 $r_n$ 的 90% 分位数得到校准系数：

$$
c_{0.1}
=
\operatorname{Quantile}_{0.9}
\left(\{r_n\}_{n=1}^{N_{\mathrm{cal}}}\right).
$$

当前 15k Critic 得到：

$$
c_{0.1}=204.568832.
$$

Actor 训练时使用校准后的 Q 分歧：

$$
\widetilde{\sigma}_Q
=
c_{0.1}\sigma_Q.
$$

校准系数较大，说明原始的三个 Q 虽然数值接近，但这种分歧低估了实际预测误差。
因此后续不能直接把原始 Q 标准差当成可靠的不确定程度。

### 6.7 $w_{\mathrm{support}}$：候选动作是否得到数据支持

除了 Q 分歧，还计算候选动作与 replay buffer 中对应动作的距离。对长度为 $H$、
每步维度为 $D$ 的完整动作块：

$$
d_{\mathrm{support}}
=
\sqrt{
\frac{1}{HD}
\sum_{h=1}^{H}
\sum_{d=1}^{D}
\left(
a_{h,d}^{\mathrm{candidate}}
-
a_{h,d}^{\mathrm{replay}}
\right)^2
}.
$$

当前 support 权重为：

$$
w_{\mathrm{support}}
=
\exp
\left(
-0.5\widetilde{\sigma}_Q
-0.5d_{\mathrm{support}}
\right).
$$

如果 $d_{\mathrm{support}}>10$，则直接令：

$$
w_{\mathrm{support}}=0.
$$

当前配置中 `use_support_weight: true`，所以这个权重正在生效。它的含义是：三个 Q
分歧越大，或者候选动作离已有数据越远，Actor 越不应该相信这次 Critic 判断。

### 6.8 $w_{\mathrm{rect}}$：flow 时间校正权重

$w_{\mathrm{rect}}$ 原本用于修正不同 flow 时间步对 PPO 更新的贡献。当前配置使用
`temporal_rectification_mode: analytic`，而当前 analytic 实现返回全 1：

$$
w_{\mathrm{rect}}=1.
$$

因此它保留在训练接口和日志中，但当前不会放大或缩小任何样本。第 6.2 节公式里的
$w_{\mathrm{time}}$，在当前实现中就是这里的 $w_{\mathrm{rect}}$。

### 6.9 这些变量如何组合

三个 Q-V 对先产生双侧保守 advantage，随后使用运行中的绝对偏差尺度进行归一化：

$$
\widehat A
=
\operatorname{clip}
\left(
\frac{A}{\operatorname{EMA\text{-}MAD}(A)+\varepsilon},
-5,\;5
\right).
$$

最终送入 PPO 的 advantage 为：

$$
A_{\mathrm{final}}
=
\widehat A
\cdot w_{\mathrm{state}}
\cdot w_{\mathrm{support}}.
$$

PPO 样本损失再乘以：

$$
w_{\mathrm{rect}}=1.
$$

因此当前真实的数据流是：

```text
三个 Q-V 对做双侧保守判断
        |
        v
运行 MAD 归一化并截断到 [-5, 5]
        |
        +--> w_state：状态不确定时减小 advantage
        |
        +--> r_n：只生成校准系数，不直接进入 PPO
        |
        +--> w_support：Q 分歧大或动作离数据远时减小 advantage
        |
        +--> w_rect：当前恒为 1
        |
        v
使用 A_final 更新 PI0.5
```

这里最重要的边界是：

- `w_state` 正在影响 advantage；
- `w_state` 不影响 PPO clip，也不影响 KL 系数；
- `r_n` 只在校准阶段计算，Actor 使用的是它校准后的 Q 分歧；
- `w_support` 正在影响 advantage；
- `w_rect` 当前恒为 1，不产生实际缩放。

## 7. Critic 实验结果

### 7.1 15k 与 30k 对比

| 指标 | 15k Critic | 30k Critic | 解释 |
|---|---:|---:|---|
| 两两排序正确率 | **0.630783** | 0.600051 | 判断两个动作谁更好的正确比例 |
| Q 的均方根误差 | 0.283842 | **0.281914** | Q 预测与真实回报的数值误差，越低越好 |
| 区间覆盖率 | 0.899867 | 0.899867 | 约 90% 的真实回报落在校准区间内 |
| 排序通过门槛 | 0.55 | 0.55 | 两个模型均通过 |

结论：

- 继续训练到 30k 后，Q 的数值误差只改善了约 0.002；
- 但 Actor 更关心的动作排序正确率从 63.1% 降到了 60.0%；
- 这说明在固定的 100 条轨迹上继续训练已经出现过拟合迹象；
- 因此正式 Actor 使用 **15k 校准后的 Critic**，而不是 30k Critic。

### 7.2 三个指标怎样理解

**两两排序正确率 0.630783**

从验证集中任取两个样本，比较 Critic 预测的高低顺序与真实回报顺序是否一致。
0.630783 表示约 63.1% 的样本对排序正确。它比随机排序的 50% 更好，并超过当前
设置的 55% 准入门槛。

**Q 的均方根误差 0.283842**

$$
\operatorname{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left(\bar Q_i-G_i\right)^2
}.
$$

$\bar Q_i$ 是三个 Q 的平均预测，$G_i$ 是数据中的真实回报。该指标衡量数值预测
误差，但不能单独代表动作排序能力。

**区间覆盖率 0.899867**

用三个 Q 的分歧构造预测区间并进行校准：

$$
\left[
\bar Q_i-c\sigma_i,\;
\bar Q_i+c\sigma_i
\right].
$$

0.899867 表示约 90.0% 的真实回报落在这个区间内，与设定的 90% 校准目标一致。

## 8. Actor 当前进展

截至本文档生成时：

| 项目 | 当前值 |
|---|---:|
| Slurm Job | `787418` |
| 目标训练步数 | 100 |
| 已记录步数 | 14（step 0 至 step 13） |
| 候选动作数 | 4 |
| flow step 数 | 8 |
| Actor 学习率 | $1\times10^{-6}$ |
| 梯度小批量 | 2 |
| 成功轨迹更新周期 | 每 4 步一次 |
| 已拒绝更新数 | 0 |
| 峰值显存 | 约 30.3 GB |
| 速度 | 约 5 至 6 分钟一步 |

目前训练链路正常，没有出现显存溢出，也没有触发 KL 回滚。部分 step 的 PPO
损失为 0，主要原因是三个 Q-V 对对候选动作的判断方向存在分歧，保守机制主动
跳过这些样本。

需要强调：以上只是训练过程指标，**还不是最终任务成功率**。Actor 完成 100 步后，
还需要在环境中分别评估原始 PI0.5 和训练后 PI0.5，才能判断策略是否真正改善。

## 9. 与 OGPO 和 LWD 的关系

| 部分 | 来源 | 当前实现 |
|---|---|---|
| Critic 的价值分布、分位数和 TD 目标 | LWD DIVL | 保留主要思路 |
| Gemma 3 + SigLIP 大 Critic | LWD 风格 | 扩展为共享主干的三个 Q-V 对 |
| 多候选动作和 PPO | OGPO | 保留 |
| 双侧保守判断 | OGPO+CA | 保留 |
| 成功轨迹模仿 | OGPO+ | 保留 |
| flow 加噪并计算概率 | OGPO | 保留 |
| Flash 单步反向传播 | 本项目 | 用较低计算量近似完整生成链更新 |
| U-state 只调整状态权重 | 本项目当前选择 | 默认不改变 PPO 截断和 KL 系数 |

当前方法不是对某一篇工作的原样复现。更准确的描述是：

> 使用 LWD DIVL 风格的三头大模型 Critic，为 OGPO 风格的 PI0.5 PPO 全参数微调
> 提供保守价值信号，并通过成功轨迹模仿和 KL 回滚提高小数据训练的稳定性。

## 10. 组会口头汇报提纲

可以按下面顺序介绍：

1. **问题**：PI0.5 已经会生成动作，但仅靠模仿学习无法主动区分成功和失败动作。
2. **数据**：先在 `click_mouse` 上采集 100 条轨迹，其中 59 条成功、41 条失败。
3. **Critic**：用 Gemma 3 + SigLIP 读取图像、语言和机器人状态，三个 Q-V 对共享
   主干，用 Q 判断动作，用 V 给出当前状态的价值门槛。
4. **保守判断**：同一状态生成 4 个动作，只有三个 Q-V 对意见一致时才更新 Actor。
5. **Actor**：使用 PPO 全参数更新 JAX PI0.5，提高好动作概率，降低坏动作概率。
6. **稳定性**：KL 限制模型不要偏离原策略；只模仿成功轨迹；过大的更新自动撤销。
7. **Critic 结果**：15k 排序正确率 63.1%，优于 30k 的 60.0%，因此选择 15k，
   说明小数据继续训练会过拟合。
8. **当前进展**：正式 Actor 正在进行 100 步训练，链路和显存已经稳定。
9. **下一步**：完成 Actor 训练后做环境成功率评估，并与原始 PI0.5、仅模仿学习和
   去掉保守判断的版本进行对比。

## 11. 当前结论与下一步

当前能够确认的是：

- Critic 已经学到高于随机水平的动作排序能力；
- 15k 比 30k 更适合指导 Actor，继续训练 Critic 并不一定更好；
- PI0.5 JAX 全参数 PPO 更新链路已经打通并能够稳定跨步运行；
- 三头保守判断、KL 回滚和成功轨迹模仿都已实际接入训练。

下一步实验：

1. 完成 Actor 的 100 步训练；
2. 保存最终 JAX PI0.5 参数和 Adafactor 状态；
3. 在 `click_mouse` 环境中评估训练前后的成功率；
4. 至少运行多个随机种子，报告平均值和波动范围；
5. 做关键对比：去掉成功轨迹模仿、去掉三头保守判断、去掉 DIVL 分布价值；
6. 后续加入新在线数据，验证“训练 Critic、更新 Actor、重新采集”的闭环。
