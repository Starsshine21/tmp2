# PI0.5 上的 OGPO-Origin 与 DIVL-OGPO 对照

> 最后核对：2026-08-10。本文以当前仓库代码、解析后的正式配置和已经生成的
> checkpoint 为准。历史实验可能使用不同超参数，不能只凭任务名判断算法。

## 1. 两条路线的定位

本仓库保留两条相互隔离的实验路线。

### 1.1 OGPO-Origin

- 目标是尽量直接迁移 vanilla OGPO 的基本结构。
- 参考 [OGPO v1](https://arxiv.org/abs/2605.03065v1) 和
  [官方仓库](https://github.com/simchowitzlabpublic/OGPO_public)。
- Critic 是共享 Gemma3+SigLIP 的 10 个标量 Q 头，当前变体使用纯
  Monte-Carlo return。
- Actor 对同一状态生成 32 条完整 flow 链，使用组均值 baseline 和整链
  likelihood ratio。
- 不使用 V、DIVL、three-QV two-sided advantage、`wstate`、`wsupport` 和
  Flash 时间修正。

### 1.2 DIVL-OGPO

- 是当前主要改进路线。
- Critic 是共享 Gemma3+SigLIP backbone 的三个 Q-V 对。
- V 是 201 个 value atoms 上的离散分布，并按分布熵选择状态分位数。
- Actor 使用 three-QV two-sided consensus、running MAD、`wstate`、
  `wsupport`、Flash 随机去噪步和解析时间修正。
- PI0.5 Actor 使用 JAX 全量微调，主模型参数为 BF16；概率、KL 和部分优化统计
  使用 FP32。
- 当前 critic validation gate 已关闭；Actor 的 reference-KL 回滚仍保留。

输出目录彼此隔离：

- Origin：`outputs/ogpo-origin/`
- DIVL：`outputs/ogpo/`

## 2. 共同的外层 MDP

PI0.5 一次生成长度为 30 的动作块：

$$
A_t=(a_t^0,a_t^1,\ldots,a_t^{29}).
$$

ClickMouse 当前通常只执行前 4 个动作。设执行掩码为
$m_i\in\{0,1\}$，一个动作块的累计奖励为：

$$
R_t^{(H)}
=
\sum_{i=0}^{H-1}\gamma^i m_i r_{t+i}.
$$

数据还保存跨动作块折扣 $\Gamma_t$ 和终止标记 $d_t$。一般 TD target 可以写成：

$$
y_t
=
R_t^{(H)}
+
\Gamma_t(1-d_t)\widehat V(s_{t+1}).
$$

这里的 $t+1$ 表示下一个动作块，而不是下一个 MuJoCo 控制步。Origin 当前使用
MC return，不使用这个 bootstrap；DIVL Critic 使用它的 10-step 扩展。

两条路线都使用原生 JAX PI0.5 Actor，并把原本确定性的 flow 变成带条件概率的
`ogpo_corrected` 随机过程。

## 3. OGPO-Origin Critic

### 3.1 网络

图像、语言和机器人状态经过共享且全量微调的 Gemma3+SigLIP：

$$
h_s=f_{\mathrm{VLM}}(I_{\mathrm{base}},I_{\mathrm{wrist}},l,p).
$$

动作块经过带执行掩码的时序编码器：

$$
h_A=f_A(A,m).
$$

10 个独立 Q 头读取同一组表示：

$$
Q_i(s,A)=g_i([h_s;h_A]),\qquad i=1,\ldots,10.
$$

当前 `MC-Full` 变体不缓存 VLM readout，梯度会更新 Gemma3、SigLIP、动作编码器
和全部 Q 头。这是面向视觉 VLA 的工程改造，不是 OGPO 官方低维 MLP critic。

### 3.2 纯 MC target

从每条 replay episode 末端向前计算：

$$
G_t=R_t^{(H)}+\Gamma_t(1-d_t)G_{t+1}.
$$

Origin 当前直接令：

$$
y_t=G_t,
$$

并训练：

$$
\mathcal L_Q^{\mathrm{Origin}}
=
\frac{1}{10B}
\sum_{i=1}^{10}\sum_{b=1}^{B}
\left(Q_i(s_b,A_b)-G_b\right)^2.
$$

因此它不生成下一动作 $A'$，也不运行 target PI0.5。target critic 仍做 Polyak
更新：

$$
\bar\phi\leftarrow(1-\tau_Q)\bar\phi+\tau_Q\phi,
\qquad \tau_Q=0.05.
$$

## 4. OGPO-Origin Actor

### 4.1 同状态 32 条完整 flow 链

当前 Origin Actor 的 replay-state batch size 为 1。对状态 $s_i$，old policy 生成
$G=32$ 条链：

$$
\tau_{ij}=(x_K^{ij},x_{K-1}^{ij},\ldots,x_0^{ij}),
\qquad j=1,\ldots,32.
$$

最终动作块 $A_{ij}=x_0^{ij}$ 使用 10Q 均值评分：

$$
q_{ij}=\frac{1}{10}\sum_{m=1}^{10}Q_m(s_i,A_{ij}).
$$

组均值 baseline 和 advantage 为：

$$
b_i=\frac{1}{32}\sum_{j=1}^{32}q_{ij},
\qquad
A_{ij}=q_{ij}-b_i.
$$

这里只减均值，不除组内标准差。

### 4.2 整链 ratio

每个随机 flow 转移是：

$$
p_\theta(x_{k-1}\mid s,x_k)
=
\mathcal N(x_{k-1};\mu_{\theta,k},\Sigma_k).
$$

整条链的 ratio 为：

$$
\rho_{ij}(\theta)
=
\exp\left[
\sum_{k=1}^{K}
\left(
\log p_\theta(x_{k-1}^{ij}\mid s_i,x_k^{ij})
-
\log p_{\mathrm{old}}(x_{k-1}^{ij}\mid s_i,x_k^{ij})
\right)
\right].
$$

Origin 使用 $\epsilon=0.01$ 的 PPO clipped objective：

$$
\mathcal L_{\mathrm{OriginActor}}
=
-\frac{1}{BG}\sum_{i,j}
\min\left(
\rho_{ij}A_{ij},
\operatorname{clip}(\rho_{ij},1-\epsilon,1+\epsilon)A_{ij}
\right).
$$

32 条链按 microbatch 累积梯度，最后只执行一次 Adafactor step，不是 32 次
优化器更新。

## 5. DIVL-OGPO Critic

### 5.1 共享 VLM 和三个 Q-V 对

当前正式 Critic 使用一个全量微调的 Gemma3+SigLIP backbone：

$$
h_s=f_\phi(I_{\mathrm{base}},I_{\mathrm{wrist}},l,p).
$$

在共享 $h_s$ 上连接三个 Q 头和三个 V 分布头：

$$
Q_m(s,A)=g_m^Q(h_s,h_A),
\qquad
Z_m(s)=\operatorname{softmax}(g_m^V(h_s)),
\qquad m=1,2,3.
$$

三个 $Z_m$ 都定义在固定 support 上：

$$
z_c=-0.1+c\frac{1.2}{200},
\qquad c=0,\ldots,200.
$$

### 5.2 熵控制的 V 分位数

设第 $m$ 个 V 分布的归一化熵为：

$$
H_m(s)
=
-\frac{1}{\log 201}
\sum_{c=0}^{200}p_{m,c}(s)\log p_{m,c}(s).
$$

当前采用 LWD 风格的线性分位数：

$$
\alpha_m(s)
=
\operatorname{clip}
\left(0.6-0.3H_m(s),\ 0.5,\ 0.6\right).
$$

然后选取离散 CDF 第一次达到 $\alpha_m$ 的 atom，不做 atom 间插值：

$$
V_m(s)=F^{-1}_{Z_m(s)}(\alpha_m(s)).
$$

### 5.3 离线 TD10 target

原始 replay 保存相邻动作块。正式 Critic 训练脚本在加载后把 10 个相邻 outer
transition 折叠成一个 TD10 样本：

$$
R_t^{(10)}
=
\sum_{j=0}^{9}
\left(\prod_{u=0}^{j-1}\Gamma_{t+u}\right)
R_{t+j}^{(H)},
$$

$$
\Gamma_t^{(10)}=\prod_{j=0}^{9}\Gamma_{t+j}.
$$

target V 使用三个 target V 分位数的均值：

$$
\bar V^-(s_{t+10})
=
\frac{1}{3}\sum_{m=1}^{3}V_m^-(s_{t+10}).
$$

纯 TD target 为：

$$
y_t^{\mathrm{TD10}}
=
R_t^{(10)}
+
\Gamma_t^{(10)}(1-d_t)\bar V^-(s_{t+10}).
$$

当前 `lambda_mc=0`，因此没有混合 MC return；`bootstrap_target=ensemble_mean`，
没有使用 $V_{\min}$。

Q 使用 bootstrap mask $M_{m,b}\sim\mathrm{Bernoulli}(0.8)$ 和 MSE：

$$
\mathcal L_Q
=
\frac{
\sum_{m,b}M_{m,b}
\left(Q_m(s_b,A_b)-y_b^{\mathrm{TD10}}\right)^2
}{
\sum_{m,b}M_{m,b}
}.
$$

V 分布学习 replay action 在 target critic 下的 Q 值。先投影到 atoms：

$$
\widetilde Z_{m,b}
=
\Pi_{\{z_c\}}
\left(Q_m^-(s_b,A_b^{\mathrm{data}})\right),
$$

再做交叉熵：

$$
\mathcal L_V
=
-\frac{1}{3B}
\sum_{m,b,c}
\widetilde Z_{m,b,c}\log Z_{m,b,c}(s_b).
$$

总损失为：

$$
\mathcal L_{\mathrm{critic}}=\mathcal L_Q+\mathcal L_V.
$$

正式离线训练使用 Adam、Q/V head 学习率 $5\times10^{-4}$、backbone 学习率
$2\times10^{-5}$、cosine decay、target EMA $\tau_Q=0.005$、有效 batch 32、
microbatch 8。训练时 Gemma3 和 SigLIP 都参与更新。

## 6. 当前 DIVL-OGPO Actor

### 6.1 每个 outer step 的采样量

当前配置一次采样 $B=8$ 个 replay 状态。每个状态生成 $G=4$ 个候选动作块，
因此一次 outer step 共评分 32 个候选。

当前 flow 有 $K=8$ 个去噪步。8 个状态按 `stratified_uniform` 覆盖全部 8 个
时间步；同一状态的 4 个候选共享选中的去噪步 $k_i$，但有各自的 flow 噪声。

Flash 路线仍运行到最终动作 $A_{ij}=x_0^{ij}$ 供 Critic 评分，但 Actor 反向只使用
随机选中的转移：

$$
(x_{k_i}^{ij},x_{k_i-1}^{ij}).
$$

### 6.2 Three-QV two-sided advantage

每个 Q-V 对先计算：

$$
a_{ijm}=Q_m(s_i,A_{ij})-V_m(s_i).
$$

只有三个 head 对符号一致时才保留：

$$
A_{ij}^{\mathrm{CA}}
=
\begin{cases}
\min_m a_{ijm}, & a_{ijm}>0,\ \forall m,\\
\max_m a_{ijm}, & a_{ijm}<0,\ \forall m,\\
0, & \text{三个 head 对方向有分歧}.
\end{cases}
$$

当前不使用组均值 baseline，也不除每组标准差。非零 conservative advantage
使用跨 batch 的 running MAD 做稳健尺度归一化：

$$
\widehat A_{ij}
=
\operatorname{clip}
\left(
\frac{A_{ij}^{\mathrm{CA}}}{\operatorname{MAD}_{\mathrm{EMA}}+\varepsilon},
-5,5
\right).
$$

### 6.3 `wstate` 和 `wsupport`

三个 V 分布的平均熵构成状态权重：

$$
w_{\mathrm{state}}(s_i)
=
\exp(-0.5\,\bar H(s_i)).
$$

它仍然乘在 advantage 上，但配置明确关闭：

- `uncertainty.adapt_ppo_clip=false`；
- `uncertainty.adapt_kl_beta=false`。

因此 `wstate` 不改变 PPO clip 宽度，也不改变 KL 系数。

候选动作的 support 权重为：

$$
w_{\mathrm{support}}(s_i,A_{ij})
=
\exp\left[
-0.5\,\sigma_Q(s_i,A_{ij})
-0.5\,d(A_{ij},A_i^{\mathrm{data}})
\right],
$$

其中 $\sigma_Q$ 是三个 Q 的分歧，$d$ 是候选动作与 replay action 的 RMS 距离。
若距离超过 10，权重直接置零。最终送入策略梯度的 advantage 为：

$$
A_{ij}^{\mathrm{final}}
=
\widehat A_{ij}
\cdot w_{\mathrm{state}}(s_i)
\cdot w_{\mathrm{support}}(s_i,A_{ij}).
$$

此外，V 熵高于 0.98，或同一状态的 4 个候选中同号共识比例低于 0.05 时，该状态
整组 advantage 置零。

### 6.4 ODE-to-SDE 和 Flash 时间修正

PI0.5 原始 Euler flow 是确定性的。当前使用：

$$
\widetilde v_\theta
=
v_\theta
+
\frac{1}{2}\sigma^2\left[(1-t)v_\theta+x_t\right],
$$

$$
\mu_{\theta,k}=x_k+\Delta t\,\widetilde v_\theta(s,x_k,t_k),
$$

$$
x_{k-1}=\mu_{\theta,k}+\sigma\sqrt{t_k}\epsilon_k,
\qquad \epsilon_k\sim\mathcal N(0,I).
$$

当前 `stochastic_variance=0.01`。为修正只采一个去噪步导致的时间尺度差异，使用
解析 TGR 权重：

$$
r_n(t)
\propto
\frac{\sqrt t}{1+\frac{1}{2}\sigma^2(1-t)},
$$

再做 batch 均值归一化并截断到 $[0.25,4.0]$。

### 6.5 单轮 on-policy surrogate

选中转移的对数概率为：

$$
\ell_\theta
=
\sum_d
\log\mathcal N
\left(x_{k-1,d};\mu_{\theta,d},\sigma_{\theta,d}^2\right).
$$

标准 PPO ratio 为：

$$
\rho_\theta=\exp(\ell_\theta-\ell_{\mathrm{old}}).
$$

当前只做一个 Actor epoch，并且每步同步 old policy。理论上更新前
$\theta=\theta_{\mathrm{old}}$，所以 ratio 必须为 1。为避免两个独立 BF16 trace
产生伪 ratio，当前使用：

$$
\widetilde\ell_\theta
=
\ell_{\mathrm{old}}
+
\ell_\theta
-
\operatorname{stopgrad}(\ell_\theta).
$$

它满足：

$$
\widetilde\ell_\theta=\ell_{\mathrm{old}}
\quad\text{（前向）},
\qquad
\nabla_\theta\widetilde\ell_\theta
=
\nabla_\theta\ell_\theta
\quad\text{（反向）}.
$$

因此前向 ratio 精确为 1，但策略梯度完整保留：

$$
\nabla_\theta\mathcal L_{\mathrm{Flash}}
=
-\mathbb E
\left[
r_n(t)A^{\mathrm{final}}
\nabla_\theta\log p_\theta(x_{k-1}\mid s,x_k)
\right].
$$

当前虽然仍保留 PPO clipped objective 的形式和 $\epsilon=0.2$，但单 epoch 下
clip 正常应为 0。严格说它更接近 Flash-GRPO 风格的单轮 on-policy policy
gradient，而不是复用同一 rollout 做多 epoch 的 PPO。

### 6.6 KL、成功轨迹 BC 和回滚

当前总 Actor 损失为：

$$
\mathcal L_{\mathrm{Actor}}
=
\mathcal L_{\mathrm{Flash}}
+0.01\mathcal L_{\mathrm{KL}}
+0.02\mathcal L_{\mathrm{success\text{-}BC}}.
$$

Flow-matching anchor 当前为 0。KL 比较当前策略和原始 PI0.5 reference 在选中
转移上的对角高斯分布：

$$
D_{\mathrm{KL}}(p_\theta\Vert p_{\mathrm{ref}})
=
\sum_d
\left[
\log\frac{\sigma_{\mathrm{ref},d}}{\sigma_{\theta,d}}
+
\frac{
\sigma_{\theta,d}^2+
(\mu_{\theta,d}-\mu_{\mathrm{ref},d})^2
}{2\sigma_{\mathrm{ref},d}^2}
-\frac{1}{2}
\right].
$$

每次 optimizer proposal 后重新计算实际 KL。若平均 KL 大于 0.15，Actor 参数和
Adafactor state 一起回滚；这项 trust-region 保护没有关闭。

## 7. 旧 PPO 数值实现与当前修复

旧实现让 current policy 和 old policy 在不同 JAX trace、不同 GPU 上各自执行
BF16 前向。虽然两者参数相同，实际计算却是：

$$
\widehat\ell_\theta=\ell_\theta+e_\theta,
\qquad
\widehat\ell_{\mathrm{old}}=\ell_{\mathrm{old}}+e_{\mathrm{old}},
$$

$$
\widehat\rho
=
\exp(e_\theta-e_{\mathrm{old}})\ne1.
$$

动作块包含约 660 个维度，逐维很小的 BF16 均值误差会在 Gaussian log-prob
求和时放大。旧任务实际出现：

- ratio mean 约 1.1 到 1.5；
- ratio std 可超过 2；
- 约 28% 到 41% 的样本被 PPO clip；
- 但这些数值出现在真正 optimizer update 之前。

后果是 PPO clip 不再限制真实策略变化，而是在过滤数值噪声；有效 advantage
梯度被无依据地削弱或改变，更新强度高方差，KL 也可能包含额外数值偏差。当前
共享 frozen-statistics kernel，并使用上一节的局部 surrogate。修复后实际日志为：

- ratio mean 约 1.00000；
- ratio std 不超过约 $6\times10^{-5}$；
- pre-update clip fraction 为 0；
- post-update reference KL 仍非零，说明参数确实在变化。

## 8. 当前实际训练 Pipeline

### 8.1 离线 Critic 阶段

1. 使用固定的 100-episode ClickMouse replay，拆分 train/validation。
2. 加载后把相邻 outer transition 折成 TD10。
3. 全量微调 Gemma3+SigLIP 和 3Q-3V heads，batch 32、microbatch 8。
4. 训练上限 30k，不使用 early stopping；按 validation 排序、相关性、RMSE 和
   exploitation gap 选 checkpoint。
5. 当前 Actor lineage 使用的 Critic 是选中的 TD10 step-12000 calibrated
   checkpoint。

### 8.2 Actor 阶段

每个 outer step：

1. 从固定旧 replay 采样 8 个状态。
2. 每个状态由 old PI0.5 产生 4 个候选，共 32 个候选动作块。
3. 三个 Q-V 对计算 two-sided advantage。
4. 乘 running-MAD、`wstate`、`wsupport` 和解析 TGR 权重。
5. 在每个状态选中的一个去噪转移上做一次 JAX 全模型反向。
6. 加 reference KL 和 success-buffer BC。
7. 若 post-update KL 大于 0.15，原子回滚 Actor 和优化器；否则接受更新。
8. 每 500 个 proposal/log step 保存编号 checkpoint。

当前 lineage 不是从 0 新训练：联合任务从旧 Actor step 7500 恢复；编号 step
8000 表示该轮执行了 500 次 proposal。其间 27 次因 KL 超限回滚，所以实际接受
473 次 Actor 更新，而不是 500 次。

### 8.3 当前周期 Critic refresh 的真实含义

当前联合配置仍保留：

- 每 25 个被接受的 Actor 更新，Critic 更新 1 step；
- 有效 batch 32，microbatch 8；
- 学习率 $10^{-5}$；
- 采样比例为 50% uniform、25% success、12.5% terminal success、12.5%
  failure；
- critic validation gate 已关闭，不再因 ranking 或 coverage 停止 Actor。

但这一 refresh **不能解决 Actor/Critic 分布偏移**，原因有两层。

第一，它仍只训练 replay 中的旧动作：

$$
(s,A^{\mathrm{data}},r,s')\sim\mathcal D_{\mathrm{old}},
$$

而 Actor 真正需要可靠评分的是：

$$
A^\pi\sim\pi_\theta(\cdot\mid s).
$$

没有执行 $A^\pi$ 后的新奖励和新状态，Critic 无法知道这些新动作的真实价值。
反复拟合 $A^{\mathrm{data}}$ 不能给 $A^\pi$ 增加监督。

第二，离线 Critic 原本通过 `apply_n_step_on_load` 使用 TD10；当前
`train_flash_ogpo.py` 直接加载原始 1-step replay，并没有执行这一步。因此联合
阶段的 refresh 实际是 TD1 口径，不是原来的 TD10 口径。这会让已经选好的 TD10
Critic 向另一个 target 定义漂移。

所以，当前周期 refresh 最多算“在旧数据上的低学习率维护”，不是 online critic
adaptation，也不是 distribution-shift 修复。

## 9. 是否需要重新训练 Critic

### 9.1 没有新数据时

不建议仅因为 Actor 在变化，就从头重训 Critic，或者持续在同一批 100 episodes
上更新。数据没有增加时，监督信息没有增加；训练更久主要增加过拟合、遗忘和
target drift 风险。

更合理的固定数据 pipeline 是：

1. 固定已经选好的 TD10 step-12000 Critic；
2. 使用 three-QV、`wsupport`、reference KL 和 success BC 限制 Actor 不要离开
   replay support 太远；
3. 独立记录 Critic 的 ranking、coverage 和 Q 分布作为诊断，但不设硬 gate；
4. 用环境成功率选择 Actor checkpoint，而不是持续改写 Critic。

若仍想在纯离线条件下研究 Critic，可以训练不同随机种子的 ensemble，或者加入
CQL/保守 OOD 惩罚。这些方法只能让 Critic 对未知动作更保守，不能凭空知道新动作
的真实回报。

### 9.2 有新 rollout 时

真正解决分布偏移需要交替采集和训练：

1. 固定一个 Critic snapshot，训练 Actor 约 250 到 500 个有效 step；
2. 用当前 Actor 新采集约 20 到 50 个 episodes，保留成功和失败；
3. 把新 transition、奖励、终止状态加入 replay，并保留旧数据防止遗忘；
4. 按统一的 TD10 定义重建 target；
5. 用新旧数据混合训练 Critic，例如 50% 新数据、50% 旧数据；
6. 重新冻结 Critic，再训练下一阶段 Actor；
7. 用独立 holdout 和环境成功率评估，不用单次 coverage 决定停止。

这才是接近 OGPO online actor-critic 交替更新的低成本 VLA 版本。

## 10. 主要差异汇总

| 项目 | OGPO-Origin | 当前 DIVL-OGPO |
|---|---|---|
| Critic | 共享 VLM + 10 个标量 Q | 共享 VLM + 3Q-3V |
| V | 无 | 201 atoms 的三个状态分布 |
| Critic target | 纯 MC return | 正式 Critic 为 ensemble-mean TD10 |
| Critic loss | MC MSE | Q MSE + V categorical CE |
| VLM | 全量微调 | 全量微调 |
| Actor state batch | 1 | 8 |
| 每状态候选 | 32 | 4 |
| Advantage baseline | 同状态候选 Q 均值 | 每个 Q 对应的熵分位数 V |
| 多头使用 | 10Q 先平均 | 3 个 Q-V 必须同号 |
| Advantage 尺度 | 原始组内差值 | running MAD + `wstate` + `wsupport` |
| Flow ratio | 10-step 完整链 | 8-step 中随机一个转移的 Flash surrogate |
| 时间修正 | 无 | 解析 TGR，范围 $[0.25,4]$ |
| PPO epoch | 1 | 1 |
| Reference KL | 关闭 | 系数 0.01，post-KL 0.15 原子回滚 |
| Success BC | 关闭 | 系数 0.02 |
| Critic gate | 强制 Actor | 已明确关闭 |
| Best-of-N | 关闭 | 关闭 |

## 11. 配置和入口

Origin：

- 公共配置：`configs/ogpo/pi05_ogpo_origin_common.yaml`
- Critic：`configs/ogpo/pi05_ogpo_origin_critic_100ep.yaml`
- Actor：`configs/ogpo/pi05_ogpo_origin_actor_100ep.yaml`
- Critic Slurm：`scripts/pi05_ogpo_origin_critic_30k.slurm`
- Actor Slurm：`scripts/pi05_ogpo_origin_actor_500.slurm`

DIVL：

- TD10 Critic：
  `configs/ogpo/pi05_gemma_udivl_critic_100ep_td10_balanced_30k_b32.yaml`
- 当前 8-GPU 联合配置：
  `configs/ogpo/pi05_jax_flash_ogpo_td10_tgr_b8_8gpu_joint_10000.yaml`
- 当前联合 Slurm：
  `scripts/pi05_actor_td10_tgr_b8_8gpu_joint_10000_hw40g.slurm`
- 当前编号 step-8000 checkpoint：
  `outputs/ogpo/checkpoints/click_mouse_pi05_jax_flash_td10_tgr_b8_8gpu_joint_10000/step_8000/`

需要注意：配置中的 `training.actor_steps` 是本次进程的 local loop 次数，而
`actor_start_step` 只用于日志编号；若从 7500 恢复并设置 `actor_steps: 10000`，
理论日志终点是 17499，而不是 9999。后续 resume 配置应按“还要新增多少步”填写。
