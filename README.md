
# 论文修改建议：从判定层升级到决策层

当前稿件只完成了**判定层**，还没有完成**决策层**：

* **判定层**：什么情况下随分辨率发散？答案是统一 \( S_2 \) 控制，幂律临界点为 \( \alpha=1/2 \)。
* **决策层**：在有限分辨率、有限扩散步数、有限算力和有限质量损失下，究竟应该怎么选预算、怎么滤、怎么训练、怎么评估？

当前 Theorem 3.1 确实给出了 \( T_2 \)、renormalized objective 和 log-density variance 之间的精确关系，但它只能回答“是否最终发散”，不能直接回答“1024 分辨率是否已经大到不能接受”。 更严重的是，现有第 4.3、4.4 节主要还是实验协议描述，而不是带有数字和质量结果的完整实验；第 8 页 Figure 2 甚至明确说明所有曲线都是 closed-form。

所以，下一版不应该只是增加一节“Limitations”，而应该把论文的中心问题改成：

> **\( S_2 \) 给出渐近可行域；实际学习问题是在有限分辨率、扩散时间、任务质量和计算成本之间进行 Schatten 预算分配。**

---

## 1. “多大算大”：必须从渐近临界点升级为有限分辨率证书

### 1.1 \( \alpha=1/2 \) 只回答增长类别，不回答实际风险

对于纯幂律谱

$$
s_j(K)=c j^{-\alpha},
$$

在保留前 \( n \) 个模式时，

$$
T_2(n)=\sum_{j=1}^{n}s_j(K)^2
      =c^2\sum_{j=1}^{n}j^{-2\alpha}.
$$

这给出

$$
T_2(n)\approx
\begin{cases}
c^2\zeta(2\alpha), & \alpha>1/2,\\[2mm]
c^2\log n, & \alpha=1/2,\\[2mm]
\dfrac{c^2}{1-2\alpha}n^{1-2\alpha}, & \alpha<1/2.
\end{cases}
$$

论文的 Corollary 3.2 已经陈述了这三个增长区间。但还缺少两个工程上更重要的结论：

1. **\( \alpha>1/2 \) 只意味着不继续发散，不意味着当前 \( T_2 \) 很小。**
2. **\( \alpha<1/2 \) 只意味着最终发散，不意味着在当前分辨率已经不可用。**

因此，实际报告中必须同时给出三个量：

$$
\boxed{
\text{asymptotic class}
\quad+\quad
T_2(R_{\max})
\quad+\quad
\text{observed growth slope}
}
$$

而不能只报一个拟合出来的 \( \alpha \)。

### 1.2 按论文自己的谱设置，1024×1024 已经能算出具体数字

第 4.1 节使用

$$
h_j=0.45j^{-\alpha},
\qquad
K_j=\frac{h_j}{1-h_j},
$$

并测试 \( \alpha=0.6,0.5,0.4 \)。

假设 \( n=1024^2=1{,}048{,}576 \)，每个像素对应一个标量模式，直接对该有限谱求和，得到：

| \( \alpha \) | \( T_2(K) \) | \( L_N^{\mathrm{ren}} \) | \( \sqrt{\operatorname{Var}(\ell_N)}=\sqrt{T_2/2} \) |
| ---------: | ---------: | ---------------------: | -------------------------------------------------: |
|       0.60 |      1.766 |                  0.362 |                                              0.940 |
|       0.50 |      3.802 |                  0.853 |                                              1.379 |
|       0.40 |     16.778 |                  4.041 |                                              2.896 |

这些数字马上比“\( \alpha=0.5 \) 是临界点”更有工程意义。

例如，可以把“多大算大”定义成一个可配置风险预算：

$$
\sqrt{\operatorname{Var}(\ell_N)}\leq \sigma_{\max}
\quad\Longleftrightarrow\quad
T_2(N)\leq 2\sigma_{\max}^2.
$$

* 如果允许 log-density ratio 的一倍标准差最多为 \( 1 \)，则预算是 \( T_2\leq2 \)。
* 如果最多为 \( 2 \)，则预算是 \( T_2\leq8 \)。

在这个标尺下：

* \( \alpha=0.6 \) 在 1024² 上尚处于严格预算附近；
* \( \alpha=0.5 \) 已经超过严格预算；
* \( \alpha=0.4 \) 明显超过较宽松预算。

这里的阈值仍然需要通过任务质量实验校准，但至少它把“发散”变成了一个有单位、有数值、可选择的预算。论文已经证明了

$$
\operatorname{Var}_{q_N}[\ell_N]=\frac12T_2(N),
$$

所以这个解释不是额外的启发式，而是定理的直接工程化。

还要明确说明：对于 RGB、latent channels 或 channel multiplicity，\( n \) 不是简单的图像宽乘高。如果三个通道具有相似且独立重复的谱，\( T_2 \) 会大约乘三。论文必须报告真实的有效模式数和内积归一化。

### 1.3 谱在 0.4 和 0.6 之间震荡时，不应该“信某个 \( \alpha \)”

这时一个全局幂律拟合本身就是错误的问题。应该改用 dyadic shell mass：

$$
B_\ell
=
\sum_{2^\ell\leq j<2^{\ell+1}}s_j(K)^2,
\qquad
T_2(2^L)=\sum_{\ell<L}B_\ell.
$$

对于纯幂律：

$$
B_\ell\asymp 2^{(1-2\alpha)\ell}.
$$

因此：

* \( \alpha=0.6 \)：shell mass 随 \( \ell \) 衰减；
* \( \alpha=0.5 \)：每个 shell 增加近似相同的质量；
* \( \alpha=0.4 \)：shell mass 随 \( \ell \) 增长。

如果相同宽度的 dyadic blocks 在 \( 0.6 \) 和 \( 0.4 \) 间交替，那么 \( 0.4 \) blocks 的 shell mass 会沿一个子序列增长，整体仍然不属于统一 \( S_2 \)。但如果 \( 0.4 \) 区域越来越稀疏，或者其振幅同步下降，也可能保持可控。

所以答案不是“信 0.4”或“信 0.6”，而是：

> **信实际累计的 \( T_2 \) 曲线和每个频谱壳层新增的平方质量，不信单一幂律指数。**

论文应增加一个 **mixed-spectrum gray-zone experiment**，至少覆盖：

* 交替幂律 block；
* 前重后轻谱；
* 少数尖峰加快速衰减尾；
* 相同平均斜率、不同 multiplicity 的谱。

这会比再增加一个纯幂律曲线有价值得多。

---

## 2. Diffusion 的 1000 步：线性、平方还是指数，取决于你究竟在累计什么

当前论文只定义了每个时间点的 downstream relative covariance update \( K_{\mathrm{diff},N,t} \)，但没有定义跨时间的整体稳定对象。因此，“1000 步是线性还是指数”在当前稿件中实际上没有答案。

需要把三个不同问题明确分开。

### 2.1 每步 Gaussian objective 的总和：通常是线性的

如果比较两个 Gaussian Markov chains，路径 KL 有 chain rule：

$$
D_{\mathrm{KL}}(q_{0:T}\|p_{0:T})
=
D_{\mathrm{KL}}(q_0\|p_0)
+
\sum_{t=1}^{T}
\mathbb E_q
D_{\mathrm{KL}}
\bigl(q_t(\cdot|x_{t-1})\|p_t(\cdot|x_{t-1})\bigr).
$$

如果每步 \( T_2(K_t)\leq\tau \)，那么总路径目标一般是 \( O(T\tau) \)。即使每一步都满足统一 \( S_2 \)，1000 步后总预算仍可能很大。

因此应定义

$$
B_{\mathrm{path}}(N,T)
=
\sum_{t=1}^{T}w_tT_2(K_{N,t}),
$$

其中 \( w_t \) 来自真实 diffusion objective 或 timestep sampling 权重。

### 2.2 sample log-ratio 的方差：在 martingale/独立增量条件下也是线性的

如果每一步的中心化 log-ratio 构成 martingale difference，则

$$
\operatorname{Var}\left(\sum_{t=1}^{T}\ell_t\right)
=
\frac12\sum_{t=1}^{T}T_2(K_t).
$$

但存在跨时间相关性时，还会出现 covariance terms。论文不能把单步 Theorem 3.1 不加说明地外推到 1000 步。

### 2.3 最终合成 covariance：可能是平方甚至指数增长

如果每步 relative map 按乘法合成，并且算子可交换，那么单个谱方向上

$$
1+\kappa_{j,1:T}
=
\prod_{t=1}^{T}(1+\kappa_{j,t}).
$$

当每步都是同一个小正扰动 \( \kappa \) 时，

$$
\kappa_{1:T}
=
(1+\kappa)^T-1.
$$

* 当 \( T\kappa\ll1 \) 时，\( \kappa_{1:T}\approx T\kappa \)，其平方质量约为 \( T^2\kappa^2 \)；
* 当 \( T\kappa \) 不再小时，增长呈指数型。

但标准 variance-preserving diffusion 中还存在收缩系数，最终 covariance 通常是带 schedule 权重的累加，而不是裸乘积。因此正确答案不是“1000 倍”或“指数爆炸”，而是必须报告：

$$
\boxed{
B_{\max}=\max_tT_2(K_t),\quad
B_{\mathrm{path}}=\sum_tw_tT_2(K_t),\quad
B_{\mathrm{end}}=T_2(K_{0\rightarrow T})
}
$$

论文应增加一个“resolution × time”二维实验：

* 分辨率：256、512、1024；
* 步数：10、50、100、250、1000；
* schedule：恒定扰动、VP schedule、实际训练 schedule；
* 输出：单步最大预算、路径预算、最终 covariance 预算和生成质量。

这会回答用户真正关心的 1000 步问题。

---

## 3. “Schatten 滤波会杀死高频”是有效风险，但不是理论上必然发生

这里需要修正一个表述：**Schatten filtering 不等同于低通滤波。**

论文已经强调它是 ideal-aware 而不是 frequency-aware；它控制的是 whitening 后累计的奇异值平方质量，而不是直接按空间频率裁剪。当前方法甚至包含两种不同 gate：

* Equation 12 的 radial projection：对所有模式统一缩放；
* Equation 13–14 的 envelope gate：根据指定模式预算进行非均匀缩放。

所以在有限维度下，也不存在一个天然可识别的“非-\( S_2 \) 高频部分”供直接砍掉。每个有限矩阵本身都在 \( S_2 \) 中，问题是整个分辨率序列的累计预算。

不过用户的核心批评仍然成立：

> 当前论文没有告诉读者，有限的 \( S_2 \) 预算应该分配给哪些模式。

如果高频毛发、雨滴、砂石模式对任务很重要，纯频率 envelope 很可能分配错误。应将方法从简单 Schatten filtering 改成 **task-aware Schatten budgeting**。

### 3.1 增加任务加权的投影问题

可以定义：

$$
\widetilde K_\tau
=
\arg\min_{\widetilde K\succeq0}
D_{\mathrm{task}}(\widetilde K,K)
\quad
\text{s.t.}\quad
\|\widetilde K\|_{S_2}^2\leq\tau,
\quad
\|\widetilde K\|_{\mathrm{op}}\leq\rho.
$$

一个可计算的谱近似是：

$$
\min_{\widetilde\lambda_j}
\sum_j q_j(\widetilde\lambda_j-\lambda_j)^2
\quad
\text{s.t.}\quad
\sum_j\widetilde\lambda_j^2\leq\tau.
$$

这里 \( q_j \) 是任务敏感度，可以来自：

* reconstruction/perceptual loss 对该模式的梯度；
* decoder Jacobian sensitivity；
* 对下游分类或生成质量的影响；
* 高频纹理专门构造的 validation loss。

其拉格朗日解具有形式

$$
\widetilde\lambda_j
=
\frac{q_j}{q_j+\mu}\lambda_j.
$$

任务重要度高的模式，即使位于高频，也会获得更多预算；冗余 channel multiplicity 或任务无关模式会被优先压缩。

这正好与论文 Figure 2 的精神一致：相同的 Fourier 外观可能隐藏完全不同的 eigenvalue multiplicity，因此不应只按频率处理。Figure 2 在 \( d=1{,}048{,}576 \) 时给出 512.5 倍的 \( T_2 \) 差异，但目前仍然只是 closed-form 构造。下一步应该证明 task-aware gate 能在真实模型中削减这种隐藏 multiplicity，而不必牺牲关键高频。

### 3.2 必须增加高频任务反例

为了防止论文给人“稳定性一定优先于信息”的印象，应主动加入一个反例：

* 任务信号刻意放在少数高频模式中；
* 普通 low-pass filter 会稳定但任务性能崩溃；
* radial gate 会保留形状但整体降低信号；
* task-aware Schatten gate 保留少数关键高频模式，同时压缩大量无关模式。

这会把用户的“我宁愿不稳定，也要细节”变成论文中的一个合法 operating point，而不是论文需要回避的反对意见。

---

## 4. Encoder 的 \( S_4 \) 条件：必须从“审判标准”变成“可训练目标”

这里也需要修正一句过强的表述：

> “在 ImageNet 上计算 VAE Jacobian，99.9% 会发散”目前既没有实验支持，在数学上也不能通过单一分辨率得出。

固定在 256×256 或 1024×1024 时，有限 Jacobian 的 \( S_4 \) 范数总是有限。所谓“发散”只能指：

$$
\sup_N
\left\|
(C^{\mathrm{ref}}_{z,N})^{-1/2}
J_N(x)\Sigma_N^{1/2}
\right\|_{S_4}
=
\infty
$$

沿分辨率 ladder 不受控。论文自己也强调，单个有限矩阵属于 Schatten 类是自动的，真正问题是 dimension-uniform control。

但用户指出的主要问题完全成立：Equation 24 给出了漂亮的等式，

$$
T_2^{(E)}(N;x)
=
\|F_N(x)\|_{S_4}^4,
$$

却没有给出 encoder 不满足条件时的训练算法。

### 4.1 增加一个明确的 Encoder Repair Algorithm

训练目标可以写成

$$
\mathcal L_{\mathrm{enc}}
=
\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{perc}}\mathcal L_{\mathrm{perc}}
+
\mu
\left[
\widehat T_2^{(E)}(N;x)-\tau_E
\right]_+
+
\nu
\left[\widehat\beta_E\right]_+^2,
$$

其中

$$
\widehat\beta_E
=
\frac{
\log \widehat T_2^{(E)}(N_2)
-
\log \widehat T_2^{(E)}(N_1)
}{
\log n_{N_2}-\log n_{N_1}
}
$$

惩罚跨分辨率增长，而不是只惩罚某一个分辨率上的绝对值。

训练时随机采样分辨率 \( N \)，而不是只在单一 ImageNet 尺寸上训练。

### 4.2 推荐两阶段修复，而不是一开始全量 double-backprop

更实际的方案是：

**第一阶段：冻结原 encoder。**

在 latent 输出后插入一个小型 diagonal/channel/frequency gate \( G_\psi \)，使

$$
F_N'(x)=G_\psi F_N(x).
$$

只优化 \( G_\psi \)、少量 adapter 或最后一个 encoder block。这样既能控制推前 covariance，又不会立刻承担完整 encoder 的二阶自动微分成本。

**第二阶段：必要时解冻最后若干层。**

使用较小权重的 \( S_4 \) regularizer 微调 encoder，同时保持 reconstruction 和 perceptual loss。

论文应该比较：

* 原始 VAE；
* spectral normalization；
* Frobenius/Jacobian norm regularization；
* \( S_4 \) regularization；
* latent adapter；
* 显式 Schatten noise gate。

论文相关工作已经指出，spectral normalization 只控制最大奇异值，不能控制累计平方质量。可以进一步给出一个简单且工程化的解释：

若有效秩为 \( r_N \)，则

$$
\|F_N\|_{S_4}^4
\leq
r_N\|F_N\|_{\mathrm{op}}^4.
$$

所以仅仅保持 \( \|F_N\|_{\mathrm{op}}\leq c \) 而让 \( r_N \) 随分辨率增长，仍然可能发散。为了固定预算 \( \tau \)，最大奇异值至少需要按

$$
\|F_N\|_{\mathrm{op}}
\lesssim
\left(\frac{\tau}{r_N}\right)^{1/4}
$$

缩小。这能非常直接地解释为什么普通 spectral normalization 不够。

### 4.3 必须给真实 Encoder 表格

当前第 4.4 节列出了 learned VAE、random encoder、partial isometry、antialiased downsampling 等 controls，但没有给出实际数字。主文至少需要一张表：

| Encoder | \( T_2^E(256) \) | \( T_2^E(512) \) | \( T_2^E(1024) \) | growth exponent | reconstruction quality | repair 后质量 |
| ------- | -------------: | -------------: | --------------: | --------------: | ---------------------: | ---------: |
| 原始 VAE | | | | | | |
| + Spectral Norm | | | | | | |
| + \( S_4 \) Reg | | | | | | |
| + Latent Adapter | | | | | | |

没有这张表，“encoder 可能是隐式滤波器”仍然只是一个研究提案，而不是论文结论。

---

## 5. Rademacher 探针：100 次并不必然慢 100 倍，但当前稿件确实没有成本模型

当前 Equation 21 给出了

$$
\widehat T_2
=
\frac1m\sum_{r=1}^m\|K_Nv_r\|_2^2,
\qquad
\mathbb E\widehat T_2=T_2,
$$

并说明可以复用探针估计 determinant series。附录也明确承认 determinant estimator 需要 \( O(mL) \) 次 operator–vector products，并可对该过程反向传播。

但论文没有回答一次 operator–vector product 相当于多少次模型 forward/backward，也没有给出 probe 数量如何选。

### 5.1 首先区分四类不同成本

| 被估计的 \( K \) | 一次 \( Kv \) 的成本 |
| ---------------------------------------- | ---------------------------- |
| Fourier-diagonal covariance | FFT 或逐点乘，可直接精确求和，通常不需要 probe |
| 显式 low-rank covariance | 两次低秩矩阵乘法 |
| Encoder pushforward \( K=FF^\ast \) | 一次 VJP 加一次 JVP |
| 完整 UNet/Attention Jacobian-induced \( K \) | 穿过模型的 JVP/VJP；训练该正则时可能需要二阶梯度 |

因此，“10 亿参数模型”本身并不决定成本。关键是 \( K \) 是否由完整模型 Jacobian 隐式定义。对于论文当前的 Fourier gate，很多量可以精确计算，根本不应使用 100 个探针。论文自己也提到 Fourier-diagonal gate 可以逐模式精确计算 determinant。

### 5.2 \( m=100 \) 不是固定要求

附录给出的单探针方差为

$$
\operatorname{Var}(v^\top Av)
=
2\sum_{i\neq j}A_{ij}^2,
\qquad A=K^\ast K,
$$

平均 \( m \) 次后除以 \( m \)。因此所需 \( m \) 取决于算子在探针基底中的 off-diagonal mass，而不是参数量。

论文应实现三级模式：

1. **训练模式：** 每个 iteration 只用一个新 probe，利用跨 iteration 的随机性，并用 EMA 监控预算。它是低成本的 noisy regularizer，不作为严格置信证书。
2. **周期审计模式：** 每隔若干步在固定 validation batch 上自适应增加 probe，直到相对置信区间达到预设精度。
3. **最终证书模式：** 对结构化 gate 精确计算；对隐式算子使用较大的自适应 \( m \)，并报告区间。

还应该使用相同 probe 比较：

* 不同分辨率；
* filter 前后；
* 不同 gate。

这种 common-random-number 设计能显著降低“差值估计”的波动。

### 5.3 “只算最后几层”可以做，但必须降级声明

只审计最后几层、少数 attention blocks 或 latent adapter 是合理的工程近似，但不能继续称为整个网络的全局 \( S_2 \) certificate。论文应区分：

* **global certificate**；
* **blockwise diagnostic**；
* **localized training surrogate**。

否则廉价近似会偷偷继承完整定理的措辞，这是 reviewer 很容易攻击的地方。

---

## 6. 任务失真与稳定性：不能再用一个“matched point”，必须画完整 Pareto Frontier

当前论文说实验以“matched task and noise budgets”为主轴，第 4.2 节也声称在 matched task distortion 和 noise budget 下比较各方法。但没有说明：

* 两个量如何同时匹配；
* 如果无法同时匹配，优先匹配哪一个；
* 匹配点是否经过挑选；
* 质量损失能否通过 retraining 恢复；
* MSE 是否真的代表毛发、雨滴等感知细节。

这会给人很强的 cherry-picking 空间。

### 6.1 定义两个独立目标

对每个 gate \( g \) 和预算 \( \tau \)，定义：

$$
Q(g,\tau)
=
\text{task/perceptual quality loss},
$$

$$
S(g,\tau)
=
\max_{R\in\mathcal R}
\left\{
\frac{T_2(R)}{T_2(R_0)},
\frac{\operatorname{GradVar}(R)}
     {\operatorname{GradVar}(R_0)},
\left|\widehat\sigma_R-\widehat\sigma_{R_0}\right|
\right\}.
$$

然后扫过：

* \( \tau \)；
* radial gate；
* Fourier envelope；
* task-aware gate；
* unfiltered；
* renormalization-only；
* spectral normalization 或 low-rank baseline。

画出

$$
\{(S(g,\tau),Q(g,\tau))\}
$$

的完整 Pareto frontier。

### 6.2 至少报告三个 operating points

而不是只报告一个“我们匹配后的最好结果”：

1. 给定稳定性上限时，质量最好的是谁；
2. 给定最大质量下降时，稳定性最好的是谁；
3. frontier 的 knee point 在哪里。

未滤波模型必须作为 frontier 的合法端点。这样“宁愿不稳定也要高频细节”不再是对论文的反驳，而是用户选择的一个工作点。

### 6.3 区分立即插入滤波和重新适配后的质量

需要两条 frontier：

* **Frozen-model frontier：** 在预训练模型中直接插入 gate，测量即时伤害；
* **Adapted-model frontier：** 允许短期或完整 retraining，测量模型能否补偿滤波。

如果只做 frozen evaluation，可能把优化不匹配误认为不可恢复的信息损失；如果只做 retrained evaluation，又可能隐藏部署时的真实代价。

### 6.4 不能只用 MSE

针对用户提出的毛发、砂石、雨滴，至少应加入：

* 标准生成质量指标；
* perceptual feature distance；
* 高频频带能量或 spectrum matching；
* texture-rich subset；
* 少量人工偏好或局部 patch fidelity；
* 高频承载任务信号的可控 synthetic benchmark。

这样“稳定性的代价”才会真正成为论文的主要结果之一。

---

## 建议直接重构论文主线

当前主线是：

> Gaussian objectives 在 \( S_2 \) 外发散，所以引入 Schatten filtering。

更强、也更落地的主线应改成：

> Gaussian noise 的渐近可行域由 \( S_2 \) 决定；在有限分辨率和有限 diffusion 时间中，实际问题是如何在任务效用约束下分配可计算的 \( T_2 \) 预算。

对应的新贡献可以压缩成三条：

### Contribution 1：Finite-resolution and temporal certificate

除了 \( \alpha=1/2 \) 的渐近定理，还提供：

* 有限 \( R \) 的实际 \( T_2 \)、objective 和 variance；
* mixed-spectrum shell diagnostic；
* diffusion path、maximum 和 end-state 三种时间预算。

### Contribution 2：Task-aware Schatten budgeting

不再把过滤描述成统一解药，而是提出：

* 任务加权预算分配；
* 保留高频重要模式；
* encoder repair adapter；
* 质量—稳定性 Pareto frontier。

### Contribution 3：Cost-bounded stochastic audit

提供：

* 一探针在线训练；
* 自适应 probe stopping；
* structured operator 精确计算；
* blockwise surrogate 与 global certificate 的明确区分；
* 实际 wall-clock、显存和 backward-equivalent 成本。

---

## 主文空间重分配建议

为了给上述内容腾主文空间，建议：

1. **保留** Theorem 3.1 和 Figure 2 的核心思想
2. **删除**：
   - 大部分 det\(_2\) 数值验证
   - 完整纯幂律曲线
   - estimator 推导细节

3. **主文优先展示**：
   - 一张真实 encoder 表（见 4.3）
   - 一张 diffusion 时间—分辨率图（见 2）
   - 一张质量—稳定性 Pareto 图（见 6.1）

现有 Figure 1 报告的 64 倍维度实验和 gradient variance 比例可以保留为理论 sanity check，而不能继续承担主要工程证据。

---

> **一句话概括下一版应发生的变化：**
>
> **定理告诉读者什么时候预算会爆；算法告诉读者怎样分配预算；Pareto 曲线告诉读者稳定性究竟值不值得。**