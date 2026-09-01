> **Status:** this is the original construction checklist. The canonical paper
> story, claim hierarchy, four-figure plan, and language locks now live in
> [`paper/notes/narrative.md`](paper/notes/narrative.md); the compilable draft is
> [`paper/main.tex`](paper/main.tex).

# ICLR 2027：When Gaussian Learning Stops Scaling: Resolution-Stable Representations via Operator-Ideal Regularization

# Part I: 施工清单

## 0. 一句话锁死论文

> **Gaussian learning objectives can be perfectly well-defined at every finite resolution yet become unstable as resolution increases; the relevant phase boundary is Schatten-2 regularity, and learned representations can act as operator-ideal filters that restore resolution stability.**

全文只服务这一句话。

核心链条固定为

[
\text{resolution }D\uparrow
\longrightarrow
\text{spectral tail}
\longrightarrow
T_2(D)=|P_D|_{S_2}^2
\longrightarrow
\begin{cases}
\text{objective drift},\
\text{gradient variance},\
\text{minibatch noise},\
\text{generalization / resolution robustness}.
\end{cases}
]

禁止扩张成泛泛的 Feldman–Hájek 论文、泛泛的 VAE paper、泛泛的 regularization paper。

---

# Part I. 施工清单

## A. P0：理论骨架——必须完成

* [ ] 定义 resolution-indexed Gaussian learning problem：
  [
  (\mu_D,\nu_D),\qquad P_D=\text{relative covariance perturbation}.
  ]

* [ ] 把所有实验使用的 (T_2) 统一成一个规范定义：
  [
  T_2(D)=|P_D|_{S_2}^2=\operatorname{Tr}(P_D^\ast P_D).
  ]

* [ ] 明确区分：

  * finite-dimensional well-definedness；
  * uniform-in-resolution stability；
  * genuine infinite-dimensional (S_2) membership。

* [ ] 主定理 A：给出一个 dimension-uniform stability implication，目标形状：
  [
  \sup_D|P_D|_{S_2}<\infty
  \Longrightarrow
  \sup_D \mathcal R_D<\infty,
  ]
  其中 (\mathcal R_D) 至少覆盖一个真正 training-relevant residue：
  gradient variance / Gaussian prior residue / ELBO correction。

* [ ] 主定理 B：在可计算模型中证明 failure direction：
  [
  |P_D|_{S_2}\to\infty
  \Longrightarrow
  \mathcal R_D\to\infty
  ]
  或至少获得 sharp lower bound。

* [ ] Corollary：若
  [
  s_j(P)\asymp j^{-\alpha},
  ]
  则出现三相：
  [
  \alpha>\frac12,\qquad
  \alpha=\frac12,\qquad
  \alpha<\frac12.
  ]

* [ ] 边界 (\alpha=1/2) 单独钉死 logarithmic growth：
  [
  T_2(N)\asymp \log N.
  ]

* [ ] 明确说明为什么 finite-(D) loss 本身发现不了这个 boundary。

* [ ] 把 Feldman–Hájek 降级成理论来源，而不是 paper 的叙事主角：
  “the infinite-dimensional Gaussian equivalence criterion suggests the correct stability quantity.”

### 理论部分完成判据

至少必须得到一个可以直接放 Abstract 的严格句子：

> Under an (S_2)-uniform perturbation condition, the relevant Gaussian training residue remains bounded uniformly over discretization; outside the (S_2) regime it diverges at the predicted spectral rate.

---

## B. P0：把 “encoder = operator-ideal filter” 做成真正命题

* [ ] 定义 encoder-induced perturbation：
  [
  P_D \mapsto P_D^{(E)}
  ]
  或对应 Jacobian/covariance pullback。

* [ ] 给出至少一个足够条件：
  [
  E_D
  \text{ 的 spectral action}
  \Longrightarrow
  P_D^{(E)}\in S_2
  ]
  或 uniform (S_2)-bound。

* [ ] 区分三种 baseline：

  1. pixel / identity；
  2. isometric encoder；
  3. learned encoder。

* [ ] 证明或解释：
  dimensionality reduction 本身不等于 Schatten filtering。

* [ ] 找一个 counterexample：
  encoder 降维，但 (T_2(D)) 仍然恶化。

这一步很重要，否则 reviewer 会说：

> “This is just compression.”

必须回答：

> **No. The relevant effect is spectral reshaping, not merely reducing ambient dimension.**

---

## C. P0：现有图重组

### Figure 1 — 整篇 paper 的 phase diagram

* [ ] 用现有 singular-spectrum + (\alpha=1/2) cutoff 做成一张极简主图。
* [ ] 左：(s_j\sim j^{-\alpha})。
* [ ] 中：(T_2(N)) scaling。
* [ ] 右：stable / critical / unstable 三个 training regime。
* [ ] 不放过多 toy curves。

目的：reviewer 30 秒内知道 theorem 在说什么。

### Figure 2 — Representation as operator-ideal filter

* [ ] 使用 VAE：
  pixel naive / CF / learned encoder / isometric encoder。
* [ ] 横轴统一 resolution 或 dimension。
* [ ] 主纵轴统一为 (T_2(D))。
* [ ] companion panel 放 effective rank 或 spectrum。
* [ ] learned encoder 是否真的改变 asymptotic scaling，必须给 confidence interval。

### Figure 3 — Gradient-variance litmus

* [ ] 保留目前最强的那张：
  filtered branch dimension-flat；
  unfiltered branch explosion。
* [ ] 加 5–10 seeds。
* [ ] error bars。
* [ ] 同图报告 (T_2(D)) 和 gradient variance scaling exponent。
* [ ] 做 log-log regression：
  [
  \operatorname{Var}\nabla \sim T_2^\gamma
  ]
  看是否存在稳定经验关系。

### Figure 4 — Real learning consequence

从下面选最强的一个作为正文，不要三个都塞满：

* [ ] diffusion prior objective；或
* [ ] FNO resolution generalization；或
* [ ] real VAE benchmark。

剩下的进 appendix。

---

## D. P0：必须从 toy 升级到 real benchmark

### 路线 1：Diffusion

* [ ] 选一个标准 dataset。
* [ ] 至少 3 个 resolutions。
* [ ] 同 architecture family。
* [ ] ordinary Gaussian prior vs (S_2)-controlled/filtered prior。
* [ ] 测：

  * ELBO / NLL proxy；
  * gradient variance；
  * minibatch variance；
  * convergence；
  * sample quality metric。

### 路线 2：VAE

* [ ] CIFAR-10 或其他标准 image dataset。
* [ ] resolution ladder。
* [ ] learned encoder / identity-like / isometric control。
* [ ] Jacobian spectrum。
* [ ] (T_2(D))。
* [ ] test ELBO / reconstruction / sample metric。

### 路线 3：FNO / operator learning

这个其实理论味最一致。

* [ ] 选标准 PDE dataset。
* [ ] train-resolution / test-resolution mismatch。
* [ ] ordinary vs (S_2)-filtered objective。
* [ ] 测 resolution transfer。
* [ ] 检查是否存在：
  [
  \sup_D T_2(D)<\infty
  \quad\Longleftrightarrow\quad
  \text{mesh-independent performance}.
  ]

### 最低提交标准

正文至少一个真实 benchmark。

最好两个 domain：

[
\text{generative learning}
+
\text{operator learning}.
]

---

## E. P0：baseline 防御

必须回答 reviewer 的四连问：

* [ ] 为什么不是 weight decay？
* [ ] 为什么不是 spectral normalization？
* [ ] 为什么不是 Jacobian regularization？
* [ ] 为什么不是 ordinary low-rank compression？

实验要求：

* [ ] parameter-matched。
* [ ] compute-matched。
* [ ] training-loss-matched。
* [ ] 比较 finite-resolution performance。
* [ ] 更重要：比较 resolution scaling。

核心论点必须是：

> Conventional regularizers may control finite-dimensional norms without enforcing dimension-uniform Schatten behavior.

---

## F. P1：统计可靠性

* [ ] 所有 headline result ≥ 5 seeds。
* [ ] 关键 phase-transition result 最好 ≥ 10 seeds。
* [ ] 报 mean ± CI。
* [ ] 不只画 endpoint。
* [ ] 报 scaling exponent。
* [ ] 做 resolution-held-out prediction：
  用低分辨率拟合 spectral exponent，预测高分辨率 instability。
* [ ] 然后真正跑高分辨率验证。

这是一个很强的实验：

[
D\le1024
\quad\Rightarrow\quad
\widehat{\alpha}
\quad\Rightarrow\quad
\widehat{T_2(4096)}
]

然后和真实 (D=4096) 对比。

这样 (T_2) 才从 descriptive statistic 升成 predictor。

---

## G. P1：最关键的 ablation

* [ ] 固定 architecture，只改 perturbation spectral tail。
* [ ] 固定 (T_2)，改变其他 norm，检查是否仍稳定。
* [ ] 固定 operator norm，改变 (T_2)，检查 instability 是否变化。
* [ ] 相同 latent dimension，不同 singular decay。
* [ ] 相同 parameter count，不同 (S_2) behavior。
* [ ] 专门跑 (\alpha=0.49,0.50,0.51)。

最后这个非常值钱。

如果 empirical transition 真能在 (1/2) 附近显出来，paper 的理论—实验连接会明显变硬。

---

## H. P1：reproducibility

已有本地 A100 + SHA256/provenance 基础直接利用。

* [ ] 固定 environment。
* [ ] 固定 random seeds。
* [ ] 保存 raw spectra，而不仅是 PNG。
* [ ] 保存每次训练的 config。
* [ ] 所有 figure 可由单个 script 从 raw logs 重建。
* [ ] checkpoint / dataset preprocessing 写 checksum。
* [ ] README 给出“一键重建 Figure 1–4”。
* [ ] 报 GPU 型号、训练时长、显存和总 compute。

---

## I. P1：Related Work 只守四块

* [ ] Feldman–Hájek / Gaussian measure equivalence。
* [ ] function-space / infinite-dimensional ML。
* [ ] Jacobian / spectral regularization。
* [ ] neural operators / resolution invariance。

不要写成 operator ideals 文献综述。

只解释：

> What existing ML regularizers control, and what they do not control about the resolution limit.

---

## J. P2：有时间才做

* [ ] Schatten-(p) family：
  [
  S_p,\quad p\neq2.
  ]
* [ ] non-Gaussian extension。
* [ ] adaptive spectral cutoff。
* [ ] online (T_2) estimator。
* [ ] Hutchinson trace estimator 的理论 variance。
* [ ] architecture search based on operator-ideal target。

这些全部禁止阻塞投稿。

---

## K. 明确砍掉

* [ ] 不重新解释完整 Borel–Kolmogorov story。
* [ ] 不讲 completion-relative determinant anomaly。
* [ ] 不展开 measure-theoretic philosophy。
* [ ] 不把所有 F–H 主论文 theorem 搬进来。
* [ ] 不提出五种新的 regularizer。
* [ ] 不做十个 toy dataset。
* [ ] 不追求 ImageNet SOTA。
* [ ] 不证明所有 architecture 都满足 theorem。
* [ ] 不再新增新的 phase variable，除非现有 (T_2) 失败。
