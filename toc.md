> **Status:** legacy table of contents. Use
> [`paper/notes/narrative.md`](paper/notes/narrative.md) and the current
> [`paper/main.tex`](paper/main.tex) section order as the source of truth.

# ICLR 2027：When Gaussian Learning Stops Scaling: Resolution-Stable Representations via Operator-Ideal Regularization

# Part II. Table of Contents

## 1. Introduction

### 1.1 The finite-resolution blind spot

每个有限维 Gaussian objective 都合法，但随着 resolution 增大可能失稳。

### 1.2 Main observation

Schatten-2 regularity给出正确的 resolution-limit boundary。

### 1.3 Contributions

控制在四条：

1. resolution-stability criterion；
2. sharp spectral phase transition；
3. representation learning as operator-ideal filtering；
4. empirical validation across generative / operator-learning settings。

---

## 2. Gaussian Learning Across Resolution

### 2.1 Resolution-indexed Gaussian objectives

定义 ((\mu_D,\nu_D))、relative perturbation (P_D)。

### 2.2 The Schatten-2 diagnostic

[
T_2(D)=|P_D|_{S_2}^2.
]

### 2.3 Why finite-dimensional diagnostics miss the transition

解释 finite (D) 与 continuum limit 的区别。

---

## 3. A Spectral Phase Transition in Gaussian Stability

### 3.1 Uniform (S_2) control

主定理。

### 3.2 Divergence outside the (S_2) regime

converse / lower bound。

### 3.3 Power-law spectra

[
s_j\asymp j^{-\alpha}.
]

### 3.4 Stable, critical, and unstable regimes

[
\alpha>\frac12,\quad
\alpha=\frac12,\quad
\alpha<\frac12.
]

Figure 1 放这里。

---

## 4. Representations as Operator-Ideal Filters

### 4.1 Encoder-induced spectral transformation

定义 (P_D^{(E)})。

### 4.2 Compression is not enough

给 counterexample / proposition。

### 4.3 Learned encoders and Schatten regularity

Figure 2。

这一节的 punchline：

> A useful representation need not merely reduce dimension; it can alter the operator ideal governing the continuum limit.

---

## 5. From Schatten Instability to Optimization Instability

### 5.1 Gradient-variance residue

连接 (T_2) 与 stochastic gradients。

### 5.2 Matrix-free diagnostics

Hutchinson / trace estimator。

### 5.3 Predicting instability before training at full resolution

低 resolution spectrum → 高 resolution prediction。

Figure 3。

这是整篇最 ML 的 section。

---

## 6. Experiments

### 6.1 Experimental protocol

resolution ladder、seeds、compute、baselines。

### 6.2 VAE spectral filtering

只保留最必要结果。

### 6.3 Diffusion Gaussian-prior stability

如果这一组更强就作为主 benchmark。

### 6.4 Resolution transfer in neural operators

FNO / PDE。

### 6.5 Comparison with conventional regularization

weight decay、spectral norm、Jacobian penalty、low rank。

Figure 4。

---

## 7. Discussion and Limitations

### 7.1 What (S_2) does and does not guarantee

不能声称 (S_2) 自动保证 generalization。

### 7.2 Finite resolution versus genuine infinite-dimensional membership

避免 reviewer 抓 logical overclaim。

### 7.3 Beyond Gaussian objectives

只点到为止。

---

## 8. Related Work

如果版面紧，可以并入 Introduction。

---

## 9. Conclusion

一句话回收：

> Resolution scaling exposes an operator-ideal boundary invisible at any fixed dimension, and representations can learn to move Gaussian learning problems across that boundary.

---

# Appendix

## A. Proofs

## B. Spectral estimators and matrix-free algorithms

## C. Full VAE Jacobian spectra

## D. Additional diffusion experiments

## E. FNO resolution-scaling curves

## F. Hyperparameters and compute

## G. Reproducibility / SHA256 provenance

---

# 最短施工顺序

* [ ] **Day 1–3：**把理论统一成一个 (P_D,T_2(D)) notation，证明主 theorem + power-law corollary。
* [ ] **Day 4–7：**整理现有 VAE / diffusion / FNO raw logs，删除重复 toy。
* [ ] **Week 2：**跑 real benchmark + seeds。
* [ ] **Week 3：**跑 conventional regularizer baselines + (\alpha=0.49/0.50/0.51)。
* [ ] **Week 4：**resolution-held-out prediction + ablations。
* [ ] **Week 5：**写 9-page submission，所有次要实验移 appendix。
* [ ] **最后：**只允许修 claim、补 reviewer hole；禁止发明新数学。
