# Paper narrative lock

## North star

> A continuum-stability theory of noise injection in high-dimensional
> learning, with Schatten filtering as the algorithmic consequence.

全篇只服务一个中心：

\[
\mathcal S_2\text{ is the stability threshold for refinement-consistent
Gaussian perturbations.}
\]

Diffusion 是最吸睛的实例，neural operator 是最干净的 resolution-scaling
test bed，Gaussian measure theory 是骨架，\(\det_2\) 是 renormalization
machinery。

## Four-layer launch

1. **病：resolution-dependent noise instability。** 固定维度上的合法性不等于
   refinement stability。相同 nominal noise scale 会在新增模式上积累 spectral
   mass，使 penalty 和 gradient 被 cutoff/high-frequency multiplicity 主导。
2. **药：Schatten-filtered noise。** 对 whitened relative covariance
   \(K_N\) 施加 ideal-aware spectral gate。\(\mathcal S_1\) 使用普通
   trace/log-det；\(\mathcal S_2\setminus\mathcal S_1\) 使用
   Wick--Carleman/\(\det_2\)；\(\mathcal S_2\) 外由 gate 改变谱尾。
3. **知识：sharp \(\mathcal S_2\) threshold。** 定理同时控制 renormalized
   objective、sample log-density fluctuation 与 power-law 三相边界。
4. **ML 世界：synthetic → operator learning → latent diffusion。** 每层证据只
   承担一个职责，不做结果仓库。

## Claim hierarchy

1. **Headline theorem:** dimension-uniform \(\mathcal S_2\) control is
   equivalent to cutoff stability under explicit uniform spectral conditions.
2. **Algorithmic consequence:** an ideal-aware gate enforces this property;
   \(\det_2\) supplies the correct finite part on
   \(\mathcal S_2\setminus\mathcal S_1\).
3. **Optimization consequence:** \(T_2\) predicts the resolution scaling of
   Gaussian objective fluctuations and probe-induced gradient variance.
4. **Representation thesis:** latent encoders may act as implicit
   operator-ideal filters; the paper tests this rather than assuming it.

## Main figures

- **Figure 1 — The disease and the boundary.** Matched-amplitude power-law
  spectra around \(\alpha=1/2\); show bounded/logarithmic/polynomial \(T_2(N)\),
  the \(\det_2\) residue, and probe-induced optimization fluctuations.
- **Figure 2 — Ideal-aware gate, not generic low-pass.** Frequency-matched
  perturbations with identical diagonal, trace, operator norm, radial PSD, and
  low-frequency response but different cumulative \(\mathcal S_2\) behavior.
- **Figure 3 — Operator-learning consequence.** Raw \(T_2(N)\), noise-scale
  drift, discretization defect, and zero-shot transfer across unseen grids.
- **Figure 4 — Representation audit.** Pixel, isometric, random, and learned
  encoders; distinguish input-noise pushforward from the actual diffusion
  covariance update, and report canonical raw \(T_2\) rather than only normalized
  effective-rank summaries.

## Nine-page budget

- Abstract: 0.25 page
- Introduction: 1.25 pages
- Problem and finite-resolution blind spot: 1.25 pages
- Method and sharp theorem: 2.25 pages
- Experiments: 3.25 pages
- Related work and conclusion: 0.75 page

Proofs, estimator derivations, full spectra, secondary domains, hyperparameters,
and provenance go to the appendix.

## Experiment admission rule

An experiment enters the main paper only if it does one of four jobs:

1. verifies the sharp threshold;
2. identifies why ideal membership matters beyond frequency appearance;
3. connects \(T_2\) to a training-relevant instability;
4. demonstrates value under resolution transfer or diagnoses an encoder.

The principal comparison is refinement consistency at matched task/noise
budget—not single-resolution SOTA.

## Language locks

- Sampling amplitudes are (B_N); the relative covariance is always
  (K_N=B_NB_N^ast). Gates, (det_2), and raw (T_2) are computed from
  (K_N), so a diagonal amplitude (b_j) contributes (b_j^4) to (T_2).
- Say **dimension-uniform Schatten control**, never finite-matrix membership.
- Say **probe-induced gradient variance** unless minibatch/SGD randomness is
  explicitly varied.
- Say **controlled linear encoder** for synthetic encoder studies.
- Present Feldman--Hajek and Carleman--Fredholm as explanatory machinery; the
  protagonist is resolution-dependent training instability.
- Describe latent diffusion with **we ask whether** until the encoder audit has
  supplied the answer.
