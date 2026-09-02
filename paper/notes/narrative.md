# Paper narrative lock

## Title

> **Task-Aware Schatten Budgeting for Resolution- and Time-Stable Latent Diffusion**

## North star

> The Schatten-\(2\) theorem defines the feasible region; task-aware finite
> resolution--time budgeting chooses the useful operating point.

The paper has one claim:

> At finite resolution and diffusion time, raw \(T_2\) is an actionable
> Gaussian risk budget, and allocating it by task sensitivity is the right way
> to preserve useful latent modes while controlling covariance accumulation.

The paper is about a decision layer, not another proof that divergence can
occur. The asymptotic theorem supplies the boundary. The method allocates a
finite budget. The quality--stability frontier determines whether an operating
point is worthwhile.

## Launch sequence

1. **Finite risk.** A scalar noise schedule does not specify aggregate risk
   when resolution changes or when perturbations repeat over diffusion time.
2. **Feasible region.** Dimension-uniform Schatten-\(2\) control is the sharp
   asymptotic boundary under the stated spectral assumptions.
3. **Finite certificate.** Report raw \(T_2(R)\), refinement growth, shell
   mass, and distinct \(B_{\max}\), \(B_{\mathrm{path}}\), and
   \(B_{\mathrm{end}}\) budgets.
4. **Decision rule.** Allocate a prescribed budget by task sensitivity rather
   than by frequency alone.
5. **Operating curve.** Compare methods over the full quality--stability
   frontier, including the unfiltered endpoint.

## Contribution lock

Use at most two contribution bullets.

1. **Finite resolution--time certificate.** The theorem is the feasible
   region; finite raw-\(T_2\), shell, path, and end-state accounting make it
   actionable for latent diffusion.
2. **Task-aware Schatten budgeting.** A task-weighted projection and latent
   adapter allocate covariance mass under \(T_2\) and operator-norm constraints.

Adaptive probes, determinant machinery, and encoder Jacobian identities are
supporting tools, not separate contributions.

## Scope lock

- The only main application is latent diffusion.
- A structured latent adapter is the main implementation.
- Exact accounting is used whenever the covariance is structured.
- Matrix-free estimates are used only for implicit operators and must declare
  whether they are global certificates, blockwise diagnostics, or local
  training surrogates.
- The paper does not claim that one universal numerical budget is optimal for
  every task. It exposes the frontier from which an operating point is chosen.

## Main evidence shape

There are two experiment suites and at most three main figures plus one table.

1. **Controlled task-bearing counterexample.** Mixed spectra and sparse
   high-frequency task signal distinguish task-aware budgeting from low-pass,
   uniform shrinkage, and operator-norm control.
2. **One real latent-diffusion suite.** The same images, model, gates, and
   adapters produce:
   - a resolution-by-time view of \(B_{\max}\), \(B_{\mathrm{path}}\), and
     \(B_{\mathrm{end}}\);
   - frozen and adapted quality--stability frontiers;
   - a compact encoder/adapter table with raw \(T_2^E\), growth, task quality,
     and measured cost.

Constant schedules, extended mixed-spectrum families, estimator convergence,
full spectra, and secondary models belong in the appendix.

## Evidence admission rule

An item enters the main paper only if it does one of these jobs:

1. calibrates a finite \(T_2\) or temporal certificate;
2. proves that task-aware allocation is not generic low-pass filtering;
3. locates a quality--stability operating point on real latent diffusion;
4. reports the cost required to certify that point.

No experiment is admitted merely because it already exists.

## Retired evidence

- Ignore all old FNO results, figures, tables, and claims.
- Ignore all old synthetic VAE-filter results.
- Ignore all old SD-VAE Jacobian numbers and conclusions; only low-level model
  loading or JVP/VJP plumbing may be reused in a new protocol.
- Ignore old ELBO-training and diffusion-prior toy results.
- Pure power-law, determinant-identity, frequency-multiplicity, and probe
  experiments may be rebuilt only as theory or estimator sanity checks. Their
  old numbers are never presented as ML evidence.

## Nine-page budget

- Introduction: 0.75 page
- Finite certificate and theorem: 1.50 pages
- Task-aware method: 1.50 pages
- Experimental protocol: 0.50 page
- Results: 3.50 pages
- Related work and scope: 0.75 page
- Conclusion: 0.50 page

Proofs, full determinant derivations, secondary schedules, hyperparameters,
probe convergence, and provenance go to the appendix.

## Language locks

- Sampling uses an amplitude \(B\); relative covariance is \(K=BB^\ast\).
  Gates and raw \(T_2\) act on \(K\), so a diagonal amplitude \(b_j\)
  contributes \(b_j^4\) to \(T_2\).
- Say **dimension-uniform Schatten control**, not finite-matrix membership.
- Say **asymptotic feasible region**, not a universal finite failure threshold.
- Keep \(B_{\max}\), \(B_{\mathrm{path}}\), and \(B_{\mathrm{end}}\) distinct.
- Say **task-critical high-frequency modes**; never equate Schatten budgeting
  with low-pass filtering.
- Say **probe-induced gradient variance** unless minibatch or SGD randomness is
  actually varied.
- A blockwise diagnostic or localized surrogate is not a global certificate.
- Never use a normalized effective-rank quantity as a substitute for raw
  \(T_2\).

## Experimental-result insertion policy

Until final outputs are frozen, pending results appear only as LaTeX comments,
for example:

```tex
% EXPERIMENTAL RESULT PLACEHOLDER: insert verified Pareto result here.
```

Visible draft prose must not mention missing experiments, anticipated wins,
empty tables, future runs, or provisional numbers. Once results are verified,
replace the comment with one precise claim whose scope matches the evidence.
