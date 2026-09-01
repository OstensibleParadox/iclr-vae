# Pre-A100 scientific readiness

This file is the acceptance contract for connecting the final experiments to
an A100. A GPU run is not evidence unless every quantity below is the same
quantity defined in the paper.

## Canonical Gaussian contract

- Sampling is parameterized by an amplitude operator \(B\):
  \(\xi=C^{1/2}B\epsilon\), \(\epsilon\sim N(0,I)\).
- The relative covariance is \(K=BB^\ast\).
- Every gate, determinant, and reported diagnostic acts on \(K\).
- \(T_2=\operatorname{Tr}(K^\ast K)\). A diagonal amplitude \(b_j\)
  contributes \(b_j^4\), not \(b_j^2\).
- The primary deterministic objective is
  \(\mathrm{KL}(q_K\Vert p)=-\tfrac12\log\det_2(I+K)\).
- Precision experiments use \(H\) only through the explicit bridge
  \(K=H(I-H)^{-1}\), and name the sampling measure and KL direction.

## CPU gates before GPU execution

- [ ] Exact/empirical small-matrix contract tests pass for covariance, \(T_2\),
  both KL orientations, and centered log-density variance.
- [ ] The power-law phase diagram validates the \(H\leftrightarrow K\) bridge
  and reports canonical raw \(T_2(K)\).
- [ ] The probe-gradient mechanism test has at least five independent seeds,
  95% confidence intervals, and saved per-seed raw rows.
- [ ] The matched-frequency test contains explicit frequency-only and
  ideal-aware controls at a stated matched low-frequency response.
- [ ] Every generated CSV uses stable field names and LF line endings; every
  plotted aggregate can be regenerated from its raw CSV.

## Protocol gates before final A100 execution

- [ ] FNO uses physical modes, correct real-FFT multiplicity, hidden-feature
  injection, raw \(T_2(K)\), common continuum fields, and base-grid to
  \(2R/4R\) zero-shot evaluation.
- [ ] FNO evaluation is deterministic and reports gradient variance,
  noise-scale drift, discretization defect, and matched baselines.
- [ ] The encoder audit uses independent \(C_z^{\mathrm{ref}}\), input
  covariance, quadrature weights, actual VAE scaling/posterior variance, and
  reports \(K_E\) separately from \(K_{\mathrm{diff}}\).
- [ ] High-resolution encoder spectra use a matrix-free estimator with an
  exact small-resolution cross-check.
- [ ] Final configs, validators, plots, and paper captions use the same raw
  field names and the same contract version.

The a100_final config keeps scientific_contract.implementation_ready false
until all gates above are executable and pass. The final preflight treats that
flag as a hard failure.
