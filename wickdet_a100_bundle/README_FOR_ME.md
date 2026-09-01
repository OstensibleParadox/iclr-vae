# WickDet A100 Bundle Notes

> **Scientific readiness guard:** final A100 execution is intentionally blocked
> while `scientific_contract.implementation_ready` is false.  The canonical
> contract is amplitude $B$ for sampling, relative covariance $K=BB^*$ for all
> gates and objectives, and $T_2=\operatorname{Tr}(K^*K)$.  Do not bypass this
> guard: it is lifted only after the CPU contract tests, toy claim tests, and
> FNO/SD-VAE protocol validators agree with the paper.

This is the black-box A100/CUDA final-run package for the WickDet / Schatten-filtered noise experiments. It is intentionally separate from local `experiments_a100/` development scaffolding and from archived M4/MPS pilot runs.

Pilot outputs are historical development records only. They must not be copied into this bundle's `outputs/<RUN_ID>/figures/`, `csv/`, or final manifests.

The infra operator should run only the shell wrappers. Scientific interpretation remains outside the infra handoff.

Important evidence policy:

- Final figures must come from `outputs/<RUN_ID>/` produced by A100/CUDA runs.
- FNO main Pareto uses normalized T2, not raw T2 as the only x-axis.
- SD-VAE power-law fits use covariance eigenvalues `lambda_i` from `G = J J^T`; the S2-admissible tail threshold is `alpha > 1/2`.
- The 512 SD-VAE stage is separate from the 64/128/256 stages and should not be silently merged into one evidence class.

Bundled model cache:

- `model_cache/hf_cache/` contains the HuggingFace cache for `stabilityai/sd-vae-ft-mse`.
- `model_cache/MODEL_CACHE_MANIFEST.csv` records file sizes and sha256 hashes.
- `model_cache/MODEL_CACHE_SUMMARY.json` records total cache size and file count.
