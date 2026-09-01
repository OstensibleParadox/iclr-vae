from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gaussian-hilbert-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gaussian-hilbert-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOY_DIR = Path(__file__).resolve().parent
if str(TOY_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_DIR))
EXPERIMENT_DIR = Path(__file__).resolve().parent

from training_elbo_toy import (  # noqa: E402
    FFTDiagonalGaussianOperator,
    TrainingConfig,
    make_initial_logits,
)
from wickdet import (  # noqa: E402
    matrix_free_precision_wick_log_q_over_p,
    rademacher_probes,
)


VARIANCE_BRANCHES = (
    "cf_probe",
    "filtered_cf_probe",
    "trace_residue_minus_0_5",
    "trace_residue_plus_0_5",
    "trace_residue_plus_1_0",
)

TRACE_RESIDUE_COEFFS = {
    "cf_probe": 0.0,
    "filtered_cf_probe": 0.0,
    "trace_residue_minus_0_5": -0.5,
    "trace_residue_plus_0_5": 0.5,
    "trace_residue_plus_1_0": 1.0,
}

AGGREGATED_METRICS = (
    "total_grad_variance",
    "mean_grad_norm",
    "mean_estimated_prior",
    "std_estimated_prior",
    "mean_trace_estimate",
    "std_trace_estimate",
    "trace_exact",
    "t2_exact",
)


@dataclass(frozen=True)
class VarianceConfig:
    """Configuration for the fixed-batch, probe-induced variance test.

    ``num_seeds`` is the outer replication count. Each outer replicate has an
    independently seeded feature batch and a small paired perturbation of the
    initial model spectrum. Within a replicate, the batch and model parameters
    stay fixed while only Rademacher probes change ``num_grad_samples`` times.
    Consequently, ``total_grad_variance`` is specifically probe-induced
    gradient variance; it is not an estimate of general minibatch/SGD variance.
    """

    dimensions: tuple[int, ...] = (256, 1024, 4096, 16384)
    batch_size: int = 16
    num_probes: int = 8
    num_grad_samples: int = 32
    series_order: int = 5
    num_seeds: int = 5
    num_threads: int = 4
    base_seed: int = 123
    model_logit_jitter: float = 0.02
    bootstrap_samples: int = 2000
    output_csv: Path = EXPERIMENT_DIR / "gradient_variance_litmus.csv"
    output_summary_csv: Path = EXPERIMENT_DIR / "gradient_variance_litmus_summary.csv"
    output_png: Path = EXPERIMENT_DIR / "gradient_variance_litmus.png"


def _validate_config(config: VarianceConfig) -> None:
    if not config.dimensions or any(dim <= 1 for dim in config.dimensions):
        raise ValueError("dimensions must contain positive values greater than one")
    if len(set(config.dimensions)) != len(config.dimensions):
        raise ValueError("dimensions must be unique")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.num_probes <= 0:
        raise ValueError("num_probes must be positive")
    if config.num_grad_samples < 2:
        raise ValueError("num_grad_samples must be at least two to estimate variance")
    if config.series_order < 2:
        raise ValueError("series_order must be at least two")
    if config.num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    if config.num_threads <= 0:
        raise ValueError("num_threads must be positive")
    if config.model_logit_jitter < 0.0:
        raise ValueError("model_logit_jitter must be nonnegative")
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")


def _derived_seed(base_seed: int, seed_index: int, dim: int, stream: int) -> int:
    """Return stable, separated seeds for model, features, probes, and CIs."""

    modulus = 2**63 - 1
    value = (
        int(base_seed)
        + 1_000_003 * int(seed_index + 1)
        + 10_007 * int(dim)
        + 97_003 * int(stream + 1)
    )
    return value % modulus


def make_variance_models(
    training_config: TrainingConfig,
    dim: int,
    *,
    model_seed: int,
    model_logit_jitter: float,
) -> dict[str, FFTDiagonalGaussianOperator]:
    """Create paired branches from one seed-specific initial spectrum."""

    initial_logits = make_initial_logits(training_config, dim)
    if model_logit_jitter > 0.0:
        generator = torch.Generator(device=initial_logits.device)
        generator.manual_seed(model_seed)
        initial_logits = initial_logits + model_logit_jitter * torch.randn(
            initial_logits.shape,
            generator=generator,
            device=initial_logits.device,
            dtype=initial_logits.dtype,
        )

    return {
        "cf_probe": FFTDiagonalGaussianOperator(
            dim, initial_logits, max_beta=training_config.max_beta
        ),
        "filtered_cf_probe": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=training_config.max_beta,
            alpha_filter=training_config.alpha_filter,
        ),
        "trace_residue_minus_0_5": FFTDiagonalGaussianOperator(
            dim, initial_logits, max_beta=training_config.max_beta
        ),
        "trace_residue_plus_0_5": FFTDiagonalGaussianOperator(
            dim, initial_logits, max_beta=training_config.max_beta
        ),
        "trace_residue_plus_1_0": FFTDiagonalGaussianOperator(
            dim, initial_logits, max_beta=training_config.max_beta
        ),
    }


def matrix_free_prior_estimator(
    branch: str,
    model: FFTDiagonalGaussianOperator,
    features: torch.Tensor,
    probes: torch.Tensor,
    *,
    series_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ``model`` applies the relative precision perturbation H. Use the explicit
    # precision-Wick API, rather than the legacy wrapper whose operator and
    # objective direction were historically ambiguous. The suppressed PyTorch
    # warning comes only from the core's detached-in-spirit scalar radius
    # diagnostic; it does not alter the differentiable estimator used here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Converting a tensor with requires_grad=True to a scalar.*",
            category=UserWarning,
        )
        value, diagnostics = matrix_free_precision_wick_log_q_over_p(
            model,
            features,
            series_order=series_order,
            num_probes=probes.shape[0],
            probes=probes,
            return_diagnostics=True,
        )
    trace_est = diagnostics["precision_trace"]
    return value + TRACE_RESIDUE_COEFFS[branch] * trace_est, trace_est


def _run_one_seed_dimension(
    config: VarianceConfig,
    training_config: TrainingConfig,
    *,
    dim: int,
    seed_index: int,
) -> list[dict[str, float | int | str]]:
    model_seed = _derived_seed(config.base_seed, seed_index, dim, stream=0)
    feature_seed = _derived_seed(config.base_seed, seed_index, dim, stream=1)
    probe_seed = _derived_seed(config.base_seed, seed_index, dim, stream=2)

    models = make_variance_models(
        training_config,
        dim,
        model_seed=model_seed,
        model_logit_jitter=config.model_logit_jitter,
    )
    feature_generator = torch.Generator().manual_seed(feature_seed)
    features = torch.randn(
        config.batch_size,
        dim,
        generator=feature_generator,
        dtype=torch.float32,
    )
    probe_generator = torch.Generator().manual_seed(probe_seed)

    gradients: dict[str, list[torch.Tensor]] = {
        branch: [] for branch in VARIANCE_BRANCHES
    }
    losses: dict[str, list[float]] = {branch: [] for branch in VARIANCE_BRANCHES}
    trace_estimates: dict[str, list[float]] = {
        branch: [] for branch in VARIANCE_BRANCHES
    }

    # All branches see the same probe draw at a given inner repetition. This
    # pairing removes probe-sequence differences from branch comparisons.
    for _ in range(config.num_grad_samples):
        probes = rademacher_probes(
            features,
            config.num_probes,
            generator=probe_generator,
        )
        for branch in VARIANCE_BRANCHES:
            model = models[branch]
            model.zero_grad(set_to_none=True)
            loss, trace_est = matrix_free_prior_estimator(
                branch,
                model,
                features,
                probes,
                series_order=config.series_order,
            )
            loss.backward()
            if model.logits.grad is None:
                raise RuntimeError(f"missing logits gradient for branch={branch}")
            gradients[branch].append(model.logits.grad.detach().clone())
            losses[branch].append(float(loss.detach()))
            trace_estimates[branch].append(float(trace_est.detach()))

    rows: list[dict[str, float | int | str]] = []
    for branch in VARIANCE_BRANCHES:
        grad_stack = torch.stack(gradients[branch])
        grad_var = grad_stack.var(dim=0, unbiased=True).sum()
        grad_norm_mean = grad_stack.norm(dim=1).mean()
        loss_tensor = torch.tensor(losses[branch], dtype=torch.float64)
        trace_tensor = torch.tensor(trace_estimates[branch], dtype=torch.float64)
        model = models[branch]
        rows.append(
            {
                "seed_index": seed_index,
                "model_seed": model_seed,
                "feature_seed": feature_seed,
                "probe_seed": probe_seed,
                "dimension": dim,
                "branch": branch,
                "trace_residue_coeff": TRACE_RESIDUE_COEFFS[branch],
                "batch_size": config.batch_size,
                "num_probes": config.num_probes,
                "num_grad_samples": config.num_grad_samples,
                "series_order": config.series_order,
                "num_threads": config.num_threads,
                "model_logit_jitter": config.model_logit_jitter,
                "variance_source": "fixed_batch_rademacher_probes",
                "operator_semantics": "relative_precision_H",
                "objective_semantics": "wick_log_q_over_p",
                "total_grad_variance": float(grad_var),
                "mean_grad_norm": float(grad_norm_mean),
                "mean_estimated_prior": float(loss_tensor.mean()),
                "std_estimated_prior": float(loss_tensor.std(unbiased=True)),
                "mean_trace_estimate": float(trace_tensor.mean()),
                "std_trace_estimate": float(trace_tensor.std(unbiased=True)),
                "trace_exact": float(model.trace_exact().detach()),
                "t2_exact": float(model.t2_exact().detach()),
            }
        )
    return rows


def run_gradient_variance_litmus(
    config: VarianceConfig = VarianceConfig(),
    training_config: TrainingConfig = TrainingConfig(),
) -> list[dict[str, float | int | str]]:
    _validate_config(config)
    torch.set_num_threads(config.num_threads)
    rows: list[dict[str, float | int | str]] = []

    for dim in sorted(config.dimensions):
        print(f"\nProbe-induced gradient variance at dimension {dim}")
        for seed_index in range(config.num_seeds):
            seed_rows = _run_one_seed_dimension(
                config,
                training_config,
                dim=dim,
                seed_index=seed_index,
            )
            rows.extend(seed_rows)
            cf_row = next(row for row in seed_rows if row["branch"] == "cf_probe")
            filtered_row = next(
                row for row in seed_rows if row["branch"] == "filtered_cf_probe"
            )
            print(
                f"  seed {seed_index + 1:>2}/{config.num_seeds}: "
                f"unfiltered var={float(cf_row['total_grad_variance']):.4e}, "
                f"filtered var={float(filtered_row['total_grad_variance']):.4e}"
            )
    return rows


def _bootstrap_mean_ci(
    values: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    values = values.detach().to(torch.float64).flatten()
    if values.numel() == 0:
        raise ValueError("cannot aggregate an empty tensor")
    mean = float(values.mean())
    if values.numel() == 1:
        return mean, mean, mean
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0,
        values.numel(),
        (samples, values.numel()),
        generator=generator,
    )
    boot_means = values[indices].mean(dim=1)
    low, high = torch.quantile(
        boot_means,
        torch.tensor([0.025, 0.975], dtype=torch.float64),
    )
    return mean, float(low), float(high)


def _log_log_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = torch.log(torch.tensor(tuple(xs), dtype=torch.float64))
    y = torch.log(torch.tensor(tuple(ys), dtype=torch.float64))
    if x.numel() < 2 or y.numel() != x.numel():
        return float("nan")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return float("nan")
    x_centered = x - x.mean()
    denominator = x_centered.square().sum()
    if float(denominator) <= 0.0:
        return float("nan")
    return float((x_centered * (y - y.mean())).sum() / denominator)


def _branch_growth_summary(
    rows: list[dict[str, float | int | str]],
    branch: str,
    metric: str,
    config: VarianceConfig,
) -> tuple[float, float, float]:
    slopes = []
    for seed_index in range(config.num_seeds):
        selected = sorted(
            (
                row
                for row in rows
                if row["branch"] == branch
                and int(row["seed_index"]) == seed_index
            ),
            key=lambda row: int(row["dimension"]),
        )
        slope = _log_log_slope(
            (float(row["dimension"]) for row in selected),
            (float(row[metric]) for row in selected),
        )
        if math.isfinite(slope):
            slopes.append(slope)
    if not slopes:
        return float("nan"), float("nan"), float("nan")
    branch_index = VARIANCE_BRANCHES.index(branch)
    metric_index = AGGREGATED_METRICS.index(metric)
    return _bootstrap_mean_ci(
        torch.tensor(slopes, dtype=torch.float64),
        samples=config.bootstrap_samples,
        seed=_derived_seed(
            config.base_seed,
            branch_index,
            max(config.dimensions),
            20 + metric_index,
        ),
    )


def aggregate_results(
    rows: list[dict[str, float | int | str]],
    config: VarianceConfig,
) -> list[dict[str, float | int | str]]:
    """Aggregate outer seeds and attach paired log-log growth exponents."""

    growth: dict[str, dict[str, tuple[float, float, float]]] = {}
    for branch in VARIANCE_BRANCHES:
        growth[branch] = {
            "total_grad_variance": _branch_growth_summary(
                rows, branch, "total_grad_variance", config
            ),
            "t2_exact": _branch_growth_summary(rows, branch, "t2_exact", config),
        }

    summary_rows: list[dict[str, float | int | str]] = []
    for dim in sorted(config.dimensions):
        for branch_index, branch in enumerate(VARIANCE_BRANCHES):
            selected = [
                row
                for row in rows
                if int(row["dimension"]) == dim and row["branch"] == branch
            ]
            summary: dict[str, float | int | str] = {
                "dimension": dim,
                "branch": branch,
                "trace_residue_coeff": TRACE_RESIDUE_COEFFS[branch],
                "num_seeds": len(selected),
                "batch_size": config.batch_size,
                "num_probes": config.num_probes,
                "num_grad_samples": config.num_grad_samples,
                "series_order": config.series_order,
                "num_threads": config.num_threads,
                "variance_source": "fixed_batch_rademacher_probes",
                "operator_semantics": "relative_precision_H",
                "objective_semantics": "wick_log_q_over_p",
                "ci_method": "percentile_bootstrap_over_outer_seeds",
                "bootstrap_samples": config.bootstrap_samples,
                "growth_exponent_fit": "paired_seed_ols_log_metric_on_log_dimension",
            }
            for metric_index, metric in enumerate(AGGREGATED_METRICS):
                values = torch.tensor(
                    [float(row[metric]) for row in selected], dtype=torch.float64
                )
                mean, low, high = _bootstrap_mean_ci(
                    values,
                    samples=config.bootstrap_samples,
                    seed=_derived_seed(
                        config.base_seed,
                        branch_index,
                        dim,
                        40 + metric_index,
                    ),
                )
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_ci_low"] = low
                summary[f"{metric}_ci_high"] = high

            grad_slope = growth[branch]["total_grad_variance"]
            t2_slope = growth[branch]["t2_exact"]
            summary.update(
                {
                    "grad_variance_dim_exponent_mean": grad_slope[0],
                    "grad_variance_dim_exponent_ci_low": grad_slope[1],
                    "grad_variance_dim_exponent_ci_high": grad_slope[2],
                    "t2_dim_exponent_mean": t2_slope[0],
                    "t2_dim_exponent_ci_low": t2_slope[1],
                    "t2_dim_exponent_ci_high": t2_slope[2],
                }
            )
            summary_rows.append(summary)
    return summary_rows


def save_results(
    rows: list[dict[str, float | int | str]],
    csv_path: Path,
) -> None:
    if not rows:
        raise ValueError("cannot save empty results")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _error_bar_values(
    branch_rows: list[dict[str, float | int | str]],
    metric: str,
) -> tuple[list[int], list[float], list[list[float]]]:
    branch_rows = sorted(branch_rows, key=lambda row: int(row["dimension"]))
    dimensions = [int(row["dimension"]) for row in branch_rows]
    means = [float(row[f"{metric}_mean"]) for row in branch_rows]
    lows = [float(row[f"{metric}_ci_low"]) for row in branch_rows]
    highs = [float(row[f"{metric}_ci_high"]) for row in branch_rows]
    lower_errors = [max(mean - low, 0.0) for mean, low in zip(means, lows)]
    upper_errors = [max(high - mean, 0.0) for mean, high in zip(means, highs)]
    return dimensions, means, [lower_errors, upper_errors]


def _exponent_label(
    row: dict[str, float | int | str],
    prefix: str,
) -> str:
    mean = float(row[f"{prefix}_mean"])
    low = float(row[f"{prefix}_ci_low"])
    high = float(row[f"{prefix}_ci_high"])
    return f"b={mean:.2f} [{low:.2f}, {high:.2f}]"


def plot_results(
    summary_rows: list[dict[str, float | int | str]],
    png_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {
        "cf_probe": "#238b45",
        "filtered_cf_probe": "#7b3294",
        "trace_residue_minus_0_5": "#8c6d31",
        "trace_residue_plus_0_5": "#e6550d",
        "trace_residue_plus_1_0": "#d33f49",
    }
    labels = {
        "cf_probe": "CF probe",
        "filtered_cf_probe": "filtered CF probe",
        "trace_residue_minus_0_5": r"CF $-0.5\,\mathrm{Tr}$",
        "trace_residue_plus_0_5": r"CF $+0.5\,\mathrm{Tr}$",
        "trace_residue_plus_1_0": r"CF $+1.0\,\mathrm{Tr}$",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for branch in VARIANCE_BRANCHES:
        branch_rows = [row for row in summary_rows if row["branch"] == branch]
        dimensions, means, errors = _error_bar_values(
            branch_rows, "total_grad_variance"
        )
        exponent = _exponent_label(
            branch_rows[0], "grad_variance_dim_exponent"
        )
        axes[0].errorbar(
            dimensions,
            means,
            yerr=errors,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[branch],
            label=f"{labels[branch]} ({exponent})",
        )

    for branch, label in (
        ("cf_probe", "unfiltered"),
        ("filtered_cf_probe", "Schatten-filtered"),
    ):
        branch_rows = [row for row in summary_rows if row["branch"] == branch]
        dimensions, means, errors = _error_bar_values(branch_rows, "t2_exact")
        exponent = _exponent_label(branch_rows[0], "t2_dim_exponent")
        axes[1].errorbar(
            dimensions,
            means,
            yerr=errors,
            marker="o",
            linewidth=2.2,
            capsize=3,
            color=colors[branch],
            label=f"{label} ({exponent})",
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Probe-induced gradient variance")
    axes[0].set_xlabel("Feature dimension")
    axes[0].set_ylabel("Trace of gradient covariance")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Relative-precision Hilbert–Schmidt diagnostic")
    axes[1].set_xlabel("Feature dimension")
    axes[1].set_ylabel(r"$\|H\|_{\mathcal{S}_2}^2$")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=7.5)
    fig.suptitle(
        "Fixed batch and model; only Rademacher probes vary (95% CI across outer seeds)",
        y=0.995,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def validate_results(
    rows: list[dict[str, float | int | str]],
    summary_rows: list[dict[str, float | int | str]],
    config: VarianceConfig,
) -> None:
    expected_raw = len(config.dimensions) * len(VARIANCE_BRANCHES) * config.num_seeds
    if len(rows) != expected_raw:
        raise RuntimeError(f"expected {expected_raw} raw rows, found {len(rows)}")
    expected_summary = len(config.dimensions) * len(VARIANCE_BRANCHES)
    if len(summary_rows) != expected_summary:
        raise RuntimeError(
            f"expected {expected_summary} summary rows, found {len(summary_rows)}"
        )

    keys = {
        (int(row["dimension"]), str(row["branch"]), int(row["seed_index"]))
        for row in rows
    }
    if len(keys) != expected_raw:
        raise RuntimeError("duplicate or missing (dimension, branch, seed) raw rows")

    for row in rows:
        for metric in ("total_grad_variance", "mean_grad_norm", "t2_exact"):
            value = float(row[metric])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(f"invalid {metric}={value} in row={row}")

    for dim in config.dimensions:
        for seed_index in range(config.num_seeds):
            selected = [
                row
                for row in rows
                if int(row["dimension"]) == dim
                and int(row["seed_index"]) == seed_index
            ]
            branches = {str(row["branch"]) for row in selected}
            if branches != set(VARIANCE_BRANCHES):
                raise RuntimeError(
                    f"dimension={dim}, seed={seed_index} has branches={sorted(branches)}"
                )
            cf_t2 = next(
                float(row["t2_exact"])
                for row in selected
                if row["branch"] == "cf_probe"
            )
            filtered_t2 = next(
                float(row["t2_exact"])
                for row in selected
                if row["branch"] == "filtered_cf_probe"
            )
            if filtered_t2 >= cf_t2:
                raise RuntimeError(
                    f"dimension={dim}, seed={seed_index}: filtered T2 did not decrease"
                )

    if len(config.dimensions) >= 2:
        first_dim = min(config.dimensions)
        cf_summary = next(
            row
            for row in summary_rows
            if int(row["dimension"]) == first_dim and row["branch"] == "cf_probe"
        )
        filtered_summary = next(
            row
            for row in summary_rows
            if int(row["dimension"]) == first_dim
            and row["branch"] == "filtered_cf_probe"
        )
        if float(filtered_summary["t2_dim_exponent_mean"]) >= float(
            cf_summary["t2_dim_exponent_mean"]
        ):
            raise RuntimeError("filtered T2 growth exponent did not improve")


def parse_dimensions(raw: str) -> tuple[int, ...]:
    dimensions = tuple(
        sorted({int(token.strip()) for token in raw.split(",") if token.strip()})
    )
    if not dimensions:
        raise argparse.ArgumentTypeError("at least one dimension is required")
    return dimensions


def parse_args() -> argparse.Namespace:
    defaults = VarianceConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Measure fixed-batch, Rademacher-probe-induced gradient variance "
            "over independent feature/model seeds."
        )
    )
    parser.add_argument("--dimensions", default=",".join(map(str, defaults.dimensions)))
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-probes", type=int, default=defaults.num_probes)
    parser.add_argument("--num-grad-samples", type=int, default=defaults.num_grad_samples)
    parser.add_argument("--series-order", type=int, default=defaults.series_order)
    parser.add_argument("--num-seeds", type=int, default=defaults.num_seeds)
    parser.add_argument("--num-threads", type=int, default=defaults.num_threads)
    parser.add_argument("--base-seed", type=int, default=defaults.base_seed)
    parser.add_argument(
        "--model-logit-jitter", type=float, default=defaults.model_logit_jitter
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=defaults.bootstrap_samples
    )
    parser.add_argument("--output-csv", type=Path, default=defaults.output_csv)
    parser.add_argument(
        "--output-summary-csv", type=Path, default=defaults.output_summary_csv
    )
    parser.add_argument("--output-png", type=Path, default=defaults.output_png)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a two-seed, small-dimension configuration for a fast CPU check.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> VarianceConfig:
    config = VarianceConfig(
        dimensions=parse_dimensions(args.dimensions),
        batch_size=args.batch_size,
        num_probes=args.num_probes,
        num_grad_samples=args.num_grad_samples,
        series_order=args.series_order,
        num_seeds=args.num_seeds,
        num_threads=args.num_threads,
        base_seed=args.base_seed,
        model_logit_jitter=args.model_logit_jitter,
        bootstrap_samples=args.bootstrap_samples,
        output_csv=args.output_csv,
        output_summary_csv=args.output_summary_csv,
        output_png=args.output_png,
    )
    if args.smoke:
        config = replace(
            config,
            dimensions=(64, 256),
            batch_size=min(config.batch_size, 4),
            num_probes=min(config.num_probes, 2),
            num_grad_samples=min(config.num_grad_samples, 4),
            num_seeds=min(config.num_seeds, 2),
            num_threads=min(config.num_threads, 2),
            bootstrap_samples=min(config.bootstrap_samples, 200),
        )
    return config


def main() -> None:
    config = config_from_args(parse_args())
    start = time.perf_counter()
    result_rows = run_gradient_variance_litmus(config)
    summary_rows = aggregate_results(result_rows, config)
    validate_results(result_rows, summary_rows, config)
    save_results(result_rows, config.output_csv)
    save_results(summary_rows, config.output_summary_csv)
    plot_results(summary_rows, config.output_png)
    elapsed = time.perf_counter() - start
    print(f"\nWrote raw seed results: {config.output_csv}")
    print(f"Wrote aggregate summary: {config.output_summary_csv}")
    print(f"Wrote figure: {config.output_png}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
