from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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
from wickdet import WickCarlemanPenalty, rademacher_probes  # noqa: E402


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


@dataclass(frozen=True)
class VarianceConfig:
    dimensions: tuple[int, ...] = (256, 1024, 4096, 16384)
    batch_size: int = 16
    num_probes: int = 8
    num_grad_samples: int = 32
    seed: int = 123
    output_csv: Path = EXPERIMENT_DIR / "gradient_variance_litmus.csv"
    output_png: Path = EXPERIMENT_DIR / "gradient_variance_litmus.png"


def make_variance_models(
    training_config: TrainingConfig,
    dim: int,
) -> dict[str, FFTDiagonalGaussianOperator]:
    initial_logits = make_initial_logits(training_config, dim)
    return {
        "cf_probe": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=training_config.max_beta,
        ),
        "filtered_cf_probe": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=training_config.max_beta,
            alpha_filter=training_config.alpha_filter,
        ),
        "trace_residue_minus_0_5": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=training_config.max_beta,
        ),
        "trace_residue_plus_0_5": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=training_config.max_beta,
        ),
        "trace_residue_plus_1_0": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=training_config.max_beta,
        ),
    }


def matrix_free_prior_estimator(
    branch: str,
    model: FFTDiagonalGaussianOperator,
    features: torch.Tensor,
    probes: torch.Tensor,
    cf_estimator: WickCarlemanPenalty,
) -> tuple[torch.Tensor, torch.Tensor]:
    value, diagnostics = cf_estimator(
        model,
        features,
        probes=probes,
        return_diagnostics=True,
    )
    trace_est = diagnostics["trace"]
    return value + TRACE_RESIDUE_COEFFS[branch] * trace_est, trace_est


def run_gradient_variance_litmus(
    config: VarianceConfig = VarianceConfig(),
    training_config: TrainingConfig = TrainingConfig(),
) -> list[dict[str, float | int | str]]:
    torch.manual_seed(config.seed)
    rows: list[dict[str, float | int | str]] = []
    cf_estimator = WickCarlemanPenalty(K=5, num_probes=config.num_probes)

    for dim in config.dimensions:
        print(f"\nGradient-variance litmus at dimension {dim}")
        models = make_variance_models(training_config, dim)
        features = torch.randn(config.batch_size, dim)

        gradients: dict[str, list[torch.Tensor]] = {branch: [] for branch in VARIANCE_BRANCHES}
        losses: dict[str, list[float]] = {branch: [] for branch in VARIANCE_BRANCHES}
        trace_estimates: dict[str, list[float]] = {branch: [] for branch in VARIANCE_BRANCHES}

        for _ in range(config.num_grad_samples):
            probes = rademacher_probes(features, config.num_probes)
            for branch in VARIANCE_BRANCHES:
                model = models[branch]
                model.zero_grad()
                loss, trace_est = matrix_free_prior_estimator(
                    branch,
                    model,
                    features,
                    probes,
                    cf_estimator,
                )
                loss.backward()
                gradients[branch].append(model.logits.grad.detach().clone())
                losses[branch].append(float(loss.detach()))
                trace_estimates[branch].append(float(trace_est.detach()))

        for branch in VARIANCE_BRANCHES:
            grad_stack = torch.stack(gradients[branch])
            grad_var = grad_stack.var(dim=0, unbiased=False).sum()
            grad_norm_mean = grad_stack.norm(dim=1).mean()
            loss_tensor = torch.tensor(losses[branch])
            trace_tensor = torch.tensor(trace_estimates[branch])
            model = models[branch]
            rows.append(
                {
                    "dimension": dim,
                    "branch": branch,
                    "trace_residue_coeff": TRACE_RESIDUE_COEFFS[branch],
                    "total_grad_variance": float(grad_var),
                    "mean_grad_norm": float(grad_norm_mean),
                    "mean_estimated_prior": float(loss_tensor.mean()),
                    "std_estimated_prior": float(loss_tensor.std(unbiased=False)),
                    "mean_trace_estimate": float(trace_tensor.mean()),
                    "std_trace_estimate": float(trace_tensor.std(unbiased=False)),
                    "trace_exact": float(model.trace_exact().detach()),
                    "t2_exact": float(model.t2_exact().detach()),
                }
            )
            print(
                f"  {branch:<26}"
                f" grad_var={float(grad_var):10.4e}"
                f" grad_norm={float(grad_norm_mean):10.4e}"
                f" T2={float(model.t2_exact().detach()):10.4e}"
            )

    return rows


def save_results(rows: list[dict[str, float | int | str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, float | int | str]], png_path: Path) -> None:
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
        "trace_residue_minus_0_5": "CF - 0.5 Tr",
        "trace_residue_plus_0_5": "CF + 0.5 Tr",
        "trace_residue_plus_1_0": "CF + 1.0 Tr",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for branch in VARIANCE_BRANCHES:
        branch_rows = [row for row in rows if row["branch"] == branch]
        branch_rows.sort(key=lambda row: int(row["dimension"]))
        dims = [int(row["dimension"]) for row in branch_rows]
        grad_var = [float(row["total_grad_variance"]) for row in branch_rows]

        axes[0].plot(dims, grad_var, "o-", color=colors[branch], label=labels[branch], linewidth=1.8)

    for branch, label in (
        ("cf_probe", "unfiltered branches"),
        ("filtered_cf_probe", "filtered CF branch"),
    ):
        branch_rows = [row for row in rows if row["branch"] == branch]
        branch_rows.sort(key=lambda row: int(row["dimension"]))
        dims = [int(row["dimension"]) for row in branch_rows]
        t2 = [float(row["t2_exact"]) for row in branch_rows]
        axes[1].plot(dims, t2, "o-", color=colors[branch], label=label, linewidth=2.2)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Matrix-Free Gradient Variance")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Sum of parameterwise variances")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("T2 Diagnostic")
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("T2 = ||P||_S2^2")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)

    fig.suptitle("Gradient-Variance Litmus for Matrix-Free Trace Residues", y=0.995)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def validate_results(rows: list[dict[str, float | int | str]]) -> None:
    dimensions = sorted({int(row["dimension"]) for row in rows})
    for dim in dimensions:
        dim_rows = [row for row in rows if int(row["dimension"]) == dim]
        branches = {row["branch"] for row in dim_rows}
        missing = set(VARIANCE_BRANCHES) - branches
        if missing:
            raise RuntimeError(f"dimension {dim} missing branches: {sorted(missing)}")
        cf_t2 = next(float(row["t2_exact"]) for row in dim_rows if row["branch"] == "cf_probe")
        filtered_t2 = next(float(row["t2_exact"]) for row in dim_rows if row["branch"] == "filtered_cf_probe")
        if filtered_t2 >= cf_t2:
            raise RuntimeError(f"dimension {dim} filtered branch did not reduce T2")


if __name__ == "__main__":
    config = VarianceConfig()
    result_rows = run_gradient_variance_litmus(config)
    validate_results(result_rows)
    save_results(result_rows, config.output_csv)
    plot_results(result_rows, config.output_png)
    print(f"\nWrote {config.output_csv}")
    print(f"Wrote {config.output_png}")
