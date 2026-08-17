from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gaussian-hilbert-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gaussian-hilbert-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_DIR = Path(__file__).resolve().parent

from wickdet import (  # noqa: E402
    WickCarlemanPenalty,
    hutchinson_trace,
    neg_logdet2_series,
    rademacher_probes,
)


BRANCHES = (
    "ordinary_branch",
    "cf_branch",
    "schatten_filtered_cf_branch",
    "trace_residue_minus_0_5",
    "trace_residue_plus_0_5",
    "trace_residue_plus_1_0",
)

TRACE_RESIDUE_COEFFS = {
    "trace_residue_minus_0_5": -0.5,
    "trace_residue_plus_0_5": 0.5,
    "trace_residue_plus_1_0": 1.0,
}


@dataclass(frozen=True)
class TrainingConfig:
    dimensions: tuple[int, ...] = (256, 1024, 4096, 16384)
    epochs: int = 80
    batch_size: int = 16
    num_probes: int = 8
    seed: int = 42
    lr: float = 3e-2
    max_beta: float = 0.95
    init_scale: float = 0.45
    init_alpha: float = 0.05
    target_scale: float = 0.72
    target_alpha: float = 0.10
    alpha_filter: float = 0.60
    mse_weight: float = 1.0
    kl_weight: float = 1e-2
    output_csv: Path = EXPERIMENT_DIR / "training_elbo_toy_results.csv"
    output_png: Path = EXPERIMENT_DIR / "training_elbo_toy_results.png"
    output_pareto_png: Path = EXPERIMENT_DIR / "training_elbo_toy_pareto.png"


def rfft_multiplicities(dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    freq_dim = dim // 2 + 1
    mult = torch.full((freq_dim,), 2.0, device=device, dtype=dtype)
    mult[0] = 1.0
    if dim % 2 == 0:
        mult[-1] = 1.0
    return mult


def decaying_spectrum(
    dim: int,
    *,
    scale: float,
    alpha: float,
    max_beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    freq_dim = dim // 2 + 1
    j = torch.arange(1, freq_dim + 1, device=device, dtype=dtype)
    return torch.clamp(scale / j.pow(alpha), max=max_beta * 0.98)


class FFTDiagonalGaussianOperator(nn.Module):
    def __init__(
        self,
        dim: int,
        initial_logits: torch.Tensor,
        *,
        max_beta: float,
        alpha_filter: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_beta = max_beta
        self.logits = nn.Parameter(initial_logits.clone())

        dtype = initial_logits.dtype
        device = initial_logits.device
        multiplicities = rfft_multiplicities(dim, device=device, dtype=dtype)
        self.register_buffer("multiplicities", multiplicities)

        if alpha_filter is None:
            filter_mask = torch.ones_like(initial_logits)
        else:
            j = torch.arange(1, initial_logits.numel() + 1, device=device, dtype=dtype)
            filter_mask = j.pow(-alpha_filter)
        self.register_buffer("filter_mask", filter_mask)

    def eigenvalues(self) -> torch.Tensor:
        return self.max_beta * torch.sigmoid(self.logits) * self.filter_mask

    def _exact_eigenvalues(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.eigenvalues().to(torch.float64), self.multiplicities.to(torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_freq = torch.fft.rfft(x, dim=-1)
        y_freq = x_freq * self.eigenvalues().to(x_freq.dtype).unsqueeze(0)
        return torch.fft.irfft(y_freq, n=self.dim, dim=-1)

    def trace_exact(self) -> torch.Tensor:
        beta, multiplicities = self._exact_eigenvalues()
        return (multiplicities * beta).sum()

    def t2_exact(self) -> torch.Tensor:
        beta, multiplicities = self._exact_eigenvalues()
        return (multiplicities * beta.square()).sum()

    def logdet_exact(self) -> torch.Tensor:
        beta, multiplicities = self._exact_eigenvalues()
        return (multiplicities * torch.log1p(-beta)).sum()

    def neg_logdet2_exact(self) -> torch.Tensor:
        return -(self.logdet_exact() + self.trace_exact())

    def ordinary_prior_kl_exact(self) -> torch.Tensor:
        beta, multiplicities = self._exact_eigenvalues()
        trace_excess = (multiplicities * beta / (1.0 - beta)).sum()
        return 0.5 * trace_excess + 0.5 * self.logdet_exact()

    def cf_prior_kl_exact(self) -> torch.Tensor:
        beta, multiplicities = self._exact_eigenvalues()
        wick_mean = (multiplicities * beta.square() / (1.0 - beta)).sum()
        logdet2 = self.logdet_exact() + self.trace_exact()
        return 0.5 * wick_mean + 0.5 * logdet2

    def sample_log_rn_ordinary(self, features: torch.Tensor) -> torch.Tensor:
        px = self(features)
        quadratic = (features * px).sum(dim=-1).mean()
        return 0.5 * quadratic + 0.5 * self.logdet_exact()

    def sample_log_rn_cf(self, features: torch.Tensor) -> torch.Tensor:
        px = self(features)
        quadratic = (features * px).sum(dim=-1).mean()
        logdet2 = self.logdet_exact() + self.trace_exact()
        return 0.5 * (quadratic - self.trace_exact()) + 0.5 * logdet2


class FixedFFTDiagonalOperator(nn.Module):
    def __init__(self, dim: int, eigenvalues: torch.Tensor) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("target_eigenvalues", eigenvalues.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_freq = torch.fft.rfft(x, dim=-1)
        y_freq = x_freq * self.target_eigenvalues.to(x_freq.dtype).unsqueeze(0)
        return torch.fft.irfft(y_freq, n=self.dim, dim=-1)


def make_initial_logits(config: TrainingConfig, dim: int) -> torch.Tensor:
    initial_beta = decaying_spectrum(
        dim,
        scale=config.init_scale,
        alpha=config.init_alpha,
        max_beta=config.max_beta,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    normalized = torch.clamp(initial_beta / config.max_beta, 1e-4, 1.0 - 1e-4)
    return torch.logit(normalized)


def make_target_operator(config: TrainingConfig, dim: int) -> FixedFFTDiagonalOperator:
    target_beta = decaying_spectrum(
        dim,
        scale=config.target_scale,
        alpha=config.target_alpha,
        max_beta=config.max_beta,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return FixedFFTDiagonalOperator(dim, target_beta)


def make_models(
    config: TrainingConfig,
    dim: int,
) -> dict[str, FFTDiagonalGaussianOperator]:
    initial_logits = make_initial_logits(config, dim)
    return {
        "ordinary_branch": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=config.max_beta,
        ),
        "cf_branch": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=config.max_beta,
        ),
        "schatten_filtered_cf_branch": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=config.max_beta,
            alpha_filter=config.alpha_filter,
        ),
        "trace_residue_minus_0_5": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=config.max_beta,
        ),
        "trace_residue_plus_0_5": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=config.max_beta,
        ),
        "trace_residue_plus_1_0": FFTDiagonalGaussianOperator(
            dim,
            initial_logits,
            max_beta=config.max_beta,
        ),
    }


def branch_prior_kl(
    branch: str,
    model: FFTDiagonalGaussianOperator,
) -> torch.Tensor:
    if branch == "ordinary_branch":
        return model.ordinary_prior_kl_exact()
    if branch == "cf_branch" or branch == "schatten_filtered_cf_branch":
        return model.cf_prior_kl_exact()
    if branch in TRACE_RESIDUE_COEFFS:
        return model.cf_prior_kl_exact() + TRACE_RESIDUE_COEFFS[branch] * model.trace_exact()
    raise ValueError(f"unknown branch: {branch}")


def collect_metrics(
    *,
    dim: int,
    epoch: int,
    branch: str,
    model: FFTDiagonalGaussianOperator,
    total_loss: torch.Tensor,
    mse_loss: torch.Tensor,
    prior_kl: torch.Tensor,
    features: torch.Tensor,
    probes: torch.Tensor,
    grad_norm: float,
    initial_logit_checksum: float,
    cf_probe_estimator: WickCarlemanPenalty,
) -> dict[str, float | int | str]:
    with torch.no_grad():
        ordinary_kl = model.ordinary_prior_kl_exact()
        cf_kl = model.cf_prior_kl_exact()
        trace_exact = model.trace_exact()
        t2_exact = model.t2_exact()
        logdet_exact = model.logdet_exact()
        neg_logdet2_exact = model.neg_logdet2_exact()
        sample_identity_error = (
            model.sample_log_rn_ordinary(features) - model.sample_log_rn_cf(features)
        ).abs()
        trace_hutch, pz = hutchinson_trace(model, probes)
        hs2_hutch = (pz * pz).sum(dim=-1).mean()
        neg_logdet2_k5 = neg_logdet2_series(model, probes, K=5, first_power=pz)
        cf_probe_value, cf_probe_diag = cf_probe_estimator(
            model,
            features,
            probes=probes,
            return_diagnostics=True,
        )
        trace_residue_coeff = TRACE_RESIDUE_COEFFS.get(branch, 0.0)
        prior_kl_current = branch_prior_kl(branch, model)
        trace_residue_delta = prior_kl_current - cf_kl
        trace_residue_expected = trace_residue_coeff * trace_exact
        trace_residue_abs_error = (trace_residue_delta - trace_residue_expected).abs()

    return {
        "dimension": dim,
        "epoch": epoch,
        "branch": branch,
        "total_loss": float(total_loss.detach()),
        "mse_loss": float(mse_loss.detach()),
        "prior_kl_used": float(prior_kl.detach()),
        "ordinary_kl_exact": float(ordinary_kl),
        "cf_kl_exact": float(cf_kl),
        "ordinary_cf_kl_abs_error": float((ordinary_kl - cf_kl).abs()),
        "trace_residue_coeff": float(trace_residue_coeff),
        "trace_residue_delta_same_params": float(trace_residue_delta),
        "trace_residue_expected_same_params": float(trace_residue_expected),
        "trace_residue_abs_error_same_params": float(trace_residue_abs_error),
        "trace_exact": float(trace_exact),
        "t2_exact": float(t2_exact),
        "logdet_exact": float(logdet_exact),
        "neg_logdet2_exact": float(neg_logdet2_exact),
        "trace_hutch": float(trace_hutch),
        "hs2_hutch": float(hs2_hutch),
        "neg_logdet2_k5_hutch": float(neg_logdet2_k5),
        "cf_probe_penalty_k5": float(cf_probe_value),
        "cf_probe_trace": float(cf_probe_diag["trace"]),
        "cf_probe_hs2": float(cf_probe_diag["hs2"]),
        "sample_log_rn_identity_abs_error": float(sample_identity_error),
        "grad_norm": grad_norm,
        "initial_logit_checksum": initial_logit_checksum,
    }


def run_training_toy(config: TrainingConfig = TrainingConfig()) -> list[dict[str, float | int | str]]:
    torch.manual_seed(config.seed)
    rows: list[dict[str, float | int | str]] = []
    cf_probe_estimator = WickCarlemanPenalty(K=5, num_probes=config.num_probes)

    for dim in config.dimensions:
        target_operator = make_target_operator(config, dim)
        models = make_models(config, dim)
        optimizers = {
            branch: torch.optim.Adam(model.parameters(), lr=config.lr)
            for branch, model in models.items()
        }
        checksums = {
            branch: float(model.logits.detach().sum())
            for branch, model in models.items()
        }

        print(f"\nTraining ELBO toy at dimension {dim}")
        for epoch in range(config.epochs):
            features = torch.randn(config.batch_size, dim)
            probes = rademacher_probes(features, config.num_probes)

            with torch.no_grad():
                target_features = target_operator(features)

            for branch in BRANCHES:
                model = models[branch]
                optimizer = optimizers[branch]
                optimizer.zero_grad()

                model_features = model(features)
                mse_loss = nn.functional.mse_loss(model_features, target_features)
                prior_kl = branch_prior_kl(branch, model)
                total_loss = config.mse_weight * mse_loss + config.kl_weight * prior_kl
                total_loss.backward()

                grad = model.logits.grad
                grad_norm = float(grad.detach().norm()) if grad is not None else 0.0

                rows.append(
                    collect_metrics(
                        dim=dim,
                        epoch=epoch,
                        branch=branch,
                        model=model,
                        total_loss=total_loss,
                        mse_loss=mse_loss,
                        prior_kl=prior_kl,
                        features=features,
                        probes=probes,
                        grad_norm=grad_norm,
                        initial_logit_checksum=checksums[branch],
                        cf_probe_estimator=cf_probe_estimator,
                    )
                )
                optimizer.step()

        final_rows = [row for row in rows if row["dimension"] == dim and row["epoch"] == config.epochs - 1]
        cf_final_mse = next(float(row["mse_loss"]) for row in final_rows if row["branch"] == "cf_branch")
        for row in final_rows:
            mse_ratio = float(row["mse_loss"]) / max(cf_final_mse, 1e-12)
            print(
                f"  {row['branch']:<30}"
                f" loss={float(row['total_loss']):9.4f}"
                f" mse={float(row['mse_loss']):8.4f}"
                f" mse/cf={mse_ratio:6.2f}"
                f" T2={float(row['t2_exact']):9.3f}"
                f" grad={float(row['grad_norm']):8.3f}"
            )

    return rows


def annotate_final_mse_ratios(rows: list[dict[str, float | int | str]]) -> None:
    dimensions = sorted({int(row["dimension"]) for row in rows})
    for dim in dimensions:
        dim_rows = [row for row in rows if int(row["dimension"]) == dim]
        final_epoch = max(int(row["epoch"]) for row in dim_rows)
        final_rows = [row for row in dim_rows if int(row["epoch"]) == final_epoch]
        cf_final_mse = next(float(row["mse_loss"]) for row in final_rows if row["branch"] == "cf_branch")
        final_mse_by_branch = {
            str(row["branch"]): float(row["mse_loss"])
            for row in final_rows
        }
        for row in dim_rows:
            final_mse = final_mse_by_branch[str(row["branch"])]
            row["final_mse"] = final_mse
            row["mse_ratio_to_cf_final"] = final_mse / max(cf_final_mse, 1e-12)


def save_results(rows: list[dict[str, float | int | str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def branch_styles() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    colors = {
        "ordinary_branch": "#2f6fbd",
        "cf_branch": "#238b45",
        "schatten_filtered_cf_branch": "#7b3294",
        "trace_residue_minus_0_5": "#8c6d31",
        "trace_residue_plus_0_5": "#e6550d",
        "trace_residue_plus_1_0": "#d33f49",
    }
    labels = {
        "ordinary_branch": "ordinary KL",
        "cf_branch": "CF KL",
        "schatten_filtered_cf_branch": "filtered CF KL",
        "trace_residue_minus_0_5": "CF - 0.5 Tr",
        "trace_residue_plus_0_5": "CF + 0.5 Tr",
        "trace_residue_plus_1_0": "CF + 1.0 Tr",
    }
    linestyles = {
        "ordinary_branch": "-",
        "cf_branch": "--",
        "schatten_filtered_cf_branch": "-",
        "trace_residue_minus_0_5": "-.",
        "trace_residue_plus_0_5": "-.",
        "trace_residue_plus_1_0": "-",
    }
    return colors, labels, linestyles


def plot_results(rows: list[dict[str, float | int | str]], png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dimensions = sorted({int(row["dimension"]) for row in rows})
    colors, labels, linestyles = branch_styles()

    fig, axes = plt.subplots(4, len(dimensions), figsize=(4.8 * len(dimensions), 12.5), sharex="col")
    if len(dimensions) == 1:
        axes = axes.reshape(4, 1)

    for col, dim in enumerate(dimensions):
        dim_rows = [row for row in rows if int(row["dimension"]) == dim]
        for branch in BRANCHES:
            branch_rows = [row for row in dim_rows if row["branch"] == branch]
            branch_rows.sort(key=lambda row: int(row["epoch"]))
            epochs = [int(row["epoch"]) for row in branch_rows]
            mse_loss = [float(row["mse_loss"]) for row in branch_rows]
            prior_kl = [float(row["prior_kl_used"]) for row in branch_rows]
            grad_norm = [float(row["grad_norm"]) for row in branch_rows]
            t2_exact = [float(row["t2_exact"]) for row in branch_rows]

            axes[0, col].plot(epochs, mse_loss, color=colors[branch], linestyle=linestyles[branch], label=labels[branch], linewidth=1.8)
            axes[1, col].plot(epochs, prior_kl, color=colors[branch], linestyle=linestyles[branch], linewidth=1.8)
            axes[2, col].plot(epochs, grad_norm, color=colors[branch], linestyle=linestyles[branch], linewidth=1.8)
            axes[3, col].plot(epochs, t2_exact, color=colors[branch], linestyle=linestyles[branch], linewidth=1.8)

        axes[0, col].set_title(f"Dimension {dim}")
        axes[0, col].set_yscale("log")
        axes[1, col].set_yscale("symlog", linthresh=1e-2)
        axes[2, col].set_yscale("log")
        axes[3, col].set_yscale("log")
        axes[3, col].set_xlabel("Epoch")
        for row_idx in range(4):
            axes[row_idx, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Task MSE")
    axes[1, 0].set_ylabel("Prior term (signed symlog)")
    axes[2, 0].set_ylabel("Gradient norm")
    axes[3, 0].set_ylabel("T2 = ||P||_S2^2")
    axes[0, 0].legend(loc="best", fontsize=9)

    fig.suptitle("ELBO Prior-KL Training Toy: MSE Parity, Trace Residues, and Schatten Filtering", y=0.995)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def plot_pareto_results(rows: list[dict[str, float | int | str]], png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dimensions = sorted({int(row["dimension"]) for row in rows})
    colors, labels, linestyles = branch_styles()
    fig, axes = plt.subplots(1, len(dimensions), figsize=(4.8 * len(dimensions), 4.4), sharey=False)
    if len(dimensions) == 1:
        axes = [axes]

    for col, dim in enumerate(dimensions):
        axis = axes[col]
        dim_rows = [row for row in rows if int(row["dimension"]) == dim]
        for branch in BRANCHES:
            branch_rows = [row for row in dim_rows if row["branch"] == branch]
            branch_rows.sort(key=lambda row: int(row["epoch"]))
            t2_values = [float(row["t2_exact"]) for row in branch_rows]
            mse_values = [float(row["mse_loss"]) for row in branch_rows]

            axis.plot(
                t2_values,
                mse_values,
                color=colors[branch],
                linestyle=linestyles[branch],
                label=labels[branch],
                linewidth=1.8,
                alpha=0.95,
            )
            axis.scatter(
                [t2_values[0]],
                [mse_values[0]],
                facecolors="white",
                edgecolors=colors[branch],
                marker="o",
                s=34,
                linewidths=1.2,
                zorder=3,
            )
            axis.scatter(
                [t2_values[-1]],
                [mse_values[-1]],
                color=colors[branch],
                marker="o",
                s=34,
                linewidths=0.8,
                zorder=4,
            )
            if len(t2_values) >= 2:
                axis.annotate(
                    "",
                    xy=(t2_values[-1], mse_values[-1]),
                    xytext=(t2_values[-2], mse_values[-2]),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": colors[branch],
                        "lw": 1.0,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )

        axis.set_title(f"Dimension {dim}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("T2 = ||P||_S2^2")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Task MSE")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Stability-Fit Pareto Trajectories: Task MSE vs T2", y=1.03)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def validate_results(rows: list[dict[str, float | int | str]]) -> None:
    dimensions = sorted({int(row["dimension"]) for row in rows})
    for dim in dimensions:
        dim_rows = [row for row in rows if int(row["dimension"]) == dim]
        branches = {row["branch"] for row in dim_rows}
        missing = set(BRANCHES) - branches
        if missing:
            raise RuntimeError(f"dimension {dim} missing branches: {sorted(missing)}")

        first_epoch = [row for row in dim_rows if int(row["epoch"]) == 0]
        checksums = {row["branch"]: row["initial_logit_checksum"] for row in first_epoch}
        checksum_values = [checksums[branch] for branch in BRANCHES]
        if max(checksum_values) - min(checksum_values) > 1e-5:
            raise RuntimeError(f"dimension {dim} branches did not share initial logits")

        max_kl_error = max(float(row["ordinary_cf_kl_abs_error"]) for row in dim_rows)
        if max_kl_error > 1e-4:
            raise RuntimeError(f"dimension {dim} ordinary/CF KL identity error {max_kl_error:.3e}")

        for branch, coeff in TRACE_RESIDUE_COEFFS.items():
            residue_rows = [row for row in dim_rows if row["branch"] == branch]
            residue_error = max(float(row["trace_residue_abs_error_same_params"]) for row in residue_rows)
            if residue_error > 1e-5:
                raise RuntimeError(
                    f"dimension {dim} {branch} residue check failed for coefficient {coeff}"
                )

        final_unfiltered = [
            row for row in dim_rows
            if row["branch"] == "cf_branch" and int(row["epoch"]) == max(int(r["epoch"]) for r in dim_rows)
        ][0]
        final_filtered = [
            row for row in dim_rows
            if row["branch"] == "schatten_filtered_cf_branch"
            and int(row["epoch"]) == max(int(r["epoch"]) for r in dim_rows)
        ][0]
        if float(final_filtered["t2_exact"]) >= float(final_unfiltered["t2_exact"]):
            raise RuntimeError(f"dimension {dim} filtered branch did not reduce T2")


if __name__ == "__main__":
    config = TrainingConfig()
    result_rows = run_training_toy(config)
    annotate_final_mse_ratios(result_rows)
    validate_results(result_rows)
    save_results(result_rows, config.output_csv)
    plot_results(result_rows, config.output_png)
    plot_pareto_results(result_rows, config.output_pareto_png)
    print(f"\nWrote {config.output_csv}")
    print(f"Wrote {config.output_png}")
    print(f"Wrote {config.output_pareto_png}")
