from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Spectrum:
    name: str
    scale: float
    alpha: Optional[float] = None

    def beta(self, dim: int) -> torch.Tensor:
        if self.alpha is None:
            return torch.full((dim,), self.scale, dtype=torch.float64)
        j = torch.arange(1, dim + 1, dtype=torch.float64)
        return self.scale / j.pow(self.alpha)


def gaussian_prior_kl_block(
    spectrum: Spectrum,
    resolution: int,
    *,
    t2_warn: float,
    t2_fail: float,
) -> dict[str, float | int | str]:
    r"""Summarize the Gaussian prior-KL block in a diffusion/ELBO toy model.

    The variational terminal law has whitened covariance (I-P)^{-1} relative
    to the base prior N(0,I).  In finite dimension,

        KL(q_P || p_0)
        = 1/2 Tr((I-P)^{-1}-I) + 1/2 log det(I-P)
        = 1/2 Tr(P^2(I-P)^{-1}) + 1/2 log det_2(I-P).

    The two expressions are algebraically identical at finite dimension.  The
    second expression is the Wick--Carleman finite-part form whose components
    remain meaningful on the Hilbert--Schmidt branch.
    """
    dim = resolution * resolution
    beta = spectrum.beta(dim)

    if torch.any(beta >= 1):
        raise ValueError(f"{spectrum.name} has beta >= 1; I-P is not positive")

    trace_p = beta.sum().item()
    t2 = beta.square().sum().item()

    trace_excess = 0.5 * (beta / (1.0 - beta)).sum().item()
    ordinary_logdet = 0.5 * torch.log1p(-beta).sum().item()
    ordinary_kl = trace_excess + ordinary_logdet

    wick_mean = 0.5 * (beta.square() / (1.0 - beta)).sum().item()
    det2_constant = 0.5 * (torch.log1p(-beta) + beta).sum().item()
    cf_kl = wick_mean + det2_constant

    beta32 = beta.to(torch.float32)
    ordinary_kl_fp32 = (
        0.5 * (beta32 / (1.0 - beta32)).sum()
        + 0.5 * torch.log1p(-beta32).sum()
    ).item()
    cf_kl_fp32 = (
        0.5 * (beta32.square() / (1.0 - beta32)).sum()
        + 0.5 * (torch.log1p(-beta32) + beta32).sum()
    ).item()
    ordinary_fp32_abs_error = abs(ordinary_kl_fp32 - ordinary_kl)
    cf_fp32_abs_error = abs(cf_kl_fp32 - cf_kl)

    # The sample-level log-RN variance under q_P.  With a uniform spectral gap,
    # this is equivalent to T2 and predicts minibatch noise in stochastic ELBO
    # estimators.
    rn_sample_var = 0.5 * (beta.square() / (1.0 - beta).square()).sum().item()
    rn_sample_sd = math.sqrt(rn_sample_var)

    cancellation_ratio = (
        (abs(trace_excess) + abs(ordinary_logdet)) / max(abs(ordinary_kl), 1e-12)
    )

    if t2 >= t2_fail:
        flag = "fail"
    elif t2 >= t2_warn:
        flag = "warn"
    else:
        flag = "ok"

    return {
        "spectrum": spectrum.name,
        "resolution": resolution,
        "dim": dim,
        "trace_P": trace_p,
        "T2_hat_exact": t2,
        "ordinary_trace_excess": trace_excess,
        "ordinary_logdet": ordinary_logdet,
        "ordinary_kl": ordinary_kl,
        "wick_mean": wick_mean,
        "det2_constant": det2_constant,
        "cf_kl": cf_kl,
        "ordinary_kl_fp32": ordinary_kl_fp32,
        "cf_kl_fp32": cf_kl_fp32,
        "ordinary_fp32_abs_error": ordinary_fp32_abs_error,
        "cf_fp32_abs_error": cf_fp32_abs_error,
        "rn_sample_sd": rn_sample_sd,
        "cancellation_ratio": cancellation_ratio,
        "T2_flag": flag,
    }


def run_litmus(
    resolutions: Iterable[int] = (256, 512, 1024, 2048),
    *,
    output: Path = EXPERIMENT_DIR / "diffusion_elbo_prior_litmus.csv",
) -> list[dict[str, float | int | str]]:
    spectra = (
        Spectrum("pixel_white_beta_0.02", scale=0.02),
        Spectrum("hs_filtered_alpha_0.60", scale=0.30, alpha=0.60),
        Spectrum("borderline_alpha_0.50", scale=0.75, alpha=0.50),
        Spectrum("non_s2_alpha_0.40", scale=0.60, alpha=0.40),
    )

    # Practical diagnostic thresholds for this toy, not theorem thresholds.
    t2_warn = 100.0
    t2_fail = 400.0

    rows: list[dict[str, float | int | str]] = []
    for spectrum in spectra:
        for resolution in resolutions:
            rows.append(
                gaussian_prior_kl_block(
                    spectrum,
                    resolution,
                    t2_warn=t2_warn,
                    t2_fail=t2_fail,
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def plot_results(rows: list[dict[str, float | int | str]], png_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/gaussian-hilbert-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gaussian-hilbert-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    spectra = list(dict.fromkeys(str(row["spectrum"]) for row in rows))
    colors = {
        "pixel_white_beta_0.02": "#4c78a8",
        "hs_filtered_alpha_0.60": "#54a24b",
        "borderline_alpha_0.50": "#f58518",
        "non_s2_alpha_0.40": "#d33f49",
    }
    labels = {
        "pixel_white_beta_0.02": "pixel white beta=0.02",
        "hs_filtered_alpha_0.60": "HS filtered alpha=0.60",
        "borderline_alpha_0.50": "borderline alpha=0.50",
        "non_s2_alpha_0.40": "non-S2 alpha=0.40",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), sharex=True)
    axes_flat = axes.ravel()

    for spectrum in spectra:
        spectrum_rows = [row for row in rows if str(row["spectrum"]) == spectrum]
        spectrum_rows.sort(key=lambda row: int(row["resolution"]))
        resolutions = [int(row["resolution"]) for row in spectrum_rows]
        t2_values = [float(row["T2_hat_exact"]) for row in spectrum_rows]
        ordinary_kl = [float(row["ordinary_kl"]) for row in spectrum_rows]
        ordinary_fp32_error = [float(row["ordinary_fp32_abs_error"]) for row in spectrum_rows]
        cf_fp32_error = [float(row["cf_fp32_abs_error"]) for row in spectrum_rows]
        rn_sample_sd = [float(row["rn_sample_sd"]) for row in spectrum_rows]

        color = colors.get(spectrum)
        label = labels.get(spectrum, spectrum)
        axes_flat[0].plot(resolutions, t2_values, marker="o", linewidth=2.0, color=color, label=label)
        axes_flat[1].plot(resolutions, ordinary_kl, marker="o", linewidth=2.0, color=color)
        axes_flat[2].plot(
            resolutions,
            ordinary_fp32_error,
            marker="o",
            linewidth=1.8,
            color=color,
            linestyle="-",
            label=f"{label} ordinary",
        )
        axes_flat[2].plot(
            resolutions,
            cf_fp32_error,
            marker="s",
            linewidth=1.8,
            color=color,
            linestyle="--",
            label=f"{label} CF",
        )
        axes_flat[3].plot(resolutions, rn_sample_sd, marker="o", linewidth=2.0, color=color)

    axes_flat[0].axhline(100.0, color="#8a8f98", linestyle="--", linewidth=1.2, label="warn T2=100")
    axes_flat[0].axhline(400.0, color="#5f6368", linestyle=":", linewidth=1.4, label="fail T2=400")
    axes_flat[0].set_ylabel("T2 = ||P||_S2^2")
    axes_flat[0].set_title("Schatten-2 Diagnostic")
    axes_flat[0].set_yscale("log")

    axes_flat[1].set_ylabel("Prior KL")
    axes_flat[1].set_title("Diffusion ELBO Prior Block")
    axes_flat[1].set_yscale("log")

    axes_flat[2].set_ylabel("Absolute FP32 Error")
    axes_flat[2].set_title("Ordinary KL vs CF Stability")
    axes_flat[2].set_yscale("log")

    axes_flat[3].set_ylabel("Sample log-RN SD")
    axes_flat[3].set_title("Minibatch Noise Proxy")
    axes_flat[3].set_yscale("log")

    for axis in axes_flat:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Resolution")
        axis.set_xticks(sorted({int(row["resolution"]) for row in rows}))
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(True, alpha=0.3)

    axes_flat[0].legend(loc="best", fontsize=8)
    axes_flat[2].legend(loc="best", fontsize=7, ncol=2)
    fig.suptitle("Diffusion ELBO Prior Litmus", y=0.99)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    print(
        "spectrum                 res      dim        T2      KL(ord) "
        "KL(CF)  err32(ord) err32(CF)  RN_SD  cancel  flag"
    )
    print("-" * 119)
    for row in rows:
        print(
            f"{row['spectrum']:<24}"
            f"{int(row['resolution']):>5}"
            f"{int(row['dim']):>11}"
            f"{float(row['T2_hat_exact']):>9.3f}"
            f"{float(row['ordinary_kl']):>12.3f}"
            f"{float(row['cf_kl']):>8.3f}"
            f"{float(row['ordinary_fp32_abs_error']):>12.2e}"
            f"{float(row['cf_fp32_abs_error']):>10.2e}"
            f"{float(row['rn_sample_sd']):>8.3f}"
            f"{float(row['cancellation_ratio']):>8.1f}"
            f"{str(row['T2_flag']):>7}"
        )


if __name__ == "__main__":
    result_rows = run_litmus()
    plot_results(result_rows, EXPERIMENT_DIR / "diffusion_elbo_prior_litmus.png")
    print_table(result_rows)
    print("\nWrote experiments/diffusion/diffusion_elbo_prior_litmus.csv")
    print("Wrote experiments/diffusion/diffusion_elbo_prior_litmus.png")
