"""Deterministic frequency-matched counterexample for Figure 2.

The construction separates a coordinate/frequency diagnostic from the
operator-ideal diagnostic that controls Gaussian objective fluctuations.  Let
``d = m**2`` and let ``U_d`` be any flat unitary (we use the DFT for explicit
small-matrix checks).  For ``0 < eta < 1``, define

    K_stable   = U diag(eta, eta/(m+1), ..., eta/(m+1)) U*,
    K_unstable = U diag(eta [m times], 0, ..., 0) U*.

Both operators have

    diag(K) = (eta/m) 1,   Tr(K) = eta m,   ||K||_op = eta,

so a coordinatewise power-spectrum diagnostic cannot distinguish them.  Their
Hilbert--Schmidt masses are nevertheless

    ||K_stable||_HS^2   = eta^2 2m/(m+1) -> 2 eta^2,
    ||K_unstable||_HS^2 = eta^2 m          = eta^2 sqrt(d).

Tensoring both families with the same physical-frequency envelope preserves
the matched radial Fourier diagonal.  Any frequency-only multiplier also
factorizes out of both Hilbert--Schmidt masses, leaving their diverging ratio
unchanged.

The script writes the exact scaling table and a four-panel figure, then
materializes small DFT examples to verify the diagonal, spectrum, trace,
operator norm, and Hilbert--Schmidt identities numerically.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gaussian-hilbert-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gaussian-hilbert-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    """Configuration for the deterministic construction."""

    eta: float = 0.60
    m_values: tuple[int, ...] = (
        2,
        3,
        4,
        6,
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        768,
        1024,
    )
    selected_m: int = 64
    spatial_resolution: int = 64
    envelope_s: float = 1.25
    lowpass_cutoff: float = 8.0
    matrix_check_m: tuple[int, ...] = (2, 3, 4, 6, 8)
    output_csv: Path = EXPERIMENT_DIR / "frequency_matched_schatten.csv"
    output_png: Path = EXPERIMENT_DIR / "frequency_matched_schatten.png"


def _validate_config(config: Config) -> None:
    if not 0.0 < config.eta < 1.0:
        raise ValueError("eta must lie in (0, 1)")
    if any(m < 2 for m in config.m_values):
        raise ValueError("all m values must be at least 2")
    if config.selected_m not in config.m_values:
        raise ValueError("selected_m must be one of m_values")
    if config.spatial_resolution < 2:
        raise ValueError("spatial_resolution must be at least 2")
    if config.envelope_s <= 1.0:
        raise ValueError("envelope_s must exceed 1 for a 2D S2 envelope")
    if config.lowpass_cutoff <= 0.0:
        raise ValueError("lowpass_cutoff must be positive")


def channel_eigenvalues(m: int, eta: float, family: str) -> np.ndarray:
    """Return the exact channel eigenvalues for one family."""

    channels = m * m
    if family == "stable":
        eigenvalues = np.full(channels, eta / (m + 1.0), dtype=np.float64)
        eigenvalues[0] = eta
        return eigenvalues
    if family == "unstable":
        eigenvalues = np.zeros(channels, dtype=np.float64)
        eigenvalues[:m] = eta
        return eigenvalues
    raise ValueError(f"unknown family: {family}")


def flat_dft(channels: int) -> np.ndarray:
    """Return the unitary DFT matrix used only in small explicit checks."""

    indices = np.arange(channels, dtype=np.float64)
    phase = -2.0j * np.pi * np.outer(indices, indices) / channels
    return np.exp(phase) / np.sqrt(channels)


def physical_frequency_factors(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radii, common covariance envelope, and frequency-only mask.

    Frequencies are physical integer modes, rather than radius normalized by
    grid size.  The covariance envelope is

        g(k) = (1 + ||k||^2)^(-s/2),

    whose squared sum is uniformly finite under 2D refinement for ``s > 1``.
    The low-pass multiplier acts directly on the covariance perturbation.
    """

    resolution = config.spatial_resolution
    modes = np.fft.fftfreq(resolution, d=1.0 / resolution)
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    radii = np.sqrt(kx * kx + ky * ky)
    envelope = np.power(1.0 + radii * radii, -0.5 * config.envelope_s)
    lowpass = np.exp(-np.power(radii / config.lowpass_cutoff, 4.0))
    return radii, envelope, lowpass


def _phi(eigenvalue: float | np.ndarray) -> float | np.ndarray:
    """The nonnegative additive-covariance det2 residue x - log(1+x)."""

    return eigenvalue - np.log1p(eigenvalue)


def make_rows(config: Config) -> list[dict[str, float | int]]:
    """Build one exact summary row per channel width ``m``."""

    _, envelope, lowpass = physical_frequency_factors(config)
    spatial_hs2 = float(np.square(envelope).sum())
    masked_spatial_hs2 = float(np.square(envelope * lowpass).sum())

    rows: list[dict[str, float | int]] = []
    for m in config.m_values:
        channels = m * m
        eta = config.eta
        stable_tail = eta / (m + 1.0)

        # Closed forms obtained from the two multiplicity patterns.
        trace_closed = eta * m
        coordinate_diagonal = eta / m
        stable_hs2_closed = eta * eta * (2.0 * m / (m + 1.0))
        unstable_hs2_closed = eta * eta * m

        # Independently evaluate the same quantities from the spectral
        # multiplicities.  This avoids materializing large d x d matrices.
        stable_trace_spectrum = eta + (channels - 1) * stable_tail
        unstable_trace_spectrum = m * eta
        stable_hs2_spectrum = eta * eta + (channels - 1) * stable_tail * stable_tail
        unstable_hs2_spectrum = m * eta * eta

        stable_logdet = np.log1p(eta) + (channels - 1) * np.log1p(stable_tail)
        unstable_logdet = m * np.log1p(eta)
        stable_ordinary_kl = 0.5 * (stable_trace_spectrum - stable_logdet)
        unstable_ordinary_kl = 0.5 * (unstable_trace_spectrum - unstable_logdet)
        stable_det2_residue = 0.5 * (
            float(_phi(eta)) + (channels - 1) * float(_phi(stable_tail))
        )
        unstable_det2_residue = 0.5 * m * float(_phi(eta))

        rows.append(
            {
                "m": m,
                "channel_dimension_d": channels,
                "eta": eta,
                "stable_tail_eigenvalue": stable_tail,
                "matched_coordinate_diagonal": coordinate_diagonal,
                "matched_trace_closed": trace_closed,
                "stable_trace_spectrum": stable_trace_spectrum,
                "unstable_trace_spectrum": unstable_trace_spectrum,
                "stable_operator_norm": eta,
                "unstable_operator_norm": eta,
                "stable_hs2_closed": stable_hs2_closed,
                "stable_hs2_spectrum": stable_hs2_spectrum,
                "unstable_hs2_closed": unstable_hs2_closed,
                "unstable_hs2_spectrum": unstable_hs2_spectrum,
                "unstable_to_stable_hs2_ratio": (m + 1.0) / 2.0,
                "stable_centered_quadratic_variance": 0.5 * stable_hs2_closed,
                "unstable_centered_quadratic_variance": 0.5 * unstable_hs2_closed,
                "stable_ordinary_kl": stable_ordinary_kl,
                "unstable_ordinary_kl": unstable_ordinary_kl,
                "stable_det2_renormalized_residue": stable_det2_residue,
                "unstable_det2_renormalized_residue": unstable_det2_residue,
                "stable_det2_identity_error": abs(stable_ordinary_kl - stable_det2_residue),
                "unstable_det2_identity_error": abs(unstable_ordinary_kl - unstable_det2_residue),
                "common_spatial_hs2": spatial_hs2,
                "masked_spatial_hs2": masked_spatial_hs2,
                "stable_factorized_hs2": spatial_hs2 * stable_hs2_closed,
                "unstable_factorized_hs2": spatial_hs2 * unstable_hs2_closed,
                "stable_lowpass_factorized_hs2": masked_spatial_hs2 * stable_hs2_closed,
                "unstable_lowpass_factorized_hs2": masked_spatial_hs2 * unstable_hs2_closed,
                "lowpass_hs2_ratio": (m + 1.0) / 2.0,
            }
        )
    return rows


def explicit_matrix_errors(m: int, eta: float) -> dict[str, float]:
    """Materialize a small pair and compare it with every closed form."""

    channels = m * m
    unitary = flat_dft(channels)
    expected_diagonal = eta / m
    errors: dict[str, float] = {}

    gram = np.einsum("ki,kj->ij", unitary.conj(), unitary, optimize=True)
    errors["unitarity"] = float(np.max(np.abs(gram - np.eye(channels))))
    errors["flat_modulus"] = float(
        np.max(np.abs(np.square(np.abs(unitary)) - 1.0 / channels))
    )

    for family in ("stable", "unstable"):
        eigenvalues = channel_eigenvalues(m, eta, family)
        operator = np.einsum(
            "ik,k,jk->ij",
            unitary,
            eigenvalues,
            unitary.conj(),
            optimize=True,
        )
        recovered_eigenvalues = np.linalg.eigvalsh(operator)[::-1]
        expected_eigenvalues = np.sort(eigenvalues)[::-1]

        errors[f"{family}_hermitian"] = float(
            np.max(np.abs(operator - operator.conj().T))
        )
        errors[f"{family}_diagonal"] = float(
            np.max(np.abs(np.diag(operator) - expected_diagonal))
        )
        errors[f"{family}_spectrum"] = float(
            np.max(np.abs(recovered_eigenvalues - expected_eigenvalues))
        )
        errors[f"{family}_trace"] = float(
            abs(np.trace(operator).real - eta * m)
        )
        expected_hs2 = (
            eta * eta * 2.0 * m / (m + 1.0)
            if family == "stable"
            else eta * eta * m
        )
        errors[f"{family}_hs2"] = float(
            abs(np.vdot(operator, operator).real - expected_hs2)
        )
        errors[f"{family}_operator_norm"] = float(
            abs(np.linalg.norm(operator, ord=2) - eta)
        )
    return errors


def validate_rows(rows: list[dict[str, float | int]], config: Config) -> dict[str, float]:
    """Fail loudly if any algebraic or explicit-matrix identity is violated."""

    if len(rows) != len(config.m_values):
        raise RuntimeError("unexpected number of rows")

    algebraic_error_keys = (
        ("stable_trace_spectrum", "matched_trace_closed"),
        ("unstable_trace_spectrum", "matched_trace_closed"),
        ("stable_hs2_spectrum", "stable_hs2_closed"),
        ("unstable_hs2_spectrum", "unstable_hs2_closed"),
    )
    max_algebraic_error = max(
        abs(float(row[left]) - float(row[right]))
        for row in rows
        for left, right in algebraic_error_keys
    )
    max_det2_error = max(
        max(
            float(row["stable_det2_identity_error"]),
            float(row["unstable_det2_identity_error"]),
        )
        for row in rows
    )
    max_ratio_error = max(
        abs(
            float(row["unstable_hs2_closed"]) / float(row["stable_hs2_closed"])
            - float(row["unstable_to_stable_hs2_ratio"])
        )
        for row in rows
    )
    max_lowpass_ratio_error = max(
        abs(
            float(row["unstable_lowpass_factorized_hs2"])
            / float(row["stable_lowpass_factorized_hs2"])
            - float(row["lowpass_hs2_ratio"])
        )
        for row in rows
    )

    matrix_errors: dict[str, float] = {}
    for m in config.matrix_check_m:
        for key, value in explicit_matrix_errors(m, config.eta).items():
            matrix_errors[f"m={m}:{key}"] = value
    max_matrix_error = max(matrix_errors.values())

    tolerance = 2.0e-10
    diagnostics = {
        "max_algebraic_error": max_algebraic_error,
        "max_det2_identity_error": max_det2_error,
        "max_ratio_error": max_ratio_error,
        "max_lowpass_ratio_error": max_lowpass_ratio_error,
        "max_explicit_matrix_error": max_matrix_error,
    }
    if max(diagnostics.values()) > tolerance:
        worst_matrix_key = max(matrix_errors, key=matrix_errors.get)
        raise RuntimeError(
            "closed-form validation failed: "
            f"diagnostics={diagnostics}, "
            f"worst matrix error={worst_matrix_key}:{matrix_errors[worst_matrix_key]:.3e}"
        )

    # The stable family must approach 2 eta^2, while the unstable family and
    # their ratio grow monotonically with m.
    stable_hs2 = np.asarray([float(row["stable_hs2_closed"]) for row in rows])
    unstable_hs2 = np.asarray([float(row["unstable_hs2_closed"]) for row in rows])
    ratios = np.asarray([float(row["unstable_to_stable_hs2_ratio"]) for row in rows])
    if not np.all(np.diff(stable_hs2) > 0.0):
        raise RuntimeError("stable HS2 should increase monotonically toward its finite limit")
    if stable_hs2[-1] >= 2.0 * config.eta * config.eta:
        raise RuntimeError("stable HS2 crossed its analytic limiting value")
    if not np.all(np.diff(unstable_hs2) > 0.0) or not np.all(np.diff(ratios) > 0.0):
        raise RuntimeError("unstable HS2 and the HS2 ratio must increase with m")
    return diagnostics


def save_rows(rows: list[dict[str, float | int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float | int]], config: Config) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stable_color = "#1b9e77"
    unstable_color = "#d95f02"
    neutral_color = "#3f4c6b"

    selected_m = config.selected_m
    selected_channels = selected_m * selected_m
    selected_diagonal = config.eta / selected_m
    stable_eigenvalues = channel_eigenvalues(selected_m, config.eta, "stable")
    unstable_eigenvalues = channel_eigenvalues(selected_m, config.eta, "unstable")

    radial_coordinate = np.linspace(0.0, config.spatial_resolution / np.sqrt(2.0), 500)
    radial_envelope = np.power(
        1.0 + radial_coordinate * radial_coordinate,
        -0.5 * config.envelope_s,
    )
    matched_radial_diagonal = selected_diagonal * radial_envelope

    channels = np.asarray([int(row["channel_dimension_d"]) for row in rows])
    stable_hs2 = np.asarray([float(row["stable_hs2_closed"]) for row in rows])
    unstable_hs2 = np.asarray([float(row["unstable_hs2_closed"]) for row in rows])
    stable_variance = np.asarray(
        [float(row["stable_centered_quadratic_variance"]) for row in rows]
    )
    unstable_variance = np.asarray(
        [float(row["unstable_centered_quadratic_variance"]) for row in rows]
    )
    stable_residue = np.asarray(
        [float(row["stable_det2_renormalized_residue"]) for row in rows]
    )
    unstable_residue = np.asarray(
        [float(row["unstable_det2_renormalized_residue"]) for row in rows]
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    ax_frequency, ax_spectrum, ax_hs2, ax_objective = axes.ravel()

    # The two curves are exactly coincident; draw both styles and state the
    # exact equality in the panel rather than introducing a visual offset.
    ax_frequency.plot(
        radial_coordinate,
        matched_radial_diagonal,
        color=stable_color,
        linewidth=3.2,
        label="stable family",
    )
    ax_frequency.plot(
        radial_coordinate,
        matched_radial_diagonal,
        color=unstable_color,
        linewidth=1.6,
        linestyle="--",
        label="unstable family",
    )
    ax_frequency.set_yscale("log")
    ax_frequency.set_title("Matched frequency observable")
    ax_frequency.set_xlabel(r"physical frequency radius $\|k\|$")
    ax_frequency.set_ylabel(r"diagonal PSD $g(k)\,\eta/m$")
    ax_frequency.text(
        0.97,
        0.94,
        "curves coincide exactly",
        ha="right",
        va="top",
        transform=ax_frequency.transAxes,
        fontsize=9,
        color=neutral_color,
    )
    ax_frequency.legend(loc="lower left", fontsize=8)

    ranks = np.arange(1, selected_channels + 1)
    ax_spectrum.plot(
        ranks,
        stable_eigenvalues,
        color=stable_color,
        linewidth=2.2,
        label="stable: mass dispersed",
    )
    ax_spectrum.plot(
        ranks,
        unstable_eigenvalues,
        color=unstable_color,
        linewidth=2.0,
        linestyle="--",
        label=r"unstable: rank $\sqrt{d}$",
    )
    ax_spectrum.set_xscale("log")
    ax_spectrum.set_title(rf"Hidden eigenspectra ($d={selected_channels:,}$)")
    ax_spectrum.set_xlabel("eigenvalue rank")
    ax_spectrum.set_ylabel(r"$\lambda_j(K)$")
    ax_spectrum.set_ylim(-0.025 * config.eta, 1.08 * config.eta)
    ax_spectrum.legend(loc="upper right", fontsize=8)

    ax_hs2.plot(
        channels,
        stable_hs2,
        "o-",
        color=stable_color,
        linewidth=2.0,
        markersize=3.7,
        label=r"stable: $2\eta^2\sqrt{d}/(\sqrt{d}+1)$",
    )
    ax_hs2.plot(
        channels,
        unstable_hs2,
        "s--",
        color=unstable_color,
        linewidth=2.0,
        markersize=3.4,
        label=r"unstable: $\eta^2\sqrt{d}$",
    )
    ax_hs2.axhline(
        2.0 * config.eta * config.eta,
        color=stable_color,
        linestyle=":",
        linewidth=1.3,
        alpha=0.8,
    )
    ax_hs2.set_xscale("log")
    ax_hs2.set_yscale("log")
    ax_hs2.set_title(r"$\mathcal{S}_2$ diagnostic separates the pair")
    ax_hs2.set_xlabel(r"channel dimension $d=m^2$")
    ax_hs2.set_ylabel(r"$T_2(d)=\|K_d\|_{\mathcal{S}_2}^2$")
    ax_hs2.legend(loc="upper left", fontsize=8)

    ax_objective.plot(
        channels,
        stable_residue,
        color=stable_color,
        linewidth=2.2,
    )
    ax_objective.plot(
        channels,
        unstable_residue,
        color=unstable_color,
        linewidth=2.2,
    )
    ax_objective.plot(
        channels,
        stable_variance,
        color=stable_color,
        linestyle="--",
        linewidth=1.8,
    )
    ax_objective.plot(
        channels,
        unstable_variance,
        color=unstable_color,
        linestyle="--",
        linewidth=1.8,
    )
    ax_objective.set_xscale("log")
    ax_objective.set_yscale("log")
    ax_objective.set_title("Gaussian objective consequence")
    ax_objective.set_xlabel(r"channel dimension $d=m^2$")
    ax_objective.set_ylabel("exact finite-cutoff value")
    family_handles = (
        Line2D([0], [0], color=stable_color, linewidth=2.2, label="stable family"),
        Line2D([0], [0], color=unstable_color, linewidth=2.2, label="unstable family"),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.0,
            label=r"$-\frac{1}{2}\log\det_2(I+K)$",
        ),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="--", label=r"$\mathrm{Var}(Q_K)=T_2/2$"),
    )
    ax_objective.legend(handles=family_handles, loc="upper left", fontsize=8)

    for axis in axes.ravel():
        axis.grid(True, alpha=0.25)

    fig.suptitle(
        "Frequency Matching Does Not Control Operator-Ideal Accumulation",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout()
    config.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run(config: Config = Config()) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    _validate_config(config)
    rows = make_rows(config)
    diagnostics = validate_rows(rows, config)
    save_rows(rows, config.output_csv)
    plot_rows(rows, config)
    return rows, diagnostics


if __name__ == "__main__":
    result_rows, result_diagnostics = run()
    first = result_rows[0]
    last = result_rows[-1]
    print(f"Wrote {Config.output_csv}")
    print(f"Wrote {Config.output_png}")
    print(
        "Largest identity errors: "
        + ", ".join(f"{key}={value:.3e}" for key, value in result_diagnostics.items())
    )
    print(
        "HS2 separation: "
        f"ratio {float(first['unstable_to_stable_hs2_ratio']):.1f}x "
        f"at d={int(first['channel_dimension_d']):,} -> "
        f"{float(last['unstable_to_stable_hs2_ratio']):.1f}x "
        f"at d={int(last['channel_dimension_d']):,}"
    )
