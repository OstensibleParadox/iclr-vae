"""Deterministic frequency-matched counterexample and baseline sweep.

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
the matched radial Fourier diagonal.  We compare four controls:

* a hard cutoff in physical Fourier modes;
* a smooth physical-frequency mask;
* a power-law frequency decay with no channel-ideal constraint; and
* an ideal-aware channel gate with an explicitly square-summable envelope.

The three frequency-only controls are calibrated to the same physical-envelope
distortion and preserve the DC response exactly.  Because they do not act on
the hidden channel multiplicity, their raw T2 remains proportional to
``sqrt(d)``.  The ideal-aware gate preserves the same DC anchor and uses the
same physical mask as the smooth control, but makes the channel T2 uniformly
bounded.  This is a deterministic operator comparison, not a surrogate
learning task.

The script writes the matched-pair table, a baseline-sweep table, and a
four-panel figure.  Small DFT matrices verify the diagonal, spectrum, trace,
operator norm, and Hilbert--Schmidt identities numerically.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    target_physical_distortion: float = 0.15
    ideal_channel_exponent: float = 0.75
    matrix_check_m: tuple[int, ...] = (2, 3, 4, 6, 8)
    output_csv: Path = EXPERIMENT_DIR / "frequency_matched_schatten.csv"
    output_summary_csv: Path = EXPERIMENT_DIR / "frequency_matched_baseline_sweep.csv"
    output_png: Path = EXPERIMENT_DIR / "frequency_matched_schatten.png"


@dataclass(frozen=True)
class PhysicalControl:
    """One radial control and its exactly reported matching statistics."""

    name: str
    label: str
    mask: np.ndarray
    parameter_name: str
    parameter_value: float
    distortion: float
    spatial_hs2: float


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
    if not 0.0 < config.target_physical_distortion < 1.0:
        raise ValueError("target_physical_distortion must lie in (0, 1)")
    if config.ideal_channel_exponent <= 0.5:
        raise ValueError("ideal_channel_exponent must exceed 1/2")
    if len(set(config.m_values)) != len(config.m_values):
        raise ValueError("m_values must be unique")
    if tuple(sorted(config.m_values)) != config.m_values:
        raise ValueError("m_values must be strictly increasing")


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


def physical_frequency_grid(config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return physical radii and the common covariance envelope.

    Frequencies are physical integer modes, rather than radius normalized by
    grid size.  The covariance envelope is

        g(k) = (1 + ||k||^2)^(-s/2),

    whose squared sum is uniformly finite under 2D refinement for ``s > 1``.
    Every baseline multiplier below acts directly on this relative covariance
    envelope, not on a noise standard deviation.
    """

    resolution = config.spatial_resolution
    modes = np.fft.fftfreq(resolution, d=1.0 / resolution)
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    radii = np.sqrt(kx * kx + ky * ky)
    envelope = np.power(1.0 + radii * radii, -0.5 * config.envelope_s)
    return radii, envelope


def physical_distortion(envelope: np.ndarray, mask: np.ndarray) -> float:
    r"""Return the relative squared distortion of the physical envelope.

    We match

        D_phys(M) = ||G - MG||_HS^2 / ||G||_HS^2,

    where ``G`` is the common diagonal physical-frequency covariance envelope.
    This deliberately measures only physical-frequency attenuation.  Channel
    reshaping is reported separately rather than hidden inside a fictitious
    task metric.
    """

    denominator = float(np.square(envelope).sum())
    if denominator <= 0.0:
        raise ValueError("physical envelope must have positive squared mass")
    return float(np.square(envelope * (1.0 - mask)).sum() / denominator)


def _bisect_parameter(
    make_mask: Callable[[float], np.ndarray],
    envelope: np.ndarray,
    target: float,
    low: float,
    high: float,
    *,
    distortion_increases: bool,
    iterations: int = 90,
) -> tuple[float, np.ndarray]:
    """Calibrate a monotone mask family to a physical distortion target."""

    low_distortion = physical_distortion(envelope, make_mask(low))
    high_distortion = physical_distortion(envelope, make_mask(high))
    if distortion_increases:
        bracketed = low_distortion <= target <= high_distortion
    else:
        bracketed = high_distortion <= target <= low_distortion
    if not bracketed:
        raise RuntimeError(
            "physical-distortion target is not bracketed: "
            f"low={low_distortion:.6g}, target={target:.6g}, "
            f"high={high_distortion:.6g}"
        )

    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        midpoint_distortion = physical_distortion(envelope, make_mask(midpoint))
        if distortion_increases:
            if midpoint_distortion < target:
                low = midpoint
            else:
                high = midpoint
        else:
            if midpoint_distortion > target:
                low = midpoint
            else:
                high = midpoint

    parameter = 0.5 * (low + high)
    return parameter, make_mask(parameter)


def build_physical_controls(config: Config) -> dict[str, PhysicalControl]:
    """Construct DC-preserving controls at matched physical distortion.

    A hard cutoff can attain only a discrete set of distortions on a finite
    grid.  We select the hard cutoff nearest the requested target, then match
    the smooth and power-law controls to that attained value to numerical
    precision.  All masks satisfy M(0)=1.
    """

    radii, envelope = physical_frequency_grid(config)
    unique_radii = np.unique(radii)
    if unique_radii.size < 2:
        raise RuntimeError("physical grid needs at least two distinct radii")
    hard_candidates: list[tuple[float, float, np.ndarray]] = []
    # Exclude the identity mask at the largest radius so every requested sweep
    # performs a nonzero, calibratable intervention.
    for cutoff in unique_radii[:-1]:
        mask = (radii <= cutoff).astype(np.float64)
        hard_candidates.append(
            (physical_distortion(envelope, mask), float(cutoff), mask)
        )
    hard_distortion, hard_cutoff, hard_mask = min(
        hard_candidates,
        key=lambda item: abs(item[0] - config.target_physical_distortion),
    )

    positive_radius = unique_radii[unique_radii > 0.0]
    if positive_radius.size == 0:
        raise RuntimeError("physical grid has no nonzero frequency")
    min_positive_radius = float(positive_radius.min())
    max_radius = float(radii.max())

    def smooth_mask(cutoff: float) -> np.ndarray:
        return np.exp(-np.power(radii / cutoff, 4.0))

    smooth_cutoff, smooth = _bisect_parameter(
        smooth_mask,
        envelope,
        hard_distortion,
        min_positive_radius * 1.0e-4,
        max_radius * 1.0e4,
        distortion_increases=False,
    )

    def decay_mask(exponent: float) -> np.ndarray:
        return np.power(1.0 + np.square(radii), -0.5 * exponent)

    decay_high = 1.0
    while physical_distortion(envelope, decay_mask(decay_high)) < hard_distortion:
        decay_high *= 2.0
        if decay_high > 1024.0:
            raise RuntimeError("could not bracket the power-law decay exponent")
    decay_exponent, decay = _bisect_parameter(
        decay_mask,
        envelope,
        hard_distortion,
        0.0,
        decay_high,
        distortion_increases=True,
    )

    identity = np.ones_like(envelope)
    definitions = (
        ("identity", "no physical mask", identity, "none", 0.0),
        ("hard", "hard physical cutoff", hard_mask, "cutoff", hard_cutoff),
        ("smooth", "smooth frequency mask", smooth, "cutoff", smooth_cutoff),
        (
            "decay",
            "unconstrained frequency decay",
            decay,
            "exponent",
            decay_exponent,
        ),
    )
    controls: dict[str, PhysicalControl] = {}
    for name, label, mask, parameter_name, parameter_value in definitions:
        controls[name] = PhysicalControl(
            name=name,
            label=label,
            mask=mask,
            parameter_name=parameter_name,
            parameter_value=float(parameter_value),
            distortion=physical_distortion(envelope, mask),
            spatial_hs2=float(np.square(envelope * mask).sum()),
        )
    return controls


def _phi(eigenvalue: float | np.ndarray) -> float | np.ndarray:
    """The nonnegative additive-covariance det2 residue x - log(1+x)."""

    return eigenvalue - np.log1p(eigenvalue)


def ideal_aware_channel_eigenvalues(m: int, config: Config) -> np.ndarray:
    r"""Return the gated nonzero channel spectrum eta * j^{-p}, j <= m.

    The raw unstable family has ``m`` eigenvalues all equal to ``eta``.  The
    gate preserves the anchored leading response because w_1=1, while p>1/2
    makes ``sum_j w_j^2`` finite independently of ``m``.
    """

    ranks = np.arange(1, m + 1, dtype=np.float64)
    return config.eta * np.power(ranks, -config.ideal_channel_exponent)


def make_rows(config: Config) -> list[dict[str, float | int]]:
    """Build one exact summary row per channel width ``m``."""

    _, envelope = physical_frequency_grid(config)
    controls = build_physical_controls(config)
    spatial_hs2 = float(np.square(envelope).sum())
    smooth_spatial_hs2 = controls["smooth"].spatial_hs2

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
        ideal_eigenvalues = ideal_aware_channel_eigenvalues(m, config)
        ideal_hs2 = float(np.square(ideal_eigenvalues).sum())

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
                "matched_physical_distortion": controls["hard"].distortion,
                "hard_cutoff": controls["hard"].parameter_value,
                "smooth_cutoff": controls["smooth"].parameter_value,
                "unconstrained_decay_exponent": controls["decay"].parameter_value,
                "hard_spatial_hs2": controls["hard"].spatial_hs2,
                "smooth_spatial_hs2": smooth_spatial_hs2,
                "decay_spatial_hs2": controls["decay"].spatial_hs2,
                "stable_factorized_hs2": spatial_hs2 * stable_hs2_closed,
                "unstable_factorized_hs2": spatial_hs2 * unstable_hs2_closed,
                "stable_smooth_factorized_hs2": smooth_spatial_hs2 * stable_hs2_closed,
                "unstable_smooth_factorized_hs2": smooth_spatial_hs2 * unstable_hs2_closed,
                "ideal_channel_hs2": ideal_hs2,
                "ideal_smooth_factorized_hs2": smooth_spatial_hs2 * ideal_hs2,
                "smooth_hs2_ratio": (m + 1.0) / 2.0,
            }
        )
    return rows


def make_baseline_rows(
    config: Config,
    controls: dict[str, PhysicalControl],
) -> list[dict[str, float | int | str]]:
    """Build the closed-form matched-distortion baseline sweep."""

    baseline_specs = (
        (
            "stable_matched_pair",
            "stable matched pair",
            "stable",
            "identity",
            "bounded",
        ),
        (
            "unstable_unfiltered",
            "unstable, unfiltered",
            "unstable",
            "identity",
            "sqrt(d)",
        ),
        (
            "hard_physical_cutoff",
            "hard physical cutoff",
            "unstable",
            "hard",
            "sqrt(d)",
        ),
        (
            "smooth_frequency_mask",
            "smooth frequency mask",
            "unstable",
            "smooth",
            "sqrt(d)",
        ),
        (
            "unconstrained_frequency_decay",
            "unconstrained decay",
            "unstable",
            "decay",
            "sqrt(d)",
        ),
        (
            "ideal_aware_s2_gate",
            "ideal-aware channel gate",
            "ideal",
            "smooth",
            "bounded",
        ),
    )

    rows: list[dict[str, float | int | str]] = []
    ideal_channel_upper_bound = config.eta**2 * (
        1.0 + 1.0 / (2.0 * config.ideal_channel_exponent - 1.0)
    )
    for m in config.m_values:
        channels = m * m
        stable_hs2 = config.eta**2 * 2.0 * m / (m + 1.0)
        unstable_hs2 = config.eta**2 * m
        ideal_eigenvalues = ideal_aware_channel_eigenvalues(m, config)
        ideal_hs2 = float(np.square(ideal_eigenvalues).sum())
        ideal_trace = float(ideal_eigenvalues.sum())

        channel_statistics = {
            "stable": (
                stable_hs2,
                config.eta * m,
                "analytic stable family",
            ),
            "unstable": (
                unstable_hs2,
                config.eta * m,
                "frequency-only control",
            ),
            "ideal": (
                ideal_hs2,
                ideal_trace,
                "channel S2 envelope",
            ),
        }

        for baseline, label, channel_kind, control_name, growth_class in baseline_specs:
            control = controls[control_name]
            channel_hs2, channel_trace, mechanism = channel_statistics[channel_kind]
            raw_t2 = control.spatial_hs2 * channel_hs2
            frequency_only_lower_bound = (
                config.eta**2 * m if channel_kind == "unstable" else float("nan")
            )
            if channel_kind == "ideal":
                uniform_t2_upper_bound = (
                    control.spatial_hs2 * ideal_channel_upper_bound
                )
            elif channel_kind == "stable":
                uniform_t2_upper_bound = control.spatial_hs2 * 2.0 * config.eta**2
            else:
                uniform_t2_upper_bound = float("nan")
            rows.append(
                {
                    "m": m,
                    "channel_dimension_d": channels,
                    "baseline": baseline,
                    "label": label,
                    "mechanism": mechanism,
                    "channel_spectrum": channel_kind,
                    "physical_control": control_name,
                    "physical_parameter_name": control.parameter_name,
                    "physical_parameter_value": control.parameter_value,
                    "physical_distortion": control.distortion,
                    "dc_mask_response": float(control.mask.flat[0]),
                    "anchored_dc_operator_gain": config.eta * float(control.mask.flat[0]),
                    "normalized_anchored_dc_response": float(control.mask.flat[0]),
                    "physical_hs2": control.spatial_hs2,
                    "channel_trace": channel_trace,
                    "channel_hs2": channel_hs2,
                    "raw_t2": raw_t2,
                    "centered_quadratic_variance": 0.5 * raw_t2,
                    "frequency_only_t2_lower_bound": frequency_only_lower_bound,
                    "analytic_uniform_t2_upper_bound": uniform_t2_upper_bound,
                    "raw_t2_over_eta2_sqrt_d": raw_t2
                    / (config.eta**2 * np.sqrt(float(channels))),
                    "growth_class": growth_class,
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


def validate_rows(
    rows: list[dict[str, float | int]],
    baseline_rows: list[dict[str, float | int | str]],
    controls: dict[str, PhysicalControl],
    config: Config,
) -> dict[str, float]:
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
    max_smooth_ratio_error = max(
        abs(
            float(row["unstable_smooth_factorized_hs2"])
            / float(row["stable_smooth_factorized_hs2"])
            - float(row["smooth_hs2_ratio"])
        )
        for row in rows
    )

    dc_response_error = max(
        abs(float(control.mask.flat[0]) - 1.0) for control in controls.values()
    )
    matched_distortion = controls["hard"].distortion
    matched_distortion_error = max(
        abs(controls[name].distortion - matched_distortion)
        for name in ("hard", "smooth", "decay")
    )

    expected_baselines = {
        "stable_matched_pair",
        "unstable_unfiltered",
        "hard_physical_cutoff",
        "smooth_frequency_mask",
        "unconstrained_frequency_decay",
        "ideal_aware_s2_gate",
    }
    expected_row_count = len(config.m_values) * len(expected_baselines)
    if len(baseline_rows) != expected_row_count:
        raise RuntimeError(
            f"expected {expected_row_count} baseline rows, got {len(baseline_rows)}"
        )
    if {str(row["baseline"]) for row in baseline_rows} != expected_baselines:
        raise RuntimeError("baseline sweep is missing a required control")

    frequency_only_names = {
        "unstable_unfiltered",
        "hard_physical_cutoff",
        "smooth_frequency_mask",
        "unconstrained_frequency_decay",
    }
    frequency_lower_bound_violation = max(
        max(
            0.0,
            float(row["frequency_only_t2_lower_bound"]) - float(row["raw_t2"]),
        )
        for row in baseline_rows
        if str(row["baseline"]) in frequency_only_names
    )
    anchor_response_error = max(
        abs(float(row["normalized_anchored_dc_response"]) - 1.0)
        for row in baseline_rows
    )

    # p > 1/2 gives sum_{j>=1} j^{-2p} <= 1 + integral_1^inf x^{-2p} dx.
    ideal_channel_bound = config.eta**2 * (
        1.0 + 1.0 / (2.0 * config.ideal_channel_exponent - 1.0)
    )
    ideal_bound_violation = max(
        max(0.0, float(row["channel_hs2"]) - ideal_channel_bound)
        for row in baseline_rows
        if str(row["baseline"]) == "ideal_aware_s2_gate"
    )

    frequency_constant_errors: list[float] = []
    for baseline in frequency_only_names:
        baseline_subset = [
            row for row in baseline_rows if str(row["baseline"]) == baseline
        ]
        ratios = np.asarray(
            [float(row["raw_t2_over_eta2_sqrt_d"]) for row in baseline_subset]
        )
        frequency_constant_errors.append(float(np.max(np.abs(ratios - ratios[0]))))
    max_frequency_growth_error = max(frequency_constant_errors)

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
        "max_smooth_ratio_error": max_smooth_ratio_error,
        "max_dc_response_error": dc_response_error,
        "max_matched_distortion_error": matched_distortion_error,
        "max_frequency_lower_bound_violation": frequency_lower_bound_violation,
        "max_anchor_response_error": anchor_response_error,
        "max_ideal_bound_violation": ideal_bound_violation,
        "max_frequency_growth_error": max_frequency_growth_error,
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

    ideal_raw_t2 = np.asarray(
        [
            float(row["raw_t2"])
            for row in baseline_rows
            if str(row["baseline"]) == "ideal_aware_s2_gate"
        ]
    )
    if not np.all(np.diff(ideal_raw_t2) > 0.0):
        raise RuntimeError("ideal-aware T2 should increase monotonically to its finite limit")
    ideal_full_bound = controls["smooth"].spatial_hs2 * ideal_channel_bound
    if ideal_raw_t2[-1] >= ideal_full_bound:
        raise RuntimeError("ideal-aware T2 crossed its analytic uniform bound")
    return diagnostics


def save_rows(
    rows: list[dict[str, float | int | str]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(
    rows: list[dict[str, float | int]],
    baseline_rows: list[dict[str, float | int | str]],
    controls: dict[str, PhysicalControl],
    config: Config,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {
        "stable_matched_pair": "#1b9e77",
        "unstable_unfiltered": "#d95f02",
        "hard_physical_cutoff": "#7570b3",
        "smooth_frequency_mask": "#e6ab02",
        "unconstrained_frequency_decay": "#66a61e",
        "ideal_aware_s2_gate": "#1f4e79",
    }
    linestyles: dict[str, str | tuple[int, tuple[int, ...]]] = {
        "stable_matched_pair": "-",
        "unstable_unfiltered": "--",
        "hard_physical_cutoff": "-.",
        "smooth_frequency_mask": ":",
        "unconstrained_frequency_decay": (0, (5, 2)),
        "ideal_aware_s2_gate": "-",
    }
    markers = {
        "stable_matched_pair": "o",
        "unstable_unfiltered": "s",
        "hard_physical_cutoff": "^",
        "smooth_frequency_mask": "D",
        "unconstrained_frequency_decay": "v",
        "ideal_aware_s2_gate": "P",
    }

    selected_m = config.selected_m
    selected_channels = selected_m * selected_m
    stable_eigenvalues = channel_eigenvalues(selected_m, config.eta, "stable")
    unstable_eigenvalues = channel_eigenvalues(selected_m, config.eta, "unstable")
    ideal_eigenvalues = np.zeros(selected_channels, dtype=np.float64)
    ideal_eigenvalues[:selected_m] = ideal_aware_channel_eigenvalues(
        selected_m, config
    )

    radial_coordinate = np.linspace(
        0.0,
        config.spatial_resolution / np.sqrt(2.0),
        800,
    )
    radial_envelope = np.power(
        1.0 + radial_coordinate * radial_coordinate,
        -0.5 * config.envelope_s,
    )
    hard_cutoff = controls["hard"].parameter_value
    smooth_cutoff = controls["smooth"].parameter_value
    decay_exponent = controls["decay"].parameter_value
    radial_masks = {
        "hard": (radial_coordinate <= hard_cutoff).astype(np.float64),
        "smooth": np.exp(-np.power(radial_coordinate / smooth_cutoff, 4.0)),
        "decay": np.power(
            1.0 + np.square(radial_coordinate),
            -0.5 * decay_exponent,
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    ax_frequency, ax_spectrum, ax_hs2, ax_certificate = axes.ravel()

    ax_frequency.plot(
        radial_coordinate,
        radial_envelope,
        color="#555555",
        linewidth=2.4,
        label="unmasked envelope",
    )
    mask_styles = {
        "hard": (colors["hard_physical_cutoff"], "-.", "hard cutoff"),
        "smooth": (colors["smooth_frequency_mask"], ":", "smooth mask"),
        "decay": (
            colors["unconstrained_frequency_decay"],
            "--",
            "unconstrained decay",
        ),
    }
    for name, mask in radial_masks.items():
        color, linestyle, label = mask_styles[name]
        ax_frequency.plot(
            radial_coordinate,
            radial_envelope * mask,
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
            label=label,
        )
    ax_frequency.set_yscale("log")
    ax_frequency.set_ylim(1.0e-8, 1.4)
    ax_frequency.set_title("Matched physical-frequency attenuation")
    ax_frequency.set_xlabel(r"physical frequency radius $\|k\|$")
    ax_frequency.set_ylabel(r"relative covariance response $g(k)M(k)$")
    ax_frequency.text(
        0.97,
        0.94,
        rf"$D_{{\rm phys}}={controls['hard'].distortion:.3f}$;  $M(0)=1$",
        ha="right",
        va="top",
        transform=ax_frequency.transAxes,
        fontsize=9,
        color="#3f4c6b",
    )
    ax_frequency.legend(loc="lower left", fontsize=8)

    ranks = np.arange(1, selected_channels + 1)
    ax_spectrum.plot(
        ranks,
        stable_eigenvalues,
        color=colors["stable_matched_pair"],
        linewidth=2.2,
        label="stable: mass dispersed",
    )
    ax_spectrum.plot(
        ranks,
        unstable_eigenvalues,
        color=colors["unstable_unfiltered"],
        linewidth=2.0,
        linestyle="--",
        label=r"unstable: rank $\sqrt{d}$",
    )
    ax_spectrum.plot(
        ranks,
        ideal_eigenvalues,
        color=colors["ideal_aware_s2_gate"],
        linewidth=2.0,
        linestyle="-.",
        label=rf"ideal-aware: $\eta j^{{-{config.ideal_channel_exponent:g}}}$",
    )
    ax_spectrum.set_xscale("log")
    ax_spectrum.set_title(rf"Hidden channel spectra ($d={selected_channels:,}$)")
    ax_spectrum.set_xlabel("eigenvalue rank")
    ax_spectrum.set_ylabel(r"$\lambda_j(K)$")
    ax_spectrum.set_ylim(-0.025 * config.eta, 1.08 * config.eta)
    ax_spectrum.legend(loc="upper right", fontsize=8)

    baseline_order = (
        "unstable_unfiltered",
        "hard_physical_cutoff",
        "smooth_frequency_mask",
        "unconstrained_frequency_decay",
        "ideal_aware_s2_gate",
        "stable_matched_pair",
    )
    for baseline in baseline_order:
        subset = [
            row for row in baseline_rows if str(row["baseline"]) == baseline
        ]
        subset.sort(key=lambda row: int(row["channel_dimension_d"]))
        dimensions = np.asarray(
            [int(row["channel_dimension_d"]) for row in subset]
        )
        raw_t2 = np.asarray([float(row["raw_t2"]) for row in subset])
        ax_hs2.plot(
            dimensions,
            raw_t2,
            color=colors[baseline],
            linestyle=linestyles[baseline],
            marker=markers[baseline],
            markevery=max(1, len(dimensions) // 7),
            linewidth=2.0 if baseline == "ideal_aware_s2_gate" else 1.65,
            markersize=4.0,
            label=str(subset[0]["label"]),
        )
    ax_hs2.set_xscale("log")
    ax_hs2.set_yscale("log")
    ax_hs2.set_title(r"Raw $T_2$ under matched controls")
    ax_hs2.set_xlabel(r"channel dimension $d=m^2$")
    ax_hs2.set_ylabel(r"$T_2(K)=\|K\|_{\mathcal{S}_2}^2$")
    ax_hs2.legend(loc="upper left", fontsize=7.2, ncol=2)

    for baseline in baseline_order:
        subset = [
            row for row in baseline_rows if str(row["baseline"]) == baseline
        ]
        subset.sort(key=lambda row: int(row["channel_dimension_d"]))
        dimensions = np.asarray(
            [int(row["channel_dimension_d"]) for row in subset]
        )
        certificate = np.asarray(
            [float(row["raw_t2_over_eta2_sqrt_d"]) for row in subset]
        )
        ax_certificate.plot(
            dimensions,
            certificate,
            color=colors[baseline],
            linestyle=linestyles[baseline],
            marker=markers[baseline],
            markevery=max(1, len(dimensions) // 7),
            linewidth=2.0 if baseline == "ideal_aware_s2_gate" else 1.65,
            markersize=4.0,
            label=str(subset[0]["label"]),
        )
    ax_certificate.axhline(
        1.0,
        color="#222222",
        linewidth=1.2,
        linestyle=":",
        label=r"DC lower bound for frequency-only gates",
    )
    ax_certificate.set_xscale("log")
    ax_certificate.set_yscale("log")
    ax_certificate.set_title(r"Frequency-only gates retain the $\sqrt{d}$ factor")
    ax_certificate.set_xlabel(r"channel dimension $d=m^2$")
    ax_certificate.set_ylabel(r"$T_2/(\eta^2\sqrt{d})$")
    ax_certificate.legend(loc="upper right", fontsize=7.2)

    for axis in axes.ravel():
        axis.grid(True, alpha=0.25)

    fig.suptitle(
        "Frequency-Only Controls Cannot Repair Hidden Operator-Ideal Growth",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout()
    config.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run(
    config: Config = Config(),
) -> tuple[
    list[dict[str, float | int]],
    list[dict[str, float | int | str]],
    dict[str, float],
]:
    _validate_config(config)
    controls = build_physical_controls(config)
    rows = make_rows(config)
    baseline_rows = make_baseline_rows(config, controls)
    diagnostics = validate_rows(rows, baseline_rows, controls, config)
    save_rows(rows, config.output_csv)
    save_rows(baseline_rows, config.output_summary_csv)
    plot_rows(rows, baseline_rows, controls, config)
    return rows, baseline_rows, diagnostics


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def parse_args() -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Run the deterministic matched-frequency Schatten sweep."
    )
    parser.add_argument("--eta", type=float, default=defaults.eta)
    parser.add_argument("--m-values", type=_parse_int_tuple, default=defaults.m_values)
    parser.add_argument("--selected-m", type=int, default=defaults.selected_m)
    parser.add_argument(
        "--spatial-resolution",
        type=int,
        default=defaults.spatial_resolution,
    )
    parser.add_argument("--envelope-s", type=float, default=defaults.envelope_s)
    parser.add_argument(
        "--target-physical-distortion",
        type=float,
        default=defaults.target_physical_distortion,
    )
    parser.add_argument(
        "--ideal-channel-exponent",
        type=float,
        default=defaults.ideal_channel_exponent,
    )
    parser.add_argument("--output-csv", type=Path, default=defaults.output_csv)
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=defaults.output_summary_csv,
    )
    parser.add_argument("--output-png", type=Path, default=defaults.output_png)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    defaults = Config()
    requested_checks = tuple(m for m in defaults.matrix_check_m if m in args.m_values)
    if not requested_checks:
        requested_checks = (min(args.m_values),)
    return Config(
        eta=args.eta,
        m_values=args.m_values,
        selected_m=args.selected_m,
        spatial_resolution=args.spatial_resolution,
        envelope_s=args.envelope_s,
        target_physical_distortion=args.target_physical_distortion,
        ideal_channel_exponent=args.ideal_channel_exponent,
        matrix_check_m=requested_checks,
        output_csv=args.output_csv,
        output_summary_csv=args.output_summary_csv,
        output_png=args.output_png,
    )


if __name__ == "__main__":
    result_config = config_from_args(parse_args())
    result_rows, result_baselines, result_diagnostics = run(result_config)
    first = result_rows[0]
    last = result_rows[-1]
    largest_dimension = int(last["channel_dimension_d"])
    largest_baselines = {
        str(row["baseline"]): row
        for row in result_baselines
        if int(row["channel_dimension_d"]) == largest_dimension
    }
    print(f"Wrote {result_config.output_csv}")
    print(f"Wrote {result_config.output_summary_csv}")
    print(f"Wrote {result_config.output_png}")
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
    hard_row = largest_baselines["hard_physical_cutoff"]
    smooth_row = largest_baselines["smooth_frequency_mask"]
    decay_row = largest_baselines["unconstrained_frequency_decay"]
    ideal_row = largest_baselines["ideal_aware_s2_gate"]
    print(
        "Matched physical distortion / DC response: "
        f"D={float(hard_row['physical_distortion']):.6f}, "
        f"R0={float(hard_row['normalized_anchored_dc_response']):.1f}; "
        f"hard/smooth/decay T2="
        f"{float(hard_row['raw_t2']):.3f}/"
        f"{float(smooth_row['raw_t2']):.3f}/"
        f"{float(decay_row['raw_t2']):.3f}, "
        f"ideal-aware T2={float(ideal_row['raw_t2']):.3f} "
        f"at d={largest_dimension:,}"
    )
