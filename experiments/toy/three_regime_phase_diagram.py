from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gaussian-hilbert-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gaussian-hilbert-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Spectrum:
    name: str
    label: str
    regime: str
    alpha: float
    scale: float = 0.45
    boundary: bool = False

    def beta(self, max_cutoff: int) -> torch.Tensor:
        j = torch.arange(1, max_cutoff + 1, dtype=torch.float64)
        beta = self.scale / j.pow(self.alpha)
        if torch.any(beta >= 1):
            raise ValueError(f"{self.name} has beta >= 1; I-P is not positive")
        return beta


SPECTRA = (
    Spectrum(
        name="trace_class_alpha_1_20",
        label=r"$S_1$: $\alpha=1.20$",
        regime="S1",
        alpha=1.20,
    ),
    Spectrum(
        name="hs_not_trace_alpha_0_60",
        label=r"$S_2\setminus S_1$: $\alpha=0.60$",
        regime="S2_not_S1",
        alpha=0.60,
    ),
    Spectrum(
        name="non_hs_alpha_0_40",
        label=r"outside $S_2$: $\alpha=0.40$",
        regime="outside_S2",
        alpha=0.40,
    ),
    Spectrum(
        name="boundary_alpha_0_50",
        label=r"boundary: $\alpha=0.50$",
        regime="boundary_outside_S2",
        alpha=0.50,
        boundary=True,
    ),
)


def default_cutoffs() -> tuple[int, ...]:
    raw = torch.logspace(1, 6, steps=90, dtype=torch.float64).round().to(torch.int64)
    values = sorted({int(x) for x in raw.tolist() if int(x) >= 2})
    return tuple(values)


def rows_for_spectrum(spectrum: Spectrum, cutoffs: Iterable[int]) -> list[dict[str, float | int | str]]:
    cutoff_values = tuple(cutoffs)
    max_cutoff = max(cutoff_values)
    beta = spectrum.beta(max_cutoff)

    trace_cumsum = beta.cumsum(dim=0)
    t2_cumsum = beta.square().cumsum(dim=0)
    logdet_cumsum = torch.log1p(-beta).cumsum(dim=0)
    det2_cumsum = (torch.log1p(-beta) + beta).cumsum(dim=0)

    rows: list[dict[str, float | int | str]] = []
    for cutoff in cutoff_values:
        idx = cutoff - 1
        trace_p = trace_cumsum[idx].item()
        t2_exact = t2_cumsum[idx].item()
        ordinary_quadratic_mean = 0.5 * trace_p
        ordinary_logdet = 0.5 * logdet_cumsum[idx].item()
        det2_constant = 0.5 * det2_cumsum[idx].item()
        wick_quadratic_sd = (0.5 * t2_exact) ** 0.5

        # At every finite cutoff, the ordinary scalar and the
        # Wick--Carleman--Fredholm scalar are the same random variable:
        #
        #   1/2 <Z,PZ> + 1/2 log det(I-P)
        # = 1/2 :<Z,PZ>: + 1/2 log det_2(I-P).
        #
        # Figure 1 is about componentwise convergence/divergence and L2
        # stability, not failure of the finite-dimensional identity.
        ordinary_total_mean = ordinary_quadratic_mean + ordinary_logdet
        cf_finite_part_mean = det2_constant
        finite_cutoff_identity_error = abs(ordinary_total_mean - cf_finite_part_mean)

        rows.append(
            {
                "spectrum": spectrum.name,
                "label": spectrum.label,
                "regime": spectrum.regime,
                "alpha": spectrum.alpha,
                "boundary": str(spectrum.boundary),
                "line_style": "dashed" if spectrum.boundary else "solid",
                "cutoff_N": cutoff,
                "trace_P": trace_p,
                "T2_exact": t2_exact,
                "ordinary_quadratic_mean": ordinary_quadratic_mean,
                "ordinary_logdet": ordinary_logdet,
                "ordinary_total_mean": ordinary_total_mean,
                "det2_constant": det2_constant,
                "wick_quadratic_sd": wick_quadratic_sd,
                "cf_finite_part_mean": cf_finite_part_mean,
                "finite_cutoff_identity_error": finite_cutoff_identity_error,
            }
        )

    return rows


def run_litmus(
    cutoffs: Iterable[int] = default_cutoffs(),
    *,
    output_csv: Path = EXPERIMENT_DIR / "three_regime_phase_diagram.csv",
    output_png: Path = EXPERIMENT_DIR / "three_regime_phase_diagram.png",
) -> list[dict[str, float | int | str]]:
    cutoff_values = tuple(cutoffs)
    rows: list[dict[str, float | int | str]] = []
    for spectrum in SPECTRA:
        rows.extend(rows_for_spectrum(spectrum, cutoff_values))

    validate_rows(rows)
    save_results(rows, output_csv)
    plot_results(rows, output_png)
    return rows


def save_results(rows: list[dict[str, float | int | str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rows_for(rows: list[dict[str, float | int | str]], spectrum_name: str) -> list[dict[str, float | int | str]]:
    selected = [row for row in rows if row["spectrum"] == spectrum_name]
    selected.sort(key=lambda row: int(row["cutoff_N"]))
    return selected


def _row_at_or_below(rows: list[dict[str, float | int | str]], cutoff: int) -> dict[str, float | int | str]:
    candidates = [row for row in rows if int(row["cutoff_N"]) <= cutoff]
    return candidates[-1]


def validate_rows(rows: list[dict[str, float | int | str]]) -> None:
    spectra = {row["spectrum"] for row in rows}
    expected = {spectrum.name for spectrum in SPECTRA}
    if spectra != expected:
        raise RuntimeError(f"unexpected spectra: got {sorted(spectra)}, expected {sorted(expected)}")

    max_identity_error = max(float(row["finite_cutoff_identity_error"]) for row in rows)
    if max_identity_error > 1e-10:
        raise RuntimeError(f"finite-cutoff identity error too large: {max_identity_error:.3e}")

    s1_rows = _rows_for(rows, "trace_class_alpha_1_20")
    s2_rows = _rows_for(rows, "hs_not_trace_alpha_0_60")
    outside_rows = _rows_for(rows, "non_hs_alpha_0_40")
    boundary_rows = _rows_for(rows, "boundary_alpha_0_50")

    max_cutoff = max(int(row["cutoff_N"]) for row in rows)
    last_decade_cutoff = max_cutoff // 10

    s1_last = s1_rows[-1]
    s1_decade = _row_at_or_below(s1_rows, last_decade_cutoff)
    if float(s1_last["trace_P"]) / float(s1_decade["trace_P"]) > 1.25:
        raise RuntimeError("S1 trace still grows too quickly over the final decade")
    if float(s1_last["T2_exact"]) / float(s1_decade["T2_exact"]) > 1.05:
        raise RuntimeError("S1 T2 still grows too quickly over the final decade")

    s2_last = s2_rows[-1]
    s2_decade = _row_at_or_below(s2_rows, last_decade_cutoff)
    if float(s2_last["trace_P"]) / float(s2_decade["trace_P"]) < 1.5:
        raise RuntimeError("S2\\S1 trace did not grow over the final decade")
    if float(s2_last["T2_exact"]) / float(s2_decade["T2_exact"]) > 1.15:
        raise RuntimeError("S2\\S1 T2 did not stabilize over the final decade")

    outside_last = outside_rows[-1]
    outside_decade = _row_at_or_below(outside_rows, last_decade_cutoff)
    if float(outside_last["T2_exact"]) / float(outside_decade["T2_exact"]) < 1.35:
        raise RuntimeError("outside-S2 T2 did not grow enough over the final decade")

    boundary_last = boundary_rows[-1]
    boundary_decade = _row_at_or_below(boundary_rows, last_decade_cutoff)
    boundary_ratio = float(boundary_last["T2_exact"]) / float(boundary_decade["T2_exact"])
    if not (1.05 < boundary_ratio < 1.40):
        raise RuntimeError(f"boundary T2 growth should be slow/logarithmic; got ratio {boundary_ratio:.3f}")


def plot_results(rows: list[dict[str, float | int | str]], png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {
        "trace_class_alpha_1_20": "#2f6fbd",
        "hs_not_trace_alpha_0_60": "#238b45",
        "non_hs_alpha_0_40": "#d33f49",
        "boundary_alpha_0_50": "#7b3294",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    ax_trace, ax_t2, ax_components, ax_finite_part = axes.ravel()

    for spectrum in SPECTRA:
        spectrum_rows = _rows_for(rows, spectrum.name)
        cutoffs = [int(row["cutoff_N"]) for row in spectrum_rows]
        linestyle = "--" if spectrum.boundary else "-"
        color = colors[spectrum.name]

        trace_p = [float(row["trace_P"]) for row in spectrum_rows]
        t2_exact = [float(row["T2_exact"]) for row in spectrum_rows]
        q_mean = [float(row["ordinary_quadratic_mean"]) for row in spectrum_rows]
        ordinary_logdet = [float(row["ordinary_logdet"]) for row in spectrum_rows]
        det2_constant = [float(row["det2_constant"]) for row in spectrum_rows]
        wick_sd = [float(row["wick_quadratic_sd"]) for row in spectrum_rows]

        ax_trace.plot(cutoffs, trace_p, color=color, linestyle=linestyle, linewidth=2.0)
        ax_t2.plot(cutoffs, t2_exact, color=color, linestyle=linestyle, linewidth=2.0)
        ax_components.plot(cutoffs, q_mean, color=color, linestyle=linestyle, linewidth=1.8)
        ax_components.plot(cutoffs, ordinary_logdet, color=color, linestyle=":" if not spectrum.boundary else "--", linewidth=1.8)
        ax_finite_part.plot(cutoffs, wick_sd, color=color, linestyle=linestyle, linewidth=1.8)
        ax_finite_part.plot(cutoffs, det2_constant, color=color, linestyle=":" if not spectrum.boundary else "--", linewidth=1.8)

    ax_trace.set_title(r"Trace component $\mathrm{Tr}(P_N)$")
    ax_trace.set_ylabel(r"$\mathrm{Tr}(P_N)$")
    ax_trace.set_xscale("log")
    ax_trace.set_yscale("log")

    ax_t2.set_title(r"Hilbert--Schmidt diagnostic $T_2$")
    ax_t2.set_ylabel(r"$T_2=\|P_N\|_{S_2}^2$")
    ax_t2.set_xscale("log")
    ax_t2.set_yscale("log")

    ax_components.set_title("Ordinary component drift")
    ax_components.set_ylabel(r"$\frac{1}{2}\mathrm{Tr}(P_N)$ and $\frac{1}{2}\log\det(I-P_N)$")
    ax_components.set_xscale("log")
    ax_components.set_yscale("symlog", linthresh=1e-2)

    ax_finite_part.set_title(r"Wick--$\det_2$ finite-part terms")
    ax_finite_part.set_ylabel(r"Wick SD and $\frac{1}{2}\log\det_2(I-P_N)$")
    ax_finite_part.set_xscale("log")
    ax_finite_part.set_yscale("symlog", linthresh=1e-2)

    for ax in axes.ravel():
        ax.set_xlabel("Spectral cutoff N")
        ax.grid(True, alpha=0.3)

    regime_handles = [
        Line2D([0], [0], color=colors[spectrum.name], linestyle="--" if spectrum.boundary else "-", linewidth=2.0, label=spectrum.label)
        for spectrum in SPECTRA
    ]
    component_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="positive component"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.8, label="negative determinant component"),
    ]
    ax_trace.legend(handles=regime_handles, loc="best", fontsize=8)
    ax_components.legend(handles=component_handles, loc="best", fontsize=8)

    fig.suptitle("Componentwise Cutoff Behavior of Gaussian Likelihood Terms", y=0.995)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    final_rows = [
        spectrum_rows[-1]
        for spectrum_rows in (_rows_for(rows, spectrum.name) for spectrum in SPECTRA)
    ]
    print("spectrum                    regime              N        Tr(P)        T2       det2")
    print("-" * 88)
    for row in final_rows:
        print(
            f"{str(row['spectrum']):<28}"
            f"{str(row['regime']):<18}"
            f"{int(row['cutoff_N']):>8}"
            f"{float(row['trace_P']):>12.4f}"
            f"{float(row['T2_exact']):>10.4f}"
            f"{float(row['det2_constant']):>11.4f}"
        )


if __name__ == "__main__":
    result_rows = run_litmus()
    print_table(result_rows)
    print("\nWrote experiments/toy/three_regime_phase_diagram.csv")
    print("Wrote experiments/toy/three_regime_phase_diagram.png")
