from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import add_common_args, ensure_run_tree, load_config, maybe_skip, now_iso, read_csv, run_dir, write_stage_metadata

STAGE = "plot_all"


def floats(rows, key):
    return [float(r[key]) for r in rows if r.get(key, "") not in {"", "nan", "NaN"}]


def write_basic_plots(run_path: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out: list[Path] = []
    figs = run_path / "figures"

    controlled = run_path / "csv" / "controlled_spectra_results.csv"
    if controlled.exists():
        rows = read_csv(controlled)
        fig, ax = plt.subplots(figsize=(6, 4))
        for regime in sorted({r["regime"] for r in rows}):
            sub = [r for r in rows if r["regime"] == regime]
            ax.plot([int(r["cutoff_N"]) for r in sub], [float(r["T2_diagnostic"]) for r in sub], marker="o", label=regime)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("cutoff N"); ax.set_ylabel("T2 diagnostic"); ax.legend(); ax.grid(True, alpha=0.25)
        p = figs / "fig1_phase_diagram.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); out.append(p)

    elbo = run_path / "csv" / "figure_source_fig2_elbo_toy_slim.csv"
    if elbo.exists():
        rows = read_csv(elbo)
        fig, ax = plt.subplots(figsize=(7, 4))
        for branch in sorted({r["branch"] for r in rows}):
            sub = [r for r in rows if r["branch"] == branch and r["dimension"] == rows[0]["dimension"]]
            ax.plot([int(r["epoch"]) for r in sub], [float(r["total_loss"]) for r in sub], label=branch)
        ax.set_xlabel("epoch"); ax.set_ylabel("total loss"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        p = figs / "fig2_elbo_toy_slim.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); out.append(p)
        p2 = figs / "fig3_elbo_pareto.png"; fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(floats(rows, "trace_exact"), floats(rows, "t2_exact"), s=8); ax.set_xlabel("trace"); ax.set_ylabel("T2"); fig.tight_layout(); fig.savefig(p2, dpi=160); plt.close(fig); out.append(p2)

    fno = run_path / "csv" / "fno_results.csv"
    if fno.exists():
        rows = read_csv(fno)
        fig, ax = plt.subplots(figsize=(6, 4))
        for branch in sorted({r["baseline"] for r in rows}):
            sub = [r for r in rows if r["baseline"] == branch]
            ax.scatter([float(r["t2_norm"]) for r in sub], [float(r["val_relative_l2"]) for r in sub], s=10, label=branch)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("normalized T2"); ax.set_ylabel("validation relative L2"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        p = figs / "fig4_fno_pareto_normT2.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); out.append(p)
        fig, ax = plt.subplots(figsize=(6, 4))
        for branch in sorted({r["baseline"] for r in rows}):
            sub = [r for r in rows if r["baseline"] == branch]
            ax.plot([int(r["epoch"]) for r in sub], [float(r["val_relative_l2"]) for r in sub], ".", label=branch)
        ax.set_xlabel("epoch"); ax.set_ylabel("validation relative L2"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        p = figs / "fig5_fno_timeseries_slim.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); out.append(p)

    sd_rows = []
    for path in sorted((run_path / "csv").glob("sdvae_jacobian_*_results.csv")):
        sd_rows.extend(read_csv(path))
    if sd_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        for input_type in sorted({r["input_type"] for r in sd_rows}):
            sub = [r for r in sd_rows if r["input_type"] == input_type]
            ax.plot([int(r["resolution"]) for r in sub], [float(r["T2_over_trace2"]) for r in sub], "o", label=input_type)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("resolution"); ax.set_ylabel("T2 / Tr(G)^2"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        p = figs / "fig6_sdvae_operator_ideal_filter.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); out.append(p)
        fig, ax = plt.subplots(figsize=(6, 4)); ax.axis("off"); ax.text(0.02, 0.8, "See spectra/*.csv for covariance eigenvalue lambda_i tables")
        p = figs / "appendix_sdvae_spectra.png"; fig.savefig(p, dpi=160); plt.close(fig); out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all figures from outputs only.")
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    run_path = run_dir(args.run_id)
    if args.dry_run:
        print(f"[dry-run] {STAGE}: would read CSV/PT/NPY under {run_path} and write figures only under figures/")
        return
    ensure_run_tree(run_path)
    if maybe_skip(args, STAGE, run_path):
        return
    start = now_iso()
    outputs = write_basic_plots(run_path)
    if not outputs:
        raise RuntimeError("No plots were produced; required CSVs missing")
    write_stage_metadata(run_path=run_path, config=config, stage=STAGE, start_time=start, output_files=outputs, extra={"figure_count": len(outputs), "source_policy": "outputs_run_dir_only"})
    print(f"plotted {len(outputs)} figures")


if __name__ == "__main__":
    main()
