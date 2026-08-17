from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import add_common_args, ensure_run_tree, find_input, load_config, load_input_manifest, maybe_skip, now_iso, run_dir, verified_input_path, write_csv, write_stage_metadata

STAGE = "controlled_spectra"


def beta_sequence(n: int, scale: float, alpha: float) -> torch.Tensor:
    j = torch.arange(1, n + 1, dtype=torch.float64)
    beta = 0.45 * scale / j.pow(alpha)
    if torch.any(beta >= 1.0):
        raise RuntimeError("beta >= 1 in controlled spectrum")
    return beta


def compute_rows(params: dict[str, Any]) -> list[dict[str, Any]]:
    dims = [int(x) for x in params["dimensions"]]
    cutoffs = sorted(set(dims + torch.logspace(1, math.log10(max(dims)), steps=70).round().to(torch.int64).tolist()))
    rows: list[dict[str, Any]] = []
    for regime in params["regimes"]:
        beta = beta_sequence(max(cutoffs), float(regime.get("amplitude", 1.0)), float(regime["exponent_alpha"]))
        trace = beta.cumsum(0)
        t2 = beta.square().cumsum(0)
        logdet = torch.log1p(-beta).cumsum(0)
        det2 = (torch.log1p(-beta) + beta).cumsum(0)
        for cutoff in cutoffs:
            idx = int(cutoff) - 1
            tr = float(trace[idx])
            t2v = float(t2[idx])
            ordinary = 0.5 * tr + 0.5 * float(logdet[idx])
            wick = 0.5 * float(det2[idx])
            rows.append({
                "regime": regime["name"],
                "alpha": float(regime["exponent_alpha"]),
                "cutoff_N": int(cutoff),
                "trace_component": tr,
                "T2_diagnostic": t2v,
                "ordinary_branch_drift": ordinary,
                "wick_det2_finite_part_branch": wick,
                "finite_cutoff_identity_error": abs(ordinary - wick),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled spectra diagnostics.")
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    run_path = run_dir(args.run_id)
    if args.dry_run:
        print(f"[dry-run] {STAGE}: would read fixed spectral params and write CSV/figure source")
        return
    ensure_run_tree(run_path)
    if maybe_skip(args, STAGE, run_path):
        return
    start = now_iso()
    manifest = load_input_manifest(run_path)
    row = find_input(manifest, experiment_name="controlled_spectra", input_type="spectral_parameters", seed="deterministic")
    params = json.loads(verified_input_path(run_path, row).read_text(encoding="utf-8"))
    rows = compute_rows(params)
    max_err = max(float(r["finite_cutoff_identity_error"]) for r in rows)
    if max_err > 1e-10:
        raise RuntimeError(f"controlled spectra identity error too large: {max_err:.3e}")
    results = run_path / "csv" / "controlled_spectra_results.csv"
    figsrc = run_path / "csv" / "figure_source_fig1_phase_diagram.csv"
    write_csv(results, rows)
    write_csv(figsrc, rows)
    write_stage_metadata(run_path=run_path, config=config, stage=STAGE, start_time=start, input_hashes={row["relative_path"]: row["sha256"]}, output_files=[results, figsrc], extra={"max_finite_cutoff_identity_error": max_err})
    print(f"controlled spectra rows={len(rows)}")


if __name__ == "__main__":
    main()
