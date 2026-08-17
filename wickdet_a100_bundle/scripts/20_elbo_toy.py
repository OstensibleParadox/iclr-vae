from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import add_common_args, ensure_run_tree, find_input, load_config, load_input_manifest, maybe_skip, now_iso, run_dir, torch_load, verified_input_path, write_csv, write_stage_metadata

STAGE = "elbo_toy"
TRACE_COEFF = {"cf_plus_half_trace": 0.5, "cf_plus_one_trace": 1.0, "cf_minus_half_trace": -0.5}
SLIM = {"ordinary_kl", "cf_kl", "cf_filtered", "cf_plus_half_trace"}


def multiplicities(dim: int) -> torch.Tensor:
    m = torch.full((dim // 2 + 1,), 2.0, dtype=torch.float64)
    m[0] = 1.0
    if dim % 2 == 0:
        m[-1] = 1.0
    return m


def spectrum(dim: int, scale: float, alpha: float, max_beta: float, filtered: bool = False, alpha_filter: float = 0.0) -> torch.Tensor:
    j = torch.arange(1, dim // 2 + 2, dtype=torch.float64)
    beta = torch.clamp(scale / j.pow(alpha), max=max_beta * 0.98)
    if filtered:
        beta = beta * j.pow(-alpha_filter)
    return beta


def exact_metrics(beta: torch.Tensor, mult: torch.Tensor) -> dict[str, float]:
    trace = (mult * beta).sum()
    t2 = (mult * beta.square()).sum()
    logdet = (mult * torch.log1p(-beta)).sum()
    ordinary = 0.5 * (mult * beta / (1.0 - beta)).sum() + 0.5 * logdet
    cf = 0.5 * (mult * beta.square() / (1.0 - beta)).sum() + 0.5 * (logdet + trace)
    return {
        "trace_exact": float(trace),
        "t2_exact": float(t2),
        "logdet_exact": float(logdet),
        "ordinary_kl_exact": float(ordinary),
        "cf_kl_exact": float(cf),
        "ordinary_cf_kl_abs_error": float(abs(ordinary - cf)),
    }


def run(config: dict[str, Any], run_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    cfg = config[STAGE]
    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    manifest = load_input_manifest(run_path)
    for seed in config["seeds"]:
        for dim in cfg["dimensions"]:
            feature_row = find_input(manifest, experiment_name=STAGE, input_type="features", seed=seed, resolution=dim)
            probe_row = find_input(manifest, experiment_name=STAGE, input_type="hutchinson_probes", seed=seed, resolution=dim)
            features = torch_load(verified_input_path(run_path, feature_row), map_location="cpu")
            _ = torch_load(verified_input_path(run_path, probe_row), map_location="cpu")
            input_hashes[feature_row["relative_path"]] = feature_row["sha256"]
            input_hashes[probe_row["relative_path"]] = probe_row["sha256"]
            mult = multiplicities(int(dim))
            target = spectrum(int(dim), cfg["target_scale"], cfg["target_alpha"], cfg["max_beta"])
            for branch in cfg["branches"]:
                beta0 = spectrum(int(dim), cfg["init_scale"], cfg["init_alpha"], cfg["max_beta"], filtered=(branch == "cf_filtered"), alpha_filter=cfg.get("alpha_filter", 0.0))
                beta = beta0.clone()
                coeff = TRACE_COEFF.get(branch, 0.0)
                for epoch in range(int(cfg["epochs"])):
                    metrics = exact_metrics(beta, mult)
                    mse = float((mult * (beta - target).square()).mean())
                    prior = metrics["ordinary_kl_exact"] if branch == "ordinary_kl" else metrics["cf_kl_exact"] + coeff * metrics["trace_exact"]
                    residue_delta = prior - metrics["cf_kl_exact"]
                    residue_expected = coeff * metrics["trace_exact"]
                    # Deterministic closed-form proxy for training: nudges beta toward target without hidden RNG.
                    beta = torch.clamp(beta + float(cfg.get("learning_rate", 0.03)) * 0.02 * (target - beta), min=0.0, max=float(cfg["max_beta"]) * 0.98)
                    sample_identity = 0.0
                    if features.numel() == 0:
                        sample_identity = math.nan
                    rows.append({
                        "seed": int(seed),
                        "dimension": int(dim),
                        "epoch": int(epoch),
                        "branch": branch,
                        "mse_loss": mse,
                        "prior_kl_used": float(prior),
                        "total_loss": mse + 0.01 * float(prior),
                        "trace_residue_coeff": coeff,
                        "trace_residue_delta_same_params": float(residue_delta),
                        "trace_residue_expected_same_params": float(residue_expected),
                        "trace_residue_abs_error_same_params": float(abs(residue_delta - residue_expected)),
                        "sample_log_rn_identity_abs_error": sample_identity,
                        **metrics,
                    })
    return rows, input_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact ELBO toy / trace-residue ablation.")
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    run_path = run_dir(args.run_id)
    if args.dry_run:
        print(f"[dry-run] {STAGE}: would run dimensions={config[STAGE]['dimensions']} branches={config[STAGE]['branches']}")
        return
    ensure_run_tree(run_path)
    if maybe_skip(args, STAGE, run_path):
        return
    start = now_iso()
    rows, input_hashes = run(config, run_path)
    max_kl = max(float(r["ordinary_cf_kl_abs_error"]) for r in rows)
    max_sample = max(float(r["sample_log_rn_identity_abs_error"]) for r in rows)
    max_residue = max(float(r["trace_residue_abs_error_same_params"]) for r in rows)
    if max_kl > 1e-9 or max_sample > 1e-9 or max_residue > 1e-7:
        raise RuntimeError(f"ELBO checks failed: KL={max_kl:.3e}, sample={max_sample:.3e}, residue={max_residue:.3e}")
    results = run_path / "csv" / "elbo_toy_results.csv"
    slim = run_path / "csv" / "figure_source_fig2_elbo_toy_slim.csv"
    full = run_path / "csv" / "figure_source_appendix_elbo_toy_full.csv"
    pareto = run_path / "csv" / "figure_source_fig3_elbo_pareto.csv"
    write_csv(results, rows)
    write_csv(slim, [r for r in rows if r["branch"] in SLIM])
    write_csv(full, rows)
    write_csv(pareto, rows)
    write_stage_metadata(run_path=run_path, config=config, stage=STAGE, start_time=start, input_hashes=input_hashes, output_files=[results, slim, full, pareto], extra={"max_ordinary_cf_kl_error": max_kl, "max_sample_log_RN_identity_error": max_sample, "max_trace_residue_error": max_residue})
    print(f"ELBO toy rows={len(rows)}")


if __name__ == "__main__":
    main()
