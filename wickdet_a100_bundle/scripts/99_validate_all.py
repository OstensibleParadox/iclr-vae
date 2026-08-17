from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import add_common_args, ensure_run_tree, input_manifest_path, load_config, now_iso, read_csv, run_dir, sha256_file, stage_is_complete, write_json, write_stage_metadata, write_status

STAGE = "validate_all"
REQUIRED_STAGES = ["preflight", "download_models", "make_inputs", "controlled_spectra", "elbo_toy", "fno_resolution", "plot_all"]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def no_pilot_paths(run_path: Path) -> None:
    offenders = []
    for p in run_path.rglob("*"):
        if "pilot_mps" in str(p):
            offenders.append(str(p))
    require(not offenders, f"pilot_mps path found in final outputs: {offenders[:5]}")


def validate_inputs(run_path: Path) -> None:
    manifest = input_manifest_path(run_path)
    require(manifest.exists(), "missing input_manifest.csv")
    for row in read_csv(manifest):
        path = run_path / row["relative_path"]
        require(path.exists(), f"missing input {path}")
        require(sha256_file(path) == row["sha256"], f"input hash mismatch {path}")


def validate_metadata(run_path: Path) -> None:
    metas = list((run_path / "metadata").glob("*.json"))
    require(metas, "no metadata JSON files")
    for meta in metas:
        payload = json.loads(meta.read_text(encoding="utf-8"))
        require("output_hashes" in payload, f"metadata missing output_hashes: {meta}")
        for rel, sha in payload.get("output_hashes", {}).items():
            path = run_path / rel
            require(path.exists(), f"metadata output missing: {path}")
            require(sha256_file(path) == sha, f"output hash mismatch: {path}")


def validate_elbo(run_path: Path) -> None:
    path = run_path / "csv" / "elbo_toy_results.csv"
    require(path.exists(), "missing ELBO results")
    rows = read_csv(path)
    require(rows, "empty ELBO results")
    max_kl = max(float(r["ordinary_cf_kl_abs_error"]) for r in rows)
    max_sample = max(float(r["sample_log_rn_identity_abs_error"]) for r in rows)
    max_residue = max(float(r["trace_residue_abs_error_same_params"]) for r in rows)
    require(max_kl < 1e-9, f"ELBO ordinary/CF KL identity failed: {max_kl}")
    require(max_sample < 1e-9, f"sample log-RN identity failed: {max_sample}")
    require(max_residue < 1e-7, f"trace-residue sanity failed: {max_residue}")


def validate_fno(run_path: Path, config: dict) -> None:
    path = run_path / "csv" / "fno_results.csv"
    require(path.exists(), "missing FNO results")
    rows = read_csv(path)
    cfg = config["fno"]
    for res in cfg["resolutions"]:
        for seed in cfg.get("seeds", config["seeds"]):
            for branch in cfg["branches"]:
                sub = [r for r in rows if int(r["resolution"]) == int(res) and int(r["seed"]) == int(seed) and r["baseline"] == branch]
                require(len(sub) == int(cfg["epochs"]), f"missing FNO grid res={res} seed={seed} branch={branch}")
                for r in sub:
                    require(float(r["high_frequency_energy_ratio"]) >= 0 and math.isfinite(float(r["high_frequency_energy_ratio"])), "invalid FNO high-frequency ratio")


def validate_sdvae(run_path: Path, config: dict) -> None:
    rows = []
    for path in sorted((run_path / "csv").glob("sdvae_jacobian_*_results.csv")):
        rows.extend(read_csv(path))
    require(rows, "missing SD-VAE results")
    for res in config["sdvae"]["resolutions"]:
        for seed in config["seeds"]:
            for input_type in config["sdvae"]["input_types"]:
                sub = [r for r in rows if int(r["resolution"]) == int(res) and int(r["seed"]) == int(seed) and r["input_type"] == input_type]
                require(len(sub) == 1, f"missing SD-VAE row res={res} seed={seed} input={input_type}")
                require(sub[0]["fit_object"] == "covariance_eigenvalues_lambda_i", "SD-VAE fit object is not covariance eigenvalues")


def validate_figures(run_path: Path) -> None:
    required = ["fig1_phase_diagram.png", "fig2_elbo_toy_slim.png", "fig3_elbo_pareto.png", "fig4_fno_pareto_normT2.png", "fig5_fno_timeseries_slim.png", "fig6_sdvae_operator_ideal_filter.png"]
    for name in required:
        path = run_path / "figures" / name
        require(path.exists(), f"missing final figure: {name}")
        require("pilot_mps" not in str(path), f"pilot path in final figure: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate WickDet A100 final outputs.")
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    run_path = run_dir(args.run_id)
    if args.dry_run:
        print("[dry-run] validate_all checks: figures provenance, CSV/metadata/hash, input hashes, FNO grid, ELBO identities, trace-residue, SD-VAE grid, covariance-eigenvalue fit object, no pilot_mps paths")
        return
    ensure_run_tree(run_path)
    start = now_iso()
    errors: list[str] = []
    try:
        no_pilot_paths(run_path)
        validate_inputs(run_path)
        validate_metadata(run_path)
        validate_elbo(run_path)
        validate_fno(run_path, config)
        validate_sdvae(run_path, config)
        validate_figures(run_path)
        for stage in REQUIRED_STAGES:
            require(stage_is_complete(run_path, stage), f"stage marker incomplete: {stage}")
    except Exception as exc:
        errors.append(str(exc))
    report = {"run_id": args.run_id, "pass": not errors, "errors": errors}
    report_path = run_path / "validation" / "validate_all.json"
    write_json(report_path, report)
    if errors:
        write_status(run_path, [f"# WickDet run {args.run_id}", "", "FINAL_PASS = false", *[f"- {e}" for e in errors]])
        write_stage_metadata(run_path=run_path, config=config, stage=STAGE, start_time=start, output_files=[report_path], extra=report)
        raise SystemExit("validation failed")
    write_status(run_path, [f"# WickDet run {args.run_id}", "", "FINAL_PASS = true", "SMOKE_TEST_PASS = true" if config.get("run_group") == "smoke" else "A100_FINAL_PASS = true"])
    write_stage_metadata(run_path=run_path, config=config, stage=STAGE, start_time=start, output_files=[report_path, run_path / "FINAL_STATUS.md"], extra=report)
    print("validation passed")


if __name__ == "__main__":
    main()
