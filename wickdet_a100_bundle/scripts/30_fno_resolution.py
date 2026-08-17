from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import add_common_args, apply_determinism, ensure_run_tree, find_input, load_config, load_input_manifest, maybe_skip, now_iso, run_dir, torch_load, verified_input_path, write_csv, write_stage_metadata
import fno_core

STAGE = "fno_resolution"


def choose_device(args: Any) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if args.allow_cpu:
        return torch.device("cpu")
    raise SystemExit("CUDA required for FNO final run")


def make_config(config: dict[str, Any], device: torch.device) -> fno_core.ExperimentConfig:
    cfg = config["fno"]
    return fno_core.ExperimentConfig(
        resolutions=tuple(int(x) for x in cfg["resolutions"]), train_samples=int(cfg["train_size"]), val_samples=int(cfg["val_size"]),
        epochs=int(cfg["epochs"]), batch_size=int(cfg["batch_size"]), width=int(cfg["width"]), learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)), max_beta=float(cfg["max_beta"]), alpha_filter=float(cfg["alpha_filter"]),
        noise_lr_penalty_weight=float(cfg["noise_lr_penalty_weight"]), jacobi_iterations=int(cfg["jacobi_iterations"]),
        jacobi_relaxation=float(cfg["jacobi_relaxation"]), modes_divisor=int(cfg["modes_divisor"]), seed_repeats=len(cfg.get("seeds", config["seeds"])),
        resolution_subset=tuple(int(x) for x in cfg.get("aggregate_resolutions", [128, 256])), t2_norm_mode=str(cfg.get("t2_norm_mode", "per_mode")),
        pareto_x_metric="t2_norm", device=str(device),
    )


def final_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device if device.type == "cuda" else "cpu")
    g.manual_seed(int(seed))
    return g


def load_tensor_input(run_path: Path, manifest: list[dict[str, str]], resolution: int, seed: int, input_type: str) -> tuple[torch.Tensor, dict[str, str]]:
    row = find_input(manifest, experiment_name="fno", input_type=input_type, seed=seed, resolution=resolution)
    return torch_load(verified_input_path(run_path, row), map_location="cpu").to(torch.long), {row["relative_path"]: row["sha256"]}


def materialize(run_path: Path, manifest: list[dict[str, str]], resolution: int, seed: int, cfg: fno_core.ExperimentConfig, device: torch.device):
    train_idx, hashes = load_tensor_input(run_path, manifest, resolution, seed, "train_indices")
    val_idx, h2 = load_tensor_input(run_path, manifest, resolution, seed, "val_indices")
    sample_seeds, h3 = load_tensor_input(run_path, manifest, resolution, seed, "sample_seeds")
    hashes.update(h2); hashes.update(h3)
    coords = fno_core.coordinate_channels(resolution, device)
    rhs = fno_core.deterministic_rhs(resolution, device)
    xs, ys = [], []
    with torch.no_grad():
        for sample_seed in sample_seeds.cpu().tolist():
            logk = fno_core.sample_lowpass_logk(resolution, generator(int(sample_seed), device), device=device)
            u = fno_core.solve_periodic_darcy(logk, rhs, iterations=cfg.jacobi_iterations, relaxation=cfg.jacobi_relaxation)
            xs.append(logk); ys.append(u)
    logk_all = torch.stack(xs)
    y_all = torch.stack(ys)
    coord = coords.unsqueeze(0).expand(len(sample_seeds), -1, -1, -1)
    X = torch.cat((logk_all.unsqueeze(1), coord), dim=1).float()
    y = y_all.unsqueeze(1).float()
    return X[train_idx.to(device)], y[train_idx.to(device)], X[val_idx.to(device)], y[val_idx.to(device)], hashes


def mean_std(vals: Iterable[float]) -> tuple[float, float, int]:
    v = list(float(x) for x in vals)
    if not v:
        return math.nan, math.nan, 0
    t = torch.tensor(v, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False)), int(t.numel())


def aggregate(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in metrics:
            groups[(int(row["resolution"]), str(row["baseline"]), int(row["epoch"]), metric)].append(float(row[metric]))
    out = []
    for (res, branch, epoch, metric), vals in sorted(groups.items()):
        mean, std, n = mean_std(vals)
        out.append({"resolution": res, "baseline": branch, "epoch": epoch, "metric": metric, "mean": mean, "std": std, "n": n})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="FNO resolution-scaling experiment.")
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    run_path = run_dir(args.run_id)
    if args.dry_run:
        print(f"[dry-run] {STAGE}: would run FNO grid resolutions={config['fno']['resolutions']} seeds={config['fno'].get('seeds', config['seeds'])} branches={config['fno']['branches']}")
        return
    ensure_run_tree(run_path)
    if maybe_skip(args, STAGE, run_path):
        return
    start = now_iso()
    apply_determinism(config)
    device = choose_device(args)
    cfg = make_config(config, device)
    fno_core.set_seed = final_seed
    manifest = load_input_manifest(run_path)
    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for resolution in cfg.resolutions:
        for seed in [int(s) for s in config["fno"].get("seeds", config["seeds"])]:
            train_x, train_y, val_x, val_y, hashes = materialize(run_path, manifest, resolution, seed, cfg, device)
            input_hashes.update(hashes)
            for branch in config["fno"]["branches"]:
                model_seed = seed + resolution * 1000
                branch_rows = fno_core.run_baseline(resolution, branch, train_x, train_y, val_x, val_y, config=cfg, device=device, seed=model_seed)
                for r in branch_rows:
                    rr = dict(r)
                    rr["seed"] = seed
                    rr["model_seed"] = model_seed
                    rows.append(rr)
    expected = len(cfg.resolutions) * len(config["fno"].get("seeds", config["seeds"])) * len(config["fno"]["branches"]) * int(config["fno"]["epochs"])
    if len(rows) != expected:
        raise RuntimeError(f"FNO row count mismatch: expected {expected}, got {len(rows)}")
    results = run_path / "csv" / "fno_results.csv"
    slim = run_path / "csv" / "figure_source_fig5_fno_timeseries_slim.csv"
    pareto = run_path / "csv" / "figure_source_fig4_fno_pareto_normT2.csv"
    full = run_path / "csv" / "figure_source_appendix_fno_timeseries_full.csv"
    write_csv(results, rows)
    write_csv(slim, aggregate(rows, ["val_relative_l2", "train_mse", "grad_norm", "t2_norm", "t2_continuum", "high_frequency_energy_ratio"]))
    write_csv(full, aggregate(rows, ["val_relative_l2", "train_mse", "grad_norm", "t2_exact", "t2_discrete", "t2_norm", "t2_continuum", "high_frequency_energy_ratio"]))
    write_csv(pareto, aggregate(rows, ["t2_norm", "val_relative_l2"]))
    write_stage_metadata(run_path=run_path, config=config, stage=STAGE, start_time=start, input_hashes=input_hashes, output_files=[results, slim, pareto, full], extra={"row_count": len(rows), "pareto_x_metric": "t2_norm"})
    print(f"FNO rows={len(rows)}")


if __name__ == "__main__":
    main()
