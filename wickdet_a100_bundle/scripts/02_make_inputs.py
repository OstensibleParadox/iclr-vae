from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import add_common_args, apply_determinism, ensure_run_tree, load_config, maybe_skip, now_iso, run_dir, sha256_file, write_csv, write_json, write_stage_metadata

STAGE = "make_inputs"


def save_tensor(run_path: Path, rel: str, tensor: torch.Tensor) -> dict[str, Any]:
    path = run_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.cpu(), path)
    return {"relative_path": rel, "sha256": sha256_file(path), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def save_json_input(run_path: Path, rel: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = run_path / rel
    write_json(path, payload)
    return {"relative_path": rel, "sha256": sha256_file(path), "shape": "json", "dtype": "json"}


def manifest_row(saved: dict[str, Any], experiment_name: str, seed: int | str, resolution: int | str, input_type: str, notes: str) -> dict[str, Any]:
    return {
        "relative_path": saved["relative_path"],
        "sha256": saved["sha256"],
        "seed": seed,
        "experiment_name": experiment_name,
        "resolution": resolution,
        "input_type": input_type,
        "dtype": saved["dtype"],
        "shape": json.dumps(saved["shape"]),
        "notes": notes,
    }


def structured_pattern(resolution: int) -> torch.Tensor:
    yy = torch.linspace(-math.pi, math.pi, resolution).unsqueeze(1).expand(resolution, resolution)
    xx = torch.linspace(-math.pi, math.pi, resolution).unsqueeze(0).expand(resolution, resolution)
    pattern = 0.5 * (torch.sin(2 * yy) + torch.cos(3 * xx))
    return torch.stack([pattern, pattern.roll(resolution // 4, 0), pattern.roll(resolution // 4, 1)], dim=0).unsqueeze(0).clamp(-1, 1).float()


def make_inputs(config: dict[str, Any], run_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cs = config["controlled_spectra"]
    regimes = []
    for name in cs["regimes"]:
        params = dict(cs.get("regime_parameters", {}).get(name, {}))
        params.setdefault("exponent_alpha", {"S1": 1.25, "S2_not_S1": 0.75, "outside_S2": 0.35}[name])
        params.setdefault("amplitude", 1.0)
        params["name"] = name
        regimes.append(params)
    saved = save_json_input(run_path, "inputs/controlled_spectra/spectral_parameters.json", {"dimensions": cs["dimensions"], "regimes": regimes})
    rows.append(manifest_row(saved, "controlled_spectra", "deterministic", "multiple", "spectral_parameters", "Controlled spectra dimensions and parameters."))

    et = config["elbo_toy"]
    for seed in config["seeds"]:
        for dim in et["dimensions"]:
            gen = torch.Generator().manual_seed(int(seed) * 1_000_000 + int(dim))
            features = torch.randn(int(et.get("batch_size", 512)), int(dim), generator=gen, dtype=torch.float32)
            probes = torch.randint(0, 2, (int(et.get("num_probe_vectors", 64)), int(dim)), generator=gen, dtype=torch.int8).float().mul_(2).sub_(1)
            base = f"inputs/elbo_toy/dim_{dim}/seed_{seed}"
            f = save_tensor(run_path, f"{base}/features.pt", features)
            p = save_tensor(run_path, f"{base}/probes.pt", probes)
            rows.append(manifest_row(f, "elbo_toy", seed, dim, "features", "Fixed Gaussian feature batch."))
            rows.append(manifest_row(p, "elbo_toy", seed, dim, "hutchinson_probes", "Fixed Rademacher probes."))

    fno = config["fno"]
    for seed in fno.get("seeds", config["seeds"]):
        for resolution in fno["resolutions"]:
            train_size = int(fno["train_size"])
            val_size = int(fno["val_size"])
            base = f"inputs/fno/resolution_{resolution}/seed_{seed}"
            train_indices = torch.arange(train_size, dtype=torch.int64)
            val_indices = torch.arange(train_size, train_size + val_size, dtype=torch.int64)
            sample_seeds = torch.tensor([int(seed) * 1_000_000 + int(resolution) * 10_000 + i for i in range(train_size + val_size)], dtype=torch.int64)
            for name, tensor in [("train_indices", train_indices), ("val_indices", val_indices), ("sample_seeds", sample_seeds)]:
                s = save_tensor(run_path, f"{base}/{name}.pt", tensor)
                rows.append(manifest_row(s, "fno", seed, resolution, name, "Fixed FNO data materialization input."))

    sd = config["sdvae"]
    for seed in config["seeds"]:
        for resolution in sd["resolutions"]:
            gen = torch.Generator().manual_seed(int(seed) * 1_000_000 + int(resolution) * 1000)
            tensors = {
                "white_noise": torch.randn(1, 3, int(resolution), int(resolution), generator=gen, dtype=torch.float32).clamp(-1, 1),
                "constant_gray": torch.zeros(1, 3, int(resolution), int(resolution), dtype=torch.float32),
                "structured_sinusoidal": structured_pattern(int(resolution)),
            }
            for input_type in sd["input_types"]:
                base = f"inputs/sdvae/resolution_{resolution}/seed_{seed}"
                s = save_tensor(run_path, f"{base}/{input_type}.pt", tensors[input_type])
                rows.append(manifest_row(s, "sdvae", seed, resolution, input_type, "Fixed SD-VAE encoder input."))
                max_probes = max(int(x) for x in sd.get("hutchinson_probe_counts", [sd.get("hutchinson_probes", 50)]))
                probe_seeds = torch.tensor([int(seed) * 10_000_000 + int(resolution) * 10_000 + i for i in range(max_probes)], dtype=torch.int64)
                ps = save_tensor(run_path, f"{base}/{input_type}_hutchinson_probe_seeds.pt", probe_seeds)
                rows.append(manifest_row(ps, "sdvae", seed, resolution, f"{input_type}_hutchinson_probe_seeds", "Fixed Hutchinson probe seeds."))
    return sorted(rows, key=lambda row: row["relative_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Make fixed inputs for WickDet A100 bundle.")
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    run_path = run_dir(args.run_id)
    if args.dry_run:
        print(f"[dry-run] {STAGE}: would write fixed inputs and input_manifest.csv under {run_path}")
        return
    ensure_run_tree(run_path)
    if maybe_skip(args, STAGE, run_path):
        return
    start = now_iso()
    apply_determinism(config)
    rows = make_inputs(config, run_path)
    manifest = run_path / "manifests" / "input_manifest.csv"
    write_csv(manifest, rows, ["relative_path", "sha256", "seed", "experiment_name", "resolution", "input_type", "dtype", "shape", "notes"])
    write_stage_metadata(run_path=run_path, config=config, stage=STAGE, start_time=start, output_files=[manifest], extra={"input_rows": len(rows)})
    print(f"wrote {len(rows)} input manifest rows")


if __name__ == "__main__":
    main()
