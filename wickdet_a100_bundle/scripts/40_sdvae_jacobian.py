from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import BUNDLE_ROOT, add_common_args, apply_determinism, ensure_run_tree, find_input, load_config, load_input_manifest, maybe_skip, now_iso, run_dir, torch_load, verified_input_path, write_csv, write_stage_metadata

STAGE_PREFIX = "sdvae_jacobian"


def choose_device(args: Any) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if args.allow_cpu:
        return torch.device("cpu")
    raise SystemExit("CUDA required for SD-VAE Jacobian final run")


def stage_name(which: str) -> str:
    return f"{STAGE_PREFIX}_{which}"


def selected_resolutions(config: dict[str, Any], which: str) -> list[int]:
    available = [int(x) for x in config["sdvae"]["resolutions"]]
    if which == "64_128":
        return [x for x in available if x in {64, 128}]
    if which == "256":
        return [x for x in available if x == 256]
    if which == "512":
        return [x for x in available if x == 512]
    if which == "smoke":
        return available[:1]
    raise ValueError(f"unknown SD-VAE stage {which}")


def load_vae(config: dict[str, Any], device: torch.device) -> nn.Module:
    from diffusers import AutoencoderKL
    os_mod = __import__("os")
    bundled_hf_home = BUNDLE_ROOT / "model_cache" / "hf_cache"
    if "HF_HOME" not in os_mod.environ and bundled_hf_home.exists():
        os_mod.environ["HF_HOME"] = str(bundled_hf_home)
        os_mod.environ.setdefault("HF_HUB_OFFLINE", "1")
    token = os_mod.environ.get("HF_TOKEN")
    kwargs: dict[str, Any] = {"torch_dtype": torch.float32}
    if token:
        kwargs["token"] = token
    if os_mod.environ.get("HF_HUB_OFFLINE") == "1" or bundled_hf_home.exists():
        kwargs["local_files_only"] = True
    vae = AutoencoderKL.from_pretrained(config["sdvae"]["model"], **kwargs)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


def encode_mean(vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return vae.encode(x).latent_dist.mean


def load_case(run_path: Path, manifest: list[dict[str, str]], seed: int, resolution: int, input_type: str, device: torch.device):
    xrow = find_input(manifest, experiment_name="sdvae", input_type=input_type, seed=seed, resolution=resolution)
    prow = find_input(manifest, experiment_name="sdvae", input_type=f"{input_type}_hutchinson_probe_seeds", seed=seed, resolution=resolution)
    x = torch_load(verified_input_path(run_path, xrow), map_location="cpu").to(device=device, dtype=torch.float32)
    probe_seeds = torch_load(verified_input_path(run_path, prow), map_location="cpu").to(torch.long)
    return x, probe_seeds, {xrow["relative_path"]: xrow["sha256"], prow["relative_path"]: prow["sha256"]}


def compute_row_shards(fn: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, latent_dim: int, row_chunk_size: int, shard_dir: Path) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pixel_dim = int(x0.numel())
    paths: list[Path] = []
    for start in range(0, latent_dim, row_chunk_size):
        end = min(start + row_chunk_size, latent_dim)
        rows = torch.empty((end - start, pixel_dim), dtype=torch.float32, device="cpu")
        for offset, row_idx in enumerate(range(start, end)):
            x_req = x0.detach().clone().requires_grad_(True)
            y = fn(x_req).reshape(-1)
            cotangent = torch.zeros_like(y)
            cotangent[row_idx] = 1.0
            grad = torch.autograd.grad(y, x_req, grad_outputs=cotangent, retain_graph=False, create_graph=False)[0]
            rows[offset] = grad.detach().reshape(-1).cpu()
            del x_req, y, cotangent, grad
        path = shard_dir / f"rows_{start:06d}_{end:06d}.pt"
        torch.save({"row_start": start, "row_end": end, "pixel_dim": pixel_dim, "rows": rows}, path)
        paths.append(path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return paths


def load_shard(path: Path, device: torch.device):
    payload = torch_load(path, map_location="cpu")
    return int(payload["row_start"]), int(payload["row_end"]), payload["rows"].to(device=device, dtype=torch.float32)


def streamed_gram(shards: list[Path], latent_dim: int, device: torch.device) -> torch.Tensor:
    gram = torch.empty((latent_dim, latent_dim), dtype=torch.float64, device="cpu")
    for i, left in enumerate(shards):
        ls, le, lrows = load_shard(left, device)
        for right in shards[i:]:
            rs, re, rrows = load_shard(right, device)
            block = lrows @ rrows.T
            gram[ls:le, rs:re] = block.detach().cpu().to(torch.float64)
            if left != right:
                gram[rs:re, ls:le] = block.detach().T.cpu().to(torch.float64)
            del rrows, block
        del lrows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return gram


def eigvals_desc(gram: torch.Tensor, device: torch.device) -> torch.Tensor:
    eig_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.linalg.eigvalsh(gram.to(eig_device)).detach().cpu().clamp_min(0).flip(0).to(torch.float64)


def fit_tail(eigs: torch.Tensor, start_fraction: float, end_fraction: float) -> dict[str, float | int]:
    positive = eigs[eigs > max(float(eigs.max()) * 1e-12, 1e-30)]
    n = int(positive.numel())
    if n < 8:
        return {"alpha": math.nan, "fit_start_rank": 0, "fit_end_rank": 0, "fit_r2": math.nan}
    start = max(1, int(start_fraction * n))
    end = min(n, max(start + 5, int(end_fraction * n)))
    ranks = torch.arange(1, n + 1, dtype=torch.float64)
    x = torch.log(ranks[start:end])
    y = torch.log(positive[start:end])
    A = torch.stack([torch.ones_like(x), -x], dim=1)
    sol = torch.linalg.lstsq(A, y).solution
    pred = A @ sol
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-30)
    return {"alpha": float(sol[1]), "fit_start_rank": start + 1, "fit_end_rank": end, "fit_r2": float(r2)}


def alpha_bootstrap(eigs: torch.Tensor, fit_start: int, fit_end: int, samples: int, seed: int) -> tuple[float, float]:
    if fit_start <= 0 or fit_end <= fit_start or samples <= 0:
        return math.nan, math.nan
    vals = eigs[fit_start - 1:fit_end]
    ranks = torch.arange(fit_start, fit_end + 1, dtype=torch.float64)
    valid = vals > 0
    vals = vals[valid]
    ranks = ranks[valid]
    if int(vals.numel()) < 5:
        return math.nan, math.nan
    x = torch.log(ranks); y = torch.log(vals)
    g = torch.Generator().manual_seed(int(seed))
    alphas = []
    for _ in range(samples):
        idx = torch.randint(0, int(vals.numel()), (int(vals.numel()),), generator=g)
        A = torch.stack([torch.ones_like(x[idx]), -x[idx]], dim=1)
        sol = torch.linalg.lstsq(A, y[idx]).solution
        alphas.append(float(sol[1]))
    alphas.sort()
    lo = alphas[int(0.025 * (len(alphas) - 1))]
    hi = alphas[int(0.975 * (len(alphas) - 1))]
    return lo, hi


def hutchinson(fn: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, probe_seeds: torch.Tensor, count: int, eps: float) -> dict[str, float | int]:
    tr, t2 = [], []
    for seed in probe_seeds[:count].cpu().tolist():
        gen = torch.Generator(device=x0.device if x0.device.type == "cuda" else "cpu").manual_seed(int(seed))
        z = torch.randint(0, 2, x0.shape, generator=gen, device=x0.device).to(x0.dtype).mul_(2).sub_(1)
        with torch.no_grad():
            jz = (fn(x0 + eps * z).reshape(-1) - fn(x0 - eps * z).reshape(-1)) / (2 * eps)
        tr.append(float(jz.square().sum()))
        xb = x0.detach().clone().requires_grad_(True)
        y = fn(xb).reshape(-1)
        jt = torch.autograd.grad(y, xb, grad_outputs=jz.detach(), retain_graph=False, create_graph=False)[0]
        t2.append(float(jt.square().sum()))
        del z, jz, xb, y, jt
    def ms(values: list[float]) -> tuple[float, float]:
        tt = torch.tensor(values, dtype=torch.float64)
        return float(tt.mean()), float(tt.std(unbiased=False))
    trm, trs = ms(tr); t2m, t2s = ms(t2)
    return {"hutchinson_trace_mean": trm, "hutchinson_trace_std": trs, "hutchinson_T2_mean": t2m, "hutchinson_T2_std": t2s, "hutchinson_num_probes": int(count)}


def run_case(run_path: Path, config: dict[str, Any], vae: nn.Module, manifest: list[dict[str, str]], seed: int, resolution: int, input_type: str, device: torch.device):
    cfg = config["sdvae"]
    x0, probe_seeds, hashes = load_case(run_path, manifest, seed, resolution, input_type, device)
    latent_dim = 4 * (resolution // 8) * (resolution // 8)
    pixel_dim = 3 * resolution * resolution
    fn = lambda x: encode_mean(vae, x)
    shard_dir = run_path / "spectra" / "row_shards" / f"res_{resolution}" / f"seed_{seed}" / input_type
    shards = compute_row_shards(fn, x0, latent_dim, int(cfg["row_chunk_size"]), shard_dir)
    gram = streamed_gram(shards, latent_dim, device)
    eigs = eigvals_desc(gram, device)
    eig_pt = run_path / "spectra" / f"eigenvalues_res{resolution}_seed{seed}_{input_type}.pt"
    eig_csv = run_path / "spectra" / f"eigenvalues_res{resolution}_seed{seed}_{input_type}.csv"
    torch.save(eigs, eig_pt)
    write_csv(eig_csv, [{"rank": i + 1, "lambda_i": float(v)} for i, v in enumerate(eigs.tolist())])
    trace = float(eigs.sum()); t2 = float(eigs.square().sum()); lam1 = float(eigs[0]) if eigs.numel() else math.nan
    fit = fit_tail(eigs, float(cfg.get("fit_start_fraction", 0.05)), float(cfg.get("fit_end_fraction", 0.8)))
    ci_lo, ci_hi = alpha_bootstrap(eigs, int(fit["fit_start_rank"]), int(fit["fit_end_rank"]), int(cfg.get("bootstrap_samples", 500)), seed + resolution * 1000)
    h = hutchinson(fn, x0, probe_seeds, int(cfg.get("hutchinson_probes", max(cfg.get("hutchinson_probe_counts", [50])))), float(cfg.get("hutchinson_eps", 1e-3)))
    h["hutchinson_trace_rel_error"] = abs(float(h["hutchinson_trace_mean"]) - trace) / (trace + 1e-30)
    h["hutchinson_T2_rel_error"] = abs(float(h["hutchinson_T2_mean"]) - t2) / (t2 + 1e-30)
    row = {
        "seed": seed, "resolution": resolution, "input_type": input_type, "method": "vjp_row_chunks_streamed_gram_eigvalsh",
        "pixel_dim": pixel_dim, "latent_dim": latent_dim, "fit_object": "covariance_eigenvalues_lambda_i", "S2_threshold_for_lambda_alpha": 0.5,
        "trace_G": trace, "T2_G": t2, "T2_over_trace2": t2 / (trace * trace + 1e-30), "lambda1_over_trace": lam1 / (trace + 1e-30),
        "effective_rank": trace * trace / (t2 + 1e-30), "eigenvalue_tail_alpha": fit["alpha"], "fit_window_start_rank": fit["fit_start_rank"],
        "fit_window_end_rank": fit["fit_end_rank"], "eigenvalue_tail_fit_r2": fit["fit_r2"], "bootstrap_alpha_ci_low": ci_lo, "bootstrap_alpha_ci_high": ci_hi,
        "spectrum_pt": eig_pt.relative_to(run_path).as_posix(), "spectrum_csv": eig_csv.relative_to(run_path).as_posix(), "row_shard_dir": shard_dir.relative_to(run_path).as_posix(),
        "input_hashes": json.dumps(hashes, sort_keys=True, separators=(",", ":")), **h,
    }
    return row, hashes, [eig_pt, eig_csv, *shards]


def main() -> None:
    parser = argparse.ArgumentParser(description="SD-VAE Jacobian Gram spectrum.")
    add_common_args(parser)
    parser.add_argument("--sdvae-stage", default="64_128", choices=["64_128", "256", "512", "smoke"])
    args = parser.parse_args()
    config = load_config(args.config)
    run_path = run_dir(args.run_id)
    stage = stage_name(args.sdvae_stage)
    resolutions = selected_resolutions(config, args.sdvae_stage)
    if args.dry_run:
        print(f"[dry-run] {stage}: would run SD-VAE row-shard eigvalsh for resolutions={resolutions} inputs={config['sdvae']['input_types']}")
        return
    ensure_run_tree(run_path)
    if maybe_skip(args, stage, run_path):
        return
    start = now_iso()
    apply_determinism(config)
    device = choose_device(args)
    vae = load_vae(config, device)
    manifest = load_input_manifest(run_path)
    rows: list[dict[str, Any]] = []
    all_hashes: dict[str, str] = {}
    outputs: list[Path] = []
    for resolution in resolutions:
        for seed in [int(s) for s in config["seeds"]]:
            for input_type in config["sdvae"]["input_types"]:
                row, hashes, files = run_case(run_path, config, vae, manifest, seed, int(resolution), str(input_type), device)
                rows.append(row); all_hashes.update(hashes); outputs.extend(files)
    results = run_path / "csv" / f"sdvae_jacobian_{args.sdvae_stage}_results.csv"
    write_csv(results, rows)
    outputs.append(results)
    write_stage_metadata(run_path=run_path, config=config, stage=stage, start_time=start, input_hashes=all_hashes, output_files=outputs, extra={"row_count": len(rows), "fit_object": "covariance_eigenvalues_lambda_i"})
    print(f"SD-VAE rows={len(rows)} stage={stage}")


if __name__ == "__main__":
    main()
