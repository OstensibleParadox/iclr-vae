from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import BUNDLE_ROOT, add_common_args, apply_determinism, disk_free_gb, ensure_run_tree, environment_snapshot, load_config, maybe_skip, now_iso, run_command, run_dir, write_json, write_stage_metadata, write_status

STAGE = "preflight"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for WickDet A100 bundle.")
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    path = run_dir(args.run_id)
    if args.dry_run:
        print(f"[dry-run] {STAGE}: would check CUDA/A100, deterministic settings, HF cache, disk, config")
        return
    ensure_run_tree(path)
    if maybe_skip(args, STAGE, path):
        return
    start = now_iso()
    import torch
    apply_determinism(config)
    env = environment_snapshot()
    errors = []
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"} and not args.allow_cpu:
        errors.append(f"expected Ubuntu x86_64/amd64 machine, got {platform.machine()!r}")
    if config.get("device") != "cuda":
        errors.append(f"config device must be cuda, got {config.get('device')!r}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        errors.append("CUDA is not available")
    if torch.cuda.is_available() and not getattr(torch.version, "cuda", None) and not args.allow_cpu:
        errors.append("torch.version.cuda is unavailable; install a CUDA-enabled PyTorch build")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    if torch.cuda.is_available() and "A100" not in gpu_name and not args.allow_cpu:
        errors.append(f"expected A100 GPU, got {gpu_name}")
    bundled_hf_home = BUNDLE_ROOT / "model_cache" / "hf_cache"
    hf_home = os.environ.get("HF_HOME")
    if not hf_home and bundled_hf_home.exists():
        hf_home = str(bundled_hf_home)
    if not hf_home:
        hf_home = os.path.join(os.environ.get("SCRATCH", str(Path.home())), "hf_cache")
    os.environ["HF_HOME"] = hf_home
    free_gb = disk_free_gb(path.parent)
    if free_gb < 100.0 and not args.allow_cpu:
        errors.append(f"expected at least 100 GB free disk for final outputs, got {free_gb:.1f} GB")
    nvidia_smi = run_command(["nvidia-smi"])
    report = {
        "environment": env,
        "machine": platform.machine(),
        "gpu_name": gpu_name,
        "hf_home": hf_home,
        "disk_free_gb": free_gb,
        "nvidia_smi": nvidia_smi,
        "errors": errors,
        "deterministic_backend_settings": config.get("deterministic_backend_settings", []),
    }
    report_path = path / "manifests" / "preflight.json"
    write_json(report_path, report)
    status_lines = [f"# WickDet run {args.run_id}", "", f"PREFLIGHT_PASS = {str(not errors).lower()}"]
    if errors:
        status_lines.append("FINAL_PASS = false")
        status_lines.extend(f"- {e}" for e in errors)
        write_status(path, status_lines)
        write_stage_metadata(run_path=path, config=config, stage=STAGE, start_time=start, output_files=[report_path], extra={"errors": errors})
        raise SystemExit("preflight failed; see FINAL_STATUS.md")
    write_status(path, status_lines)
    write_stage_metadata(run_path=path, config=config, stage=STAGE, start_time=start, output_files=[report_path], extra={"errors": []})
    print(f"preflight ok: {gpu_name}; free disk {free_gb:.1f} GB")


if __name__ == "__main__":
    main()
