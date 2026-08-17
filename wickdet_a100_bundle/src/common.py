from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BUNDLE_ROOT / "configs" / "a100_final.yaml"
SMOKE_CONFIG = BUNDLE_ROOT / "configs" / "smoke.yaml"
STAGE_DIRNAME = "stages"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_dir(run_id: str) -> Path:
    return BUNDLE_ROOT / "outputs" / run_id


def ensure_run_tree(path: Path) -> None:
    for rel in ["csv", "figures", "spectra", "inputs", "manifests", "metadata", "logs", "validation", f"manifests/{STAGE_DIRNAME}"]:
        (path / rel).mkdir(parents=True, exist_ok=True)


def run_command(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, cwd=BUNDLE_ROOT, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return f"[command not found] {args[0]}"
    return ((proc.stdout or "") + (proc.stderr or "")).strip()


def git_commit() -> str:
    return run_command(["git", "rev-parse", "HEAD"]) or "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Config is not a mapping: {path}")
    return loaded


def torch_load(path: Path, *, map_location: Any | None = None) -> Any:
    if torch is None:
        raise RuntimeError("torch is required")
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def apply_determinism(config: dict[str, Any]) -> None:
    if torch is None:
        raise RuntimeError("torch is required")
    if bool(config.get("tf32", True)) is False and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if bool(config.get("deterministic", False)):
        torch.use_deterministic_algorithms(True, warn_only=True)


def environment_snapshot() -> dict[str, Any]:
    gpu_name = "unavailable"
    cuda_available = False
    cuda_version = "unavailable"
    torch_version = "unavailable"
    if torch is not None:
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_version = str(getattr(torch.version, "cuda", "unavailable"))
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "git_commit": git_commit(),
        "hf_home": os.environ.get("HF_HOME", ""),
        "scratch": os.environ.get("SCRATCH", ""),
    }


def output_hashes(paths: Iterable[Path], root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        if path.exists() and path.is_file():
            out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def stage_marker(path: Path, stage: str) -> Path:
    return path / "manifests" / STAGE_DIRNAME / f"{stage}.done.json"


def stage_is_complete(path: Path, stage: str) -> bool:
    marker = stage_marker(path, stage)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    for rel, expected in payload.get("output_hashes", {}).items():
        candidate = path / rel
        if not candidate.exists() or sha256_file(candidate) != expected:
            return False
    return True


def write_stage_metadata(
    *,
    run_path: Path,
    config: dict[str, Any],
    stage: str,
    start_time: str,
    input_hashes: dict[str, str] | None = None,
    output_files: Iterable[Path] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outputs = list(output_files)
    payload = {
        "run_id": run_path.name,
        "run_group": config.get("run_group"),
        "stage": stage,
        "hardware": config.get("hardware"),
        "device": config.get("device"),
        "dtype": config.get("dtype"),
        "amp": config.get("amp"),
        "tf32": config.get("tf32"),
        "deterministic": config.get("deterministic"),
        "seeds": config.get("seeds"),
        "start_time": start_time,
        "end_time": now_iso(),
        "command_line": " ".join(sys.argv),
        "environment": environment_snapshot(),
        "input_hashes": input_hashes or {},
        "output_files": [p.relative_to(run_path).as_posix() for p in outputs if p.exists()],
        "output_hashes": output_hashes(outputs, run_path),
    }
    if extra:
        payload.update(extra)
    metadata_path = run_path / "metadata" / f"{stage}.json"
    write_json(metadata_path, payload)
    marker_payload = {
        "stage": stage,
        "completed_at": payload["end_time"],
        "metadata": metadata_path.relative_to(run_path).as_posix(),
        "output_hashes": payload["output_hashes"],
    }
    write_json(stage_marker(run_path, stage), marker_payload)
    return payload


def append_status(run_path: Path, line: str) -> None:
    path = run_path / "FINAL_STATUS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def write_status(run_path: Path, lines: list[str]) -> None:
    path = run_path / "FINAL_STATUS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def add_common_args(parser: Any) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")


def maybe_skip(args: Any, stage: str, run_path: Path) -> bool:
    if args.force:
        return False
    if args.resume and stage_is_complete(run_path, stage):
        print(f"[skip] {stage}: existing marker and hashes validated")
        return True
    return False


def input_manifest_path(run_path: Path) -> Path:
    return run_path / "manifests" / "input_manifest.csv"


def load_input_manifest(run_path: Path) -> list[dict[str, str]]:
    path = input_manifest_path(run_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input manifest: {path}")
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"Empty input manifest: {path}")
    return rows


def find_input(rows: list[dict[str, str]], *, experiment_name: str, input_type: str, seed: int | str | None = None, resolution: int | str | None = None) -> dict[str, str]:
    matches = []
    for row in rows:
        if row.get("experiment_name") != experiment_name:
            continue
        if row.get("input_type") != input_type:
            continue
        if seed is not None and str(row.get("seed")) != str(seed):
            continue
        if resolution is not None and str(row.get("resolution")) != str(resolution):
            continue
        matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one input for {experiment_name}/{input_type}/seed={seed}/resolution={resolution}, got {len(matches)}")
    return matches[0]


def verified_input_path(run_path: Path, row: dict[str, str]) -> Path:
    path = run_path / row["relative_path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed input: {path}")
    actual = sha256_file(path)
    if actual != row["sha256"]:
        raise RuntimeError(f"Input hash mismatch for {path}: expected {row['sha256']}, got {actual}")
    return path


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / (1024.0 ** 3)
