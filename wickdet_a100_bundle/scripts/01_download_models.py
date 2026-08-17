from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import BUNDLE_ROOT, add_common_args, ensure_run_tree, load_config, maybe_skip, now_iso, run_dir, write_json, write_stage_metadata

STAGE = "download_models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache models for WickDet A100 bundle.")
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    path = run_dir(args.run_id)
    model_name = config["sdvae"]["model"]
    if args.dry_run:
        print(f"[dry-run] {STAGE}: would download/cache {model_name}; HF_TOKEN read only from environment")
        return
    ensure_run_tree(path)
    if maybe_skip(args, STAGE, path):
        return
    start = now_iso()
    bundled_hf_home = BUNDLE_ROOT / "model_cache" / "hf_cache"
    using_bundled_cache = False
    if "HF_HOME" not in os.environ and bundled_hf_home.exists():
        os.environ["HF_HOME"] = str(bundled_hf_home)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        using_bundled_cache = True
    else:
        os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("SCRATCH", str(Path.home())), "hf_cache"))
        using_bundled_cache = Path(os.environ["HF_HOME"]).resolve() == bundled_hf_home.resolve() and bundled_hf_home.exists()
    from diffusers import AutoencoderKL
    kwargs = {"torch_dtype": __import__("torch").float32}
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    if using_bundled_cache or os.environ.get("HF_HUB_OFFLINE") == "1":
        kwargs["local_files_only"] = True
    vae = AutoencoderKL.from_pretrained(model_name, **kwargs)
    param_count = sum(p.numel() for p in vae.parameters())
    manifest = {
        "model": model_name,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_token_used_from_env": bool(token),
        "hf_token_written_to_file": False,
        "local_files_only": bool(kwargs.get("local_files_only", False)),
        "using_bundled_cache": using_bundled_cache,
        "parameter_count": int(param_count),
    }
    out = path / "manifests" / "download_models.json"
    write_json(out, manifest)
    write_stage_metadata(run_path=path, config=config, stage=STAGE, start_time=start, output_files=[out], extra=manifest)
    print(f"model cached: {model_name}")


if __name__ == "__main__":
    main()
