#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_ID="smoke_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=0
FORCE=0
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --resume) RESUME=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
CONFIG="configs/smoke.yaml"
COMMON=(--config "$CONFIG" --run-id "$RUN_ID")
[[ $FORCE -eq 1 ]] && COMMON+=(--force)
[[ $RESUME -eq 1 ]] && COMMON+=(--resume)
[[ $DRY_RUN -eq 1 ]] && COMMON+=(--dry-run)
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
if [[ -z "${HF_HOME:-}" && -d "$ROOT/model_cache/hf_cache" ]]; then
  export HF_HOME="$ROOT/model_cache/hf_cache"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
else
  export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"
fi
mkdir -p logs
: > logs/latest.out
: > logs/latest.err
run() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@" >> logs/latest.out 2>> logs/latest.err
  fi
}
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=outputs/$RUN_ID"
run python scripts/00_preflight.py "${COMMON[@]}"
run python scripts/01_download_models.py "${COMMON[@]}"
run python scripts/02_make_inputs.py "${COMMON[@]}"
run python scripts/10_controlled_spectra.py "${COMMON[@]}"
run python scripts/20_elbo_toy.py "${COMMON[@]}"
run python scripts/30_fno_resolution.py "${COMMON[@]}"
run python scripts/40_sdvae_jacobian.py "${COMMON[@]}" --sdvae-stage smoke
run python scripts/90_plot_all.py "${COMMON[@]}"
run python scripts/99_validate_all.py "${COMMON[@]}"
echo "smoke pipeline complete: outputs/$RUN_ID"
