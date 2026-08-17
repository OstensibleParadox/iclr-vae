#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$RUN_ID" ]]; then
  if find outputs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' >/dev/null 2>&1; then
    RUN_ID="$(find outputs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 1)"
  else
    RUN_ID="$(python - <<'PY'
from pathlib import Path
runs = sorted(p.name for p in Path("outputs").iterdir() if p.is_dir())
print(runs[-1] if runs else "")
PY
)"
  fi
fi
if [[ -z "$RUN_ID" || ! -d "outputs/$RUN_ID" ]]; then
  echo "missing run directory: outputs/$RUN_ID" >&2
  exit 1
fi
TAR="wickdet_a100_results_${RUN_ID}.tar.gz"
tar -czf "$TAR" "outputs/$RUN_ID"
echo "$TAR"
