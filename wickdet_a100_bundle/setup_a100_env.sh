#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="wickdet-a100"
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLED_HF_HOME="$BUNDLE_ROOT/model_cache/hf_cache"
if [[ -z "${HF_HOME:-}" && -d "$BUNDLED_HF_HOME" ]]; then
  export HF_HOME="$BUNDLED_HF_HOME"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
else
  export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"
fi
mkdir -p "$HF_HOME"

echo "[setup] HF_HOME=$HF_HOME"
if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] conda not found. Load your cluster's conda/anaconda module first, then rerun this script." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "[setup] creating/updating conda env: $ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  conda env create -f environment.yml
fi

conda activate "$ENV_NAME"
python - <<'PY'
import torch
print('torch.cuda.is_available() =', torch.cuda.is_available())
print('torch version =', torch.__version__)
print('torch CUDA version =', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU =', torch.cuda.get_device_name(0))
PY
