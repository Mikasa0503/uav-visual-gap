#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${UAV_GAP_ENV:-}" ]]; then
    ENV_ROOT="$(cd "$UAV_GAP_ENV" && pwd)"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    ENV_ROOT="$(cd "$CONDA_PREFIX" && pwd)"
else
    echo "Activate the uav_gap Conda environment or set UAV_GAP_ENV." >&2
    exit 2
fi
PYTHON="$ENV_ROOT/bin/python"
if [[ ! -x "$PYTHON" || ! -d "$ENV_ROOT/conda-meta" ]]; then
    echo "Selected environment is not a valid Conda environment with Python: $ENV_ROOT" >&2
    exit 2
fi
export UAV_GAP_ENV="$ENV_ROOT"
export PATH="$ENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src"
export LD_LIBRARY_PATH="$ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/.cache/torch_extensions"
export WARP_CACHE_PATH="$PROJECT_ROOT/.cache/warp"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export OMP_NUM_THREADS=4
cd "$PROJECT_ROOT"
exec "$PYTHON" "$@"
