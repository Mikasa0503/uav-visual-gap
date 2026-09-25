#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$PROJECT_ROOT/scripts/run.sh"
for required in \
    "$PROJECT_ROOT/third_party/isaacgym/isaacgym/python" \
    "$PROJECT_ROOT/third_party/aerial_gym_simulator"; do
    if [[ ! -d "$required" ]]; then
        echo "Required local dependency is missing: $required" >&2
        echo "See docs/THIRD_PARTY.md for licensed/source setup." >&2
        exit 2
    fi
done
mkdir -p "$PROJECT_ROOT/.cache/pip" "$PROJECT_ROOT/logs"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
"$RUN" -m pip install -c configs/runtime-constraints.txt -r requirements-runtime.txt
# Replace the CPU-only Conda Warp build with the pinned official wheel.
"$RUN" -m pip install --force-reinstall --no-deps warp-lang==1.0.0
"$RUN" -m pip install --no-deps --no-build-isolation \
    -e third_party/isaacgym/isaacgym/python \
    -e third_party/aerial_gym_simulator -e .
"$RUN" -m pip check | tee logs/pip-check.log
"$RUN" scripts/check_runtime.py --write-inventory
