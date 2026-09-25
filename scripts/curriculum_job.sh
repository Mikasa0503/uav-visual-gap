#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
JOB_NAME="$1"
shift
[[ "$JOB_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
mkdir -p runs/jobs
printf '%s\n' "$$" > "runs/jobs/$JOB_NAME.pid"
trap 'code=$?; printf "%s\n" "$code" > "runs/jobs/$JOB_NAME.exit"' EXIT
scripts/run.sh scripts/run_teacher_curriculum.py "$@"
