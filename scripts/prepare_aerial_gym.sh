#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$PROJECT_ROOT/third_party/aerial_gym_simulator"
REVISION=f0d0f05283f7897bab5a1bcc7b19b91cebbab218
REPOSITORY=https://github.com/ntnu-arl/aerial_gym_simulator.git
if [[ -e "$DEST" ]]; then
    if [[ -d "$DEST/.git" ]]; then
        CURRENT="$(git -C "$DEST" rev-parse HEAD)"
        if [[ "$CURRENT" == "$REVISION" ]]; then
            echo "Aerial Gym already matches the pinned revision."
            exit 0
        fi
        echo "Refusing to replace an existing Aerial Gym checkout at a different revision." >&2
        exit 2
    fi
    echo "Aerial Gym already exists without Git metadata; inspect it manually. No files were changed." >&2
    exit 2
fi
mkdir -p "$PROJECT_ROOT/third_party"
TEMP="$PROJECT_ROOT/third_party/.aerial_gym_simulator.tmp.$$"
trap 'rm -rf "$TEMP"' EXIT
git clone --no-checkout "$REPOSITORY" "$TEMP"
git -C "$TEMP" checkout --detach "$REVISION"
ACTUAL="$(git -C "$TEMP" rev-parse HEAD)"
if [[ "$ACTUAL" != "$REVISION" ]]; then
    echo "Fetched Aerial Gym revision does not match the pin." >&2
    exit 3
fi
mv "$TEMP" "$DEST"
trap - EXIT
echo "Prepared Aerial Gym at the pinned revision."
