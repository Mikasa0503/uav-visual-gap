#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION=4.2.21
ARCHIVE="blender-${VERSION}-linux-x64.tar.xz"
EXPECTED_SHA=b9ee313018de52697eeabcb76fc2cd6d404dbb670be9b0d3a5847a09ca325981
DEST="$ROOT_DIR/third_party/blender-${VERSION}-linux-x64"
if [[ -x "$DEST/blender" ]]; then
  "$DEST/blender" --version
  exit 0
fi
mkdir -p "$ROOT_DIR/.cache/downloads" "$ROOT_DIR/third_party"
PACKAGE="$ROOT_DIR/.cache/downloads/$ARCHIVE"
curl --noproxy '*' -fL --retry 3 --connect-timeout 20 --max-time 600 \
  "https://download.blender.org/release/Blender4.2/$ARCHIVE" -o "$PACKAGE"
printf '%s  %s\n' "$EXPECTED_SHA" "$PACKAGE" | sha256sum --check -
tar -xJf "$PACKAGE" -C "$ROOT_DIR/third_party"
"$DEST/blender" --version
