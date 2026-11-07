#!/usr/bin/env bash
# Run with the same glslc version when regenerating and checking SPIR-V.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for source in wayland-display-core/src/utils/shaders/*.comp; do
  binary=${source%.comp}.spv
  glslc "$source" -o "$tmp/shader.spv"
  if [[ ${1:-} == --update ]]; then
    cp "$tmp/shader.spv" "$binary"
  else
    cmp "$tmp/shader.spv" "$binary" || {
      echo "Stale shader: $binary; run ci/shaders.sh --update" >&2
      exit 1
    }
  fi
done
