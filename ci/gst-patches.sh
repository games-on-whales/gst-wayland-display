#!/usr/bin/env bash
#
# Build a GStreamer checkout with the patches/ chain applied, with the
# GStreamer unit tests enabled, and run the Vulkan tests.
#
# The devcontainer and CI harness build GStreamer without the unit tests, so
# this script is the check that builds the patched tree with the tests enabled
# and runs the ones the patches add, against the same patches/series.
#
# Without a Vulkan video-encode device the device-backed tests skip; the
# GPU-free ones (libs_vkvideoencodeav1refs, the AV1 reference model) still run.
# On a machine with a GPU every test runs for real.
#
# Usage:
#   ci/gst-patches.sh [workdir]        (default workdir: ./.gst-patches)
#
# Environment:
#   GST_VERSION   GStreamer tag to build (default: 1.28.4)
#   GST_REPO      GStreamer monorepo URL (default: gitlab.freedesktop.org)
#   JOBS          parallel build jobs (default: nproc)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mkdir -p "${1:-$ROOT/.gst-patches}" && cd "${1:-$ROOT/.gst-patches}" && pwd)"
GST_VERSION="${GST_VERSION:-1.28.4}"
GST_REPO="${GST_REPO:-https://gitlab.freedesktop.org/gstreamer/gstreamer.git}"
JOBS="${JOBS:-$(nproc)}"
SRC="$WORK/gstreamer"
BUILD="$WORK/build"

echo "==> GStreamer $GST_VERSION from $GST_REPO into $SRC"
rm -rf "$SRC" "$BUILD"
git clone --quiet --depth 1 --branch "$GST_VERSION" "$GST_REPO" "$SRC"

echo "==> applying patches/series"
grep -v -e '^#' -e '^[[:space:]]*$' "$ROOT/patches/series" | while read -r p; do
  echo "    $p"
  git -C "$SRC" apply "$ROOT/patches/$p"
done

# auto_features=disabled leaves several subprojects' docs/meson.build referring
# to an undefined plugins_cache_generator; short-circuit each when doc is off
# (the same workaround as the devcontainer build).
for d in "$SRC"/subprojects/*/docs/meson.build "$SRC"/docs/meson.build; do
  [ -f "$d" ] || continue
  { printf "if not get_option('doc').allowed()\n  subdir_done()\nendif\n"; cat "$d"; } > "$d.tmp"
  mv "$d.tmp" "$d"
done

echo "==> configure"
meson setup "$BUILD" "$SRC" \
  -Dauto_features=disabled \
  -Dbase=enabled -Dbad=enabled -Dgood=disabled -Dtools=enabled \
  -Dugly=disabled -Dlibav=disabled -Dges=disabled \
  -Drtsp_server=disabled -Ddevtools=disabled -Dpython=disabled -Dsharp=disabled \
  -Dgst-plugins-base:app=enabled -Dgst-plugins-base:videotestsrc=enabled \
  -Dgst-plugins-bad:vulkan=enabled -Dgst-plugins-bad:vulkan-video=enabled \
  -Dgst-plugins-bad:videoparsers=enabled \
  -Dorc=disabled -Ddoc=disabled -Dintrospection=disabled -Dexamples=disabled \
  -Dnls=disabled -Dgst-examples=disabled -Drs=disabled -Dlibnice=disabled \
  -Dtests=enabled -Dgstreamer:check=enabled

# Without these the Vulkan video encoders, and every test below, are silently
# compiled out, and the run would pass while testing nothing.
if ! grep -q 'GST_VULKAN_HAVE_VIDEO_EXTENSIONS 1' \
    "$BUILD"/subprojects/gst-plugins-bad/gst-libs/gst/vulkan/gstvkconfig.h; then
  echo "error: Vulkan video extensions not enabled (Vulkan headers >= 1.4.317 needed)" >&2
  exit 1
fi

echo "==> build"
meson compile -C "$BUILD" -j "$JOBS"

TESTS=(
  libs_vkvideoencodeav1refs
  libs_vkvideoencodeav1
  libs_vkvideoencodeh264
  libs_vkvideoencodeh265
  elements_vkencoderetarget
  elements_vkav1enc
  elements_vkh265enc
)
echo "==> test: ${TESTS[*]}"
meson test -C "$BUILD" --no-rebuild --print-errorlogs "${TESTS[@]}"
