#!/usr/bin/env bash
#
# Measure the per-frame *source-thread* NV12 conversion cost -- the time
# VulkanNv12::convert() spends on the producer's critical path (including any GPU
# wait). This is exactly what the pipelining lever removes, so running it against the
# pre-lever and post-lever builds quantifies the win.
#
# Requires the conv-timing.patch applied to the build under test (see README.md).
#
# Usage:
#   GST_PLUGIN_PATH=<plugin dir> ./run-conv-timing.sh <render-node> <tail> [w] [h] [fps] [nframes]
#
#   <tail>  downstream of the source, e.g. 'vah265enc ! fakesink sync=false'
#   Set DRM_FORMAT=NV12 to force the source's NV12 drm-format (needed for the Nvidia
#   dmabuftocuda path; Intel vah265enc negotiates NV12 from bare DMABuf caps instead).
#
# Examples:
#   # Intel / AMD VA (bare DMABuf caps, encoder pulls NV12):
#   GST_PLUGIN_PATH=target/debug ./run-conv-timing.sh /dev/dri/renderD129 'vah265enc ! fakesink sync=false'
#   # AMD forcing LINEAR (the only modifier its vah265enc accepts) under nodcc:
#   AMD_DEBUG=nodcc RADV_DEBUG=nodcc DRM_FORMAT=NV12 GST_PLUGIN_PATH=target/debug \
#     ./run-conv-timing.sh /dev/dri/renderD128 'vah265enc ! fakesink sync=false'
#   # Nvidia (dmabuftocuda -> nvenc):
#   DRM_FORMAT=NV12 GST_PLUGIN_PATH=target/debug \
#     ./run-conv-timing.sh /dev/dri/renderD128 'dmabuftocuda render-node=/dev/dri/renderD128 ! nvh265enc ! fakesink sync=false' 1280 720 120 300
#
set -euo pipefail

NODE="${1:?render node, e.g. /dev/dri/renderD128}"
TAIL="${2:?downstream pipeline, e.g. 'vah265enc ! fakesink sync=false'}"
W="${3:-1280}"; H="${4:-720}"; FPS="${5:-120}"; N="${6:-300}"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/wlrun}"
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
export CONV_TIMING=1
export RUST_LOG="${RUST_LOG:-waylanddisplaycore::utils::allocator=info}"

caps="video/x-raw(memory:DMABuf),width=$W,height=$H,framerate=$FPS/1"
if [ -n "${DRM_FORMAT:-}" ]; then
  caps="video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=$DRM_FORMAT,width=$W,height=$H,framerate=$FPS/1"
fi

echo "node=$NODE caps=$caps tail=$TAIL frames=$N"
# shellcheck disable=SC2086
timeout 180 gst-launch-1.0 -q \
  waylanddisplaysrc render-node="$NODE" num-buffers="$N" \
  ! "$caps" \
  ! $TAIL 2>&1 | grep CONVTIMING | tail -1
