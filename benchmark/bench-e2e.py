#!/usr/bin/env python3
"""
End-to-end latency + throughput harness for comparing the in-source Vulkan NV12
converter (this PR) against the downstream-vapostproc path (main/stable).

Measures, via pad probes matched by buffer PTS:
  - latency: time from the buffer leaving waylanddisplaysrc to the matching encoded
    buffer leaving the encoder (source-produced -> encoded-out).
  - throughput: encoded frames / wall-clock once flowing.

Topology-independent, so it compares fairly across:
  stable: waylanddisplaysrc(RGBA DMABuf) ! vapostproc ! VAMemory,NV12 ! vah265enc
  pr:     waylanddisplaysrc(NV12 DMABuf, in-source Vulkan convert) ! vah265enc

Usage:
  bench-e2e.py --node /dev/dri/renderD129 --mode {stable,pr} \
      [--enc vah265enc] [-w 1280] [-h 720] [--fps 120] [-n 300] [--drm-format NV12]

For the PR path on AMD/Nvidia pass --drm-format NV12 to force the NV12 modifier; on
Intel omit it (vah265enc pulls NV12 from bare DMABuf caps). stable always uses bare
DMABuf RGBA caps into vapostproc.
"""
import argparse
import os
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

Gst.init(None)


def build_pipeline(a):
    # do-timestamp stamps each buffer with a monotonic PTS so the src/enc probes can be
    # matched per frame (the latency tracer can't follow buffer-replacing converters).
    src = f"waylanddisplaysrc name=wl render_node={a.node} num-buffers={a.n} do-timestamp=true"
    # --raw-tail lets the caller express the whole post-source pipeline (must include
    # `fakesink name=sink`); used for the Nvidia paths (glupload/glcolorconvert vs
    # dmabuftocuda) which don't fit the stable/pr templates below.
    if a.raw_tail:
        desc = f"{src} ! {a.raw_tail}"
        print(f"[pipeline] {desc}", flush=True)
        return Gst.parse_launch(desc)
    # Measure the PRE-ENCODER stage only: source + RGBA->NV12 conversion, terminating at a
    # fakesink right after NV12. This PR changes only the conversion, not the encoder, so
    # including the encoder would just bottleneck on it and hide the difference. Pass
    # --enc <element> to re-insert an encoder before the sink if needed.
    if a.mode == "stable":
        # main/stable: source emits RGBA DMABuf, vapostproc converts to NV12.
        caps = f"video/x-raw(memory:DMABuf),width={a.w},height={a.h},framerate={a.fps}/1"
        stage = f"{caps} ! vapostproc ! video/x-raw(memory:VAMemory),format=NV12"
    elif a.drm_format:
        # this PR: in-source Vulkan converter emits NV12 DMABuf directly.
        stage = (
            f"video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format={a.drm_format},"
            f"width={a.w},height={a.h},framerate={a.fps}/1"
        )
    else:
        stage = f"video/x-raw(memory:DMABuf),width={a.w},height={a.h},framerate={a.fps}/1"
    tail = f"{a.enc} ! fakesink name=sink sync=false" if a.enc else "fakesink name=sink sync=false"
    desc = f"{src} ! {stage} ! {tail}"
    print(f"[pipeline] {desc}", flush=True)
    return Gst.parse_launch(desc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node", required=True)
    p.add_argument("--mode", choices=["stable", "pr"], default="pr")
    p.add_argument("--raw-tail", default=None, help="full post-source pipeline incl. 'fakesink name=sink' (overrides --mode)")
    p.add_argument("--enc", default=None, help="optional encoder element to insert before the sink (default: none = pre-encoder only)")
    p.add_argument("-w", "--width", dest="w", type=int, default=1280)
    p.add_argument("--height", dest="h", type=int, default=720)
    p.add_argument("--fps", type=int, default=120)
    p.add_argument("-n", type=int, default=300)
    p.add_argument("--drm-format", default=None)
    a = p.parse_args()

    os.makedirs("/tmp/wlrun", exist_ok=True)
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/wlrun")

    pipe = build_pipeline(a)
    loop = GLib.MainLoop()
    t_in = {}      # pts -> monotonic time entering pipeline (source src pad)
    lat = []       # measured latencies (s)
    out_times = [] # monotonic time of each encoded buffer

    wl = pipe.get_by_name("wl")
    sink = pipe.get_by_name("sink")

    def on_src(_pad, info):
        buf = info.get_buffer()
        if buf is not None:
            t_in[buf.pts] = time.monotonic()
        return Gst.PadProbeReturn.OK

    def on_sink(_pad, info):
        now = time.monotonic()
        buf = info.get_buffer()
        if buf is not None:
            out_times.append(now)
            t0 = t_in.get(buf.pts)
            if t0 is not None:
                lat.append(now - t0)
        return Gst.PadProbeReturn.OK

    wl.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_src)
    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_sink)

    def on_msg(_bus, m):
        if m.type == Gst.MessageType.EOS:
            loop.quit()
        elif m.type == Gst.MessageType.ERROR:
            e, d = m.parse_error()
            print(f"[ERROR] {e.message} | {d}", flush=True)
            loop.quit()
        return True

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_msg)

    pipe.set_state(Gst.State.PLAYING)
    GLib.timeout_add_seconds(120, loop.quit)
    loop.run()
    pipe.set_state(Gst.State.NULL)

    # Sustained throughput from the encoder-output timestamps (robust; needs no PTS
    # match). Discard the first 20 frames as warm-up.
    warm = 20
    fps = 0.0
    if len(out_times) > warm + 5:
        span = out_times[-1] - out_times[warm]
        fps = (len(out_times) - 1 - warm) / span if span > 0 else 0.0
    lat_str = "n/a"
    if lat:
        lat.sort()
        us = [x * 1e6 for x in lat]
        n = len(us)
        lat_str = (
            f"avg={sum(us) / n:.0f} p50={us[n // 2]:.0f} "
            f"p99={us[min(n - 1, int(n * 0.99))]:.0f} max={us[-1]:.0f}"
        )
    print(
        f"[result] mode={a.mode} {a.w}x{a.h} req_fps={a.fps} enc={a.enc} "
        f"enc_out={len(out_times)} throughput_fps={fps:.1f} latency_us[{lat_str}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
