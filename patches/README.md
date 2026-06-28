# gstreamer patches

Patches to gstreamer (not this plugin) needed by the Vulkan-encode path.
Apply against a gstreamer monorepo checkout before building:

```
git apply patches/vkh264enc-dpb-pool-in-new-sequence.patch
git apply patches/vulkanh265enc.patch
```

## vkh264enc-dpb-pool-in-new-sequence.patch

`vulkanh264enc` creates its DPB pool in `propose_allocation`. Behind
`interpipesrc` the allocation query reaches the encoder before its
`set_format`, so the encoder isn't started yet and `create_dpb_pool` fails.
Moves the call into `new_sequence`, after `gst_vulkan_encoder_start`.

Tested on gst 1.28.4 and 1.29.1. Upstream fix pending.

## vulkanh265enc.patch

Adds a Vulkan **H.265/HEVC** video-encode element (`vulkanh265enc`), ported from
the upstream `vulkanh264enc` (`GstH264Encoder` → new `GstH265Encoder` base +
`vkh265enc` element). HEVC specifics: VPS+SPS+PPS (std structs use pointer
sub-structs: profile-tier-level, DecPicBufMgr, per-slice ShortTermRefPicSet,
VUI), POC-based picture order, segment-based slice headers, `no_output_of_prior_
pics_flag` on IDR, CABAC-implicit (no entropy-mode flag, WPP/tiles off), and a
2-slot DPB for single-reference P frames. Includes the same DPB-pool-in-
`new_sequence` interpipe fix as the H.264 patch above.

New files (`subprojects/gst-plugins-bad/ext/vulkan/`): `base/gsth265encoder.{c,h}`,
`vkh265enc.{c,h}`; plus `meson.build` + `gstvulkan.c` registration. Apply *after*
the H.264 patch (independent files; no conflict).

Status: compiles + links + loads clean on 1.28.4; HEVC bitstream design
roundtable-approved. Pending hardware validation on AMD (RADV
`RADV_PERFTEST=video_encode`) before any upstream MR.

(The Vulkan-encode path also needs the device to enable the external-memory
extensions that `GstVulkanDevice` does not — `VK_KHR_external_memory_fd` etc. —
to import the compositor's RGBA dmabuf. Rather than fork gstreamer for that, the
plugin **creates its own `GstVulkanInstance`/`GstVulkanDevice`** with those
extensions enabled and hands it to the encoder via a context-query answer, so no
gstreamer patch is required. See `wayland-display-core/src/utils/vulkan_share.rs`.)

## Building the patched gstreamer

The dev container (`.devcontainer/Dockerfile`) builds this automatically — see
[`CONTRIBUTING.md`](../CONTRIBUTING.md). The manual recipe below documents what
it does.

Built against the `gstreamer` monorepo at tag **1.28.4** (also tested on
1.29.1). `gstreamer-rs` 0.25 needs the GStreamer C library **>= 1.28**, which no
distro (or games-on-whales) image ships yet — so it must be built from source.

Build deps:

```
apt-get install -y meson ninja-build glslang-tools libvulkan-dev \
                   nasm flex bison build-essential pkg-config
```

Configure with `auto_features` off plus explicit per-plugin enables:

```
meson setup builddir \
  -Dauto_features=disabled \
  -Dbase=enabled \
  -Dbad=enabled \
  -Dtools=enabled \
  -Dgst-plugins-base:videotestsrc=enabled \
  -Dgst-plugins-base:app=enabled \
  -Dgst-plugins-base:videoconvertscale=enabled \
  -Dgst-plugins-base:typefind=enabled \
  -Dgst-plugins-bad:vulkan=enabled \
  -Dgst-plugins-bad:vulkan-video=enabled \
  -Dgst-plugins-bad:videoparsers=enabled \
  -Ddoc=disabled \
  --prefix=/opt/gst
ninja -C builddir install
```

Two gotchas:

1. **Vulkan headers must be >= 1.4.317** for the vulkan-video encode plugin.
   Older system headers build `vulkanupload` but leave `vulkanh264enc`
   *silently absent*. Point `PKG_CONFIG_PATH` at a newer `vulkan.pc` (a
   `VK_HEADER_VERSION` 341 / 1.4.341 set works) *first* so meson's `vulkan_dep`
   probe sets `GST_VULKAN_HAVE_VIDEO_EXTENSIONS=1`.
2. With `auto_features=disabled`, **every** subproject's `docs/meson.build`
   (gst-plugins-base, -bad, -good, gst-rtsp-server, the core `gstreamer` tree…)
   references an undefined `plugins_cache_generator` and aborts configure. Prepend
   `if not get_option('doc').allowed() subdir_done() endif` to each
   `subprojects/*/docs/meson.build` (and `docs/meson.build`). (gst-interpipe, if
   built in the same tree, likewise needs `-Denable-gtk-doc=false`.)
