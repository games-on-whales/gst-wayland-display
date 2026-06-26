# gstreamer patches

Patches to gstreamer (not this plugin) needed by the Vulkan-encode path.
Apply against a gstreamer monorepo checkout before building:

```
git apply patches/vkh264enc-dpb-pool-in-new-sequence.patch
```

## vkh264enc-dpb-pool-in-new-sequence.patch

`vulkanh264enc` creates its DPB pool in `propose_allocation`. Behind
`interpipesrc` the allocation query reaches the encoder before its
`set_format`, so the encoder isn't started yet and `create_dpb_pool` fails.
Moves the call into `new_sequence`, after `gst_vulkan_encoder_start`.

Tested on gst 1.28.4 and 1.29.1. Upstream fix pending.

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
