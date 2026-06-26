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

Built against the `gstreamer` monorepo at tag **1.28.4** (also tested on
1.29.1). Note the prebuilt `ghcr.io/games-on-whales/gstreamer:1.26.7` image
used by the devcontainer does **not** ship `vulkanh264enc` — the Vulkan-encode
path needs a hand-built gstreamer as below.

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
2. With `auto_features=disabled`, `gst-plugins-bad/docs/meson.build` needs an
   early `if not get_option('doc').allowed() subdir_done() endif` or the
   `plugins_cache_generator` target goes missing. (gst-interpipe, if built in
   the same tree, likewise needs `-Denable-gtk-doc=false`.)
