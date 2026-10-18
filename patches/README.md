# gstreamer patches

Patches to gstreamer (not this plugin) needed by the Vulkan-encode path.
Apply them against a gstreamer monorepo checkout, in the order listed in
[`series`](series), before building:

```
for p in $(grep -v -e '^#' -e '^[[:space:]]*$' patches/series); do
  git -C <gstreamer> apply "$PWD/patches/$p"
done
```

`series` is the single source of the apply order: `docker/vulkan.Dockerfile`
and [`ci/gst-patches.sh`](../ci/gst-patches.sh) both read it. That script (run
in CI by `.github/workflows/gst-patches.yml`) builds the patched tree with the
GStreamer unit tests enabled and runs the Vulkan tests; run it locally on a
machine with a Vulkan encode GPU to exercise the device-backed tests too.
After changing a patch, regenerate it with `git diff` from a checkout where the
chain is committed, rather than editing hunks by hand.

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
VUI), POC-based picture order, segment-based slice headers, CABAC-implicit (no
entropy-mode flag, WPP/tiles off), and an explicit short-term RPS per slice.
Includes the same DPB-pool-in-`new_sequence` interpipe fix as the H.264 patch
above.

Each slice's RPS lists every picture held for reference, nearest first, with
the pictures in RefPicList0 / RefPicList1 flagged as used by the current
picture and all of them active, so a decoder predicts from the same pictures
as the encoder. IDR pictures leave `no_output_of_prior_pics_flag` at 0, so a
GOP that ends on B frames doesn't lose the pictures still waiting for output.
`elements_vkh265enc` parses the encoder's output and checks both, with P and
B frames.

New files (`subprojects/gst-plugins-bad/ext/vulkan/`): `base/gsth265encoder.{c,h}`,
`vkh265enc.{c,h}`; plus `meson.build` + `gstvulkan.c` registration. Apply *after*
the H.264 patch (independent files; no conflict).

The element encodes the 4:2:0 Main and Main 10 profiles, from NV12 and P010
input. Its pad templates are narrowed to exactly that, whatever else the
driver's H.265 encode profiles report (NVIDIA also reports `main-444` from
NV24), so a downstream that leaves `profile` open still negotiates a profile
the element can start.

Status: validated on NVIDIA RTX 5090, including Main/Main-10, multiple
references, B frames, strict decoding, RPS/IDR output and live bitrate changes
(see the RPS/IDR regression tests in the patch). AMD H.265 encode validation
is still pending; the successful AMD H.264 tests do not establish H.265 support.

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
1.29.1). The shared Vulkan encode path enables the bindings' `v1_28` API and requires
**GStreamer Vulkan >= 1.28**, including at build time for the C bridge. This
also applies to CUDA-only use of the current unified plugin build. Distributions
with older GStreamer must build a newer prefix; the Vulkan module is not yet
an optional Cargo feature.

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

### Live bitrate lock scope

In GStreamer 1.28.4, H.264's only `update_properties_unlocked()` caller is
`gst_vulkan_h264_encoder_encode_frame()`: it takes the object lock immediately
before the call and releases it immediately after. The rc-fix helper temporarily
releases that lock to query encoder capabilities, then reacquires it to snapshot
and apply the latest property value. A concurrent setter only records the next
request; `self->rc` is updated on the streaming thread. No caller relies on other
state remaining locked across this helper. Preserve the streaming-thread-only
contract when rebasing the patch; the live-retarget test exercises concurrent
property changes.
