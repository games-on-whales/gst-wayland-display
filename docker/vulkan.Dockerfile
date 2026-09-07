# gst-wayland-display:vulkan
#
# A Fedora image carrying patched GStreamer 1.28.4 (Vulkan video encode enabled +
# the vulkanh264enc DPB-pool patch + the vulkanh265enc element) with the
# gst-wayland-display plugin installed.
# It is the single source of the gst-1.28.4-Vulkan + plugin build and serves as the
# base image for wolf:vulkan (which compiles Wolf on top and inherits the plugin).
#
# Why only GStreamer is built from source (vs the Ubuntu .devcontainer which also
# builds libwayland + overlays Vulkan headers): Fedora already ships libwayland
# >= 1.23 (wl_client_set_max_buffer_size) and Vulkan headers >= 1.4.317
# (GST_VULKAN_HAVE_VIDEO_EXTENSIONS), so the only reason to build gst ourselves is
# to apply patches/vkh264enc-dpb-pool-in-new-sequence.patch, which the Wolf
# interpipe path requires (without it, behind interpipesrc the encoder asserts on
# priv->layered_buffer).
ARG BASE_IMAGE=ghcr.io/games-on-whales/base-app:fedora
FROM ${BASE_IMAGE}

ARG GST_VERSION=1.28.4
ARG RUST_VERSION=1.94.0

# --- Build dependencies -----------------------------------------------------
# `dnf builddep` pulls GStreamer's own build-requires (vulkan-headers, glslang,
# shaderc, wayland-protocols, glib, etc.) exactly as Fedora built its 1.28.x, so
# we don't hand-maintain that list; we add the toolchain + the few -devel libs the
# plugin's Rust bindings need (clang/llvm for bindgen, libinput, etc.).
RUN dnf install -y dnf-plugins-core 'dnf-command(builddep)' && \
    dnf builddep -y gstreamer1 gstreamer1-plugins-base gstreamer1-plugins-good gstreamer1-plugins-bad-free && \
    dnf install -y \
      git ca-certificates curl gcc gcc-c++ make cmake pkgconf-pkg-config \
      ninja-build nasm flex bison meson \
      glib2-devel libdrm-devel mesa-libgbm-devel systemd-devel \
      wayland-devel wayland-protocols-devel libxkbcommon-devel libX11-devel \
      libinput-devel openssl-devel clang clang-devel llvm-devel \
      libffi-devel expat-devel vulkan-headers vulkan-loader-devel \
      opus-devel pulseaudio-libs-devel && \
    dnf clean all

# --- RADV with Vulkan H.264/H.265 video encode (RPM Fusion freeworld mesa) ---
# Fedora's stock mesa strips the patent codecs, so the RADV device exposes no
# video-encode queue and vulkanh264enc fails at runtime. The freeworld build
# restores it. (No-op for nvidia, which uses its own ICD.)
#
# MUST pin .x86_64 + pass --allowerasing: freeworld is a drop-in replacement that
# CONFLICTS with the stock x86_64 mesa-vulkan-drivers. Without --allowerasing dnf
# can't erase the stock x86_64 driver, so it silently installs only the non-
# conflicting i686 freeworld package — leaving the 64-bit RADV ICD stock (no
# encode queue) and vulkanh264enc absent. --skip-unavailable keeps nvidia builds
# (where the package may be absent) a no-op.
RUN dnf install -y \
      "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" && \
    dnf install -y --allowerasing --skip-unavailable mesa-vulkan-drivers-freeworld.x86_64 && \
    dnf clean all

# --- Patched GStreamer 1.28.4 -> /opt/gst -----------------------------------
COPY patches/vkh264enc-dpb-pool-in-new-sequence.patch /tmp/dpb.patch
COPY patches/vulkanh265enc.patch /tmp/h265.patch
RUN git clone --depth 1 --branch ${GST_VERSION} \
      https://gitlab.freedesktop.org/gstreamer/gstreamer.git /tmp/gstreamer && \
    cd /tmp/gstreamer && \
    git apply /tmp/dpb.patch && \
    git apply /tmp/h265.patch && \
    # auto_features=disabled leaves several subprojects' docs/meson.build referring
    # to an undefined plugins_cache_generator; short-circuit each when doc is off.
    for d in subprojects/*/docs/meson.build docs/meson.build; do \
      [ -f "$d" ] || continue; \
      { printf "if not get_option('doc').allowed()\n  subdir_done()\nendif\n"; cat "$d"; } > "$d.tmp"; \
      mv "$d.tmp" "$d"; \
    done && \
    meson setup build --prefix=/opt/gst --libdir=lib64 \
      -Dauto_features=disabled \
      -Dbase=enabled -Dbad=enabled -Dgood=enabled -Dtools=enabled \
      -Dugly=disabled -Dlibav=disabled -Dges=disabled \
      -Drtsp_server=disabled -Ddevtools=disabled -Dpython=disabled -Dsharp=disabled \
      -Dgst-plugins-base:app=enabled -Dgst-plugins-base:typefind=enabled \
      -Dgst-plugins-base:videotestsrc=enabled -Dgst-plugins-base:videoconvertscale=enabled \
      -Dgst-plugins-base:playback=enabled -Dgst-plugins-base:drm=enabled \
      -Dgst-plugins-base:audioconvert=enabled -Dgst-plugins-base:audioresample=enabled \
      -Dgst-plugins-base:audiorate=enabled -Dgst-plugins-base:opus=enabled \
      -Dgst-plugins-base:volume=enabled \
      -Dgst-plugins-good:pulse=enabled \
      -Dgst-plugins-bad:vulkan=enabled -Dgst-plugins-bad:vulkan-video=enabled \
      -Dgst-plugins-bad:videoparsers=enabled -Dgst-plugins-bad:wayland=enabled \
      -Dorc=disabled -Ddoc=disabled -Dgtk_doc=disabled \
      -Dintrospection=disabled -Dexamples=disabled -Dtests=disabled \
      -Dnls=disabled -Dgst-examples=disabled -Drs=disabled && \
    meson compile -C build && \
    meson install -C build && \
    rm -rf /tmp/gstreamer /tmp/dpb.patch /tmp/h265.patch

ENV PKG_CONFIG_PATH=/opt/gst/lib64/pkgconfig \
    LD_LIBRARY_PATH=/opt/gst/lib64 \
    GST_PLUGIN_PATH=/opt/gst/lib64/gstreamer-1.0 \
    PATH=/opt/gst/bin:/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
    LIBCLANG_PATH=/usr/lib64

# --- gst-interpipe (games-on-whales fork) -> /opt/gst -----------------------
# Wolf links its producer and consumer GstPipelines with interpipesink/
# interpipesrc; without this plugin every Wolf pipeline fails to parse ("no
# element interpipesrc") and no video/audio is produced. It isn't packaged by
# Fedora, and Fedora's gst is 1.26 (ABI-incompatible with /opt/gst's 1.28.4),
# so build it against /opt/gst (PKG_CONFIG_PATH set above). Same fork/recipe
# Wolf's own gstreamer image uses.
RUN git clone --depth 1 https://github.com/games-on-whales/gst-interpipe.git /tmp/gst-interpipe && \
    cd /tmp/gst-interpipe && \
    meson setup build --prefix=/opt/gst --libdir=lib64 -Denable-gtk-doc=false && \
    meson compile -C build && \
    meson install -C build && \
    rm -rf /tmp/gst-interpipe

# --- Rust toolchain (gstreamer-rs 0.25 + cargo-c need >= 1.94) + the plugin ---
ENV CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup
# cargo-c is pinned because it is installed from crates.io at build time, so an
# unpinned `cargo install` picks up whatever is newest and drifts away from the
# toolchain pinned above. cargo-c 0.10.24 raised its MSRV to rustc 1.95 and broke
# this layer against RUST_VERSION=1.94.0; 0.10.23 is the last release supporting
# 1.94. `--locked` uses cargo-c's own Cargo.lock so a transitive dependency
# raising ITS MSRV cannot break the build again without a deliberate bump here.
# Raise both this pin and RUST_VERSION together.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal && \
    cargo install cargo-c --version 0.10.23 --locked

# Cache-bust the plugin source COPY + compile. The registry build cache (cache-from/
# cache-to mode=max) can serve a STALE `cargo cinstall` layer even when src/ changed,
# leaving an outdated plugin .so in the image. Bump this to force a clean recompile.
ARG PLUGIN_CACHEBUST=2026-06-30-fmt-fix2
RUN echo "plugin rebuild: ${PLUGIN_CACHEBUST}"

COPY . /src
WORKDIR /src
# Install the plugin .so into /opt/gst's plugin dir (so anything FROM this image,
# e.g. wolf:vulkan, inherits it on GST_PLUGIN_PATH) plus its pkg-config/header.
RUN cargo cinstall --release \
      --prefix=/opt/gst \
      --libdir=/opt/gst/lib64/gstreamer-1.0 \
      --pkgconfigdir=/opt/gst/lib64/pkgconfig

WORKDIR /
CMD ["bash"]
