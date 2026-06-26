# Reproducible build environment for gst-plugin-wayland-display.
#
# Why this is non-trivial:
#  * gstreamer-rs 0.25 requires the GStreamer C library >= 1.28, which no distro
#    (or games-on-whales) image ships yet.
#  * The Vulkan-encode path needs the vulkanh264enc DPB-pool patch
#    (patches/vkh264enc-dpb-pool-in-new-sequence.patch).
#  * vulkanh264enc only builds against Vulkan headers >= 1.4.317.
#  * The "raise wayland max buffer size" change needs libwayland >= 1.23
#    (wl_client_set_max_buffer_size); Ubuntu 24.04 ships 1.22.
#  * gstreamer-rs 0.25 + current cargo-c need rustc >= 1.94.
#
# So we build patched GStreamer 1.28.4 and libwayland 1.23 from source.
FROM ubuntu:24.04

ARG GST_VERSION=1.28.4
ARG WL_VERSION=1.23.1
ARG VK_HEADERS_TAG=v1.4.341
ARG RUST_VERSION=1.94.0
ENV DEBIAN_FRONTEND=noninteractive

# --- gcc anchor -------------------------------------------------------------
# This ubuntu:24.04 image's gcc cannot self-locate its install prefix: it emits
# a CWD-relative `../lib/gcc` / `../libexec/gcc` and then fails to find cc1/as/ld
# and its startfiles (crtbeginS.o, libgcc) once a compiler runs from a deep build
# directory (e.g. meson's temp dir). Anchor both with absolute paths. This is a
# no-op on a healthy host. (Paths are gcc-13 / x86_64 specific = this base image.)
ENV GCC_EXEC_PREFIX=/usr/lib/gcc/
ENV COMPILER_PATH=/usr/libexec/gcc/x86_64-linux-gnu/13:/usr/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl build-essential pkg-config \
      python3 python3-pip ninja-build nasm flex bison \
      glslang-tools libshaderc-dev libvulkan-dev \
      libglib2.0-dev libdrm-dev libgbm-dev libudev-dev \
      libwayland-dev wayland-protocols libxkbcommon-dev libx11-dev \
      libffi-dev libexpat1-dev \
      libinput-dev libssl-dev clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu's meson is too old for gst 1.28; get a recent one from PyPI.
RUN pip3 install --break-system-packages --no-cache-dir 'meson>=1.4'

# --- libwayland >= 1.23 (provides wl_client_set_max_buffer_size) -------------
RUN git clone --depth 1 --branch ${WL_VERSION} \
      https://gitlab.freedesktop.org/wayland/wayland.git /tmp/wayland \
    && cd /tmp/wayland \
    && meson setup build --prefix=/usr --libdir=lib/x86_64-linux-gnu \
         -Ddocumentation=false -Dtests=false -Ddtd_validation=false \
    && meson install -C build \
    && rm -rf /tmp/wayland

# --- Vulkan headers >= 1.4.317 overlay --------------------------------------
# vulkan-video encode needs newer headers than Ubuntu ships; expose them plus a
# bumped vulkan.pc so gstreamer's meson vulkan_dep probe enables
# GST_VULKAN_HAVE_VIDEO_EXTENSIONS (else vulkanh264enc is silently absent).
RUN git clone --depth 1 --branch ${VK_HEADERS_TAG} \
      https://github.com/KhronosGroup/Vulkan-Headers /opt/vkhdr \
    && mkdir -p /opt/vkpc && { \
      echo 'prefix=/opt/vkhdr'; \
      echo 'includedir=${prefix}/include'; \
      echo ''; \
      echo 'Name: Vulkan'; \
      echo 'Description: Vulkan headers overlay (video extensions)'; \
      echo 'Version: 1.4.341'; \
      echo 'Libs: -lvulkan'; \
      echo 'Cflags: -I${includedir}'; \
    } > /opt/vkpc/vulkan.pc

# --- patched GStreamer 1.28.4 -> /opt/gst -----------------------------------
COPY patches/vkh264enc-dpb-pool-in-new-sequence.patch /tmp/dpb.patch
RUN git clone --depth 1 --branch ${GST_VERSION} \
      https://gitlab.freedesktop.org/gstreamer/gstreamer.git /tmp/gstreamer \
    && cd /tmp/gstreamer \
    && git apply /tmp/dpb.patch \
    # With auto_features=disabled several subprojects' docs/meson.build reference
    # an undefined plugins_cache_generator; short-circuit each when doc is off.
    && for d in subprojects/*/docs/meson.build docs/meson.build; do \
         [ -f "$d" ] || continue; \
         { printf "if not get_option('doc').allowed()\n  subdir_done()\nendif\n"; cat "$d"; } > "$d.tmp"; \
         mv "$d.tmp" "$d"; \
       done \
    && PKG_CONFIG_PATH=/opt/vkpc meson setup build --prefix=/opt/gst \
         -Dauto_features=disabled \
         -Dbase=enabled -Dbad=enabled -Dtools=enabled \
         -Dgood=disabled -Dugly=disabled -Dlibav=disabled -Dges=disabled \
         -Drtsp_server=disabled -Ddevtools=disabled -Dpython=disabled -Dsharp=disabled \
         -Dgst-plugins-base:app=enabled -Dgst-plugins-base:typefind=enabled \
         -Dgst-plugins-base:videotestsrc=enabled -Dgst-plugins-base:videoconvertscale=enabled \
         -Dgst-plugins-base:playback=enabled -Dgst-plugins-base:drm=enabled \
         -Dgst-plugins-bad:vulkan=enabled -Dgst-plugins-bad:vulkan-video=enabled \
         -Dgst-plugins-bad:videoparsers=enabled -Dgst-plugins-bad:wayland=enabled \
         -Dorc=disabled -Ddoc=disabled -Dgtk_doc=disabled \
         -Dintrospection=disabled -Dexamples=disabled -Dtests=disabled \
         -Dnls=disabled -Dgst-examples=disabled -Drs=disabled \
    && meson compile -C build \
    && meson install -C build \
    && rm -rf /tmp/gstreamer /tmp/dpb.patch

ENV PKG_CONFIG_PATH=/opt/gst/lib/x86_64-linux-gnu/pkgconfig:/opt/gst/lib/pkgconfig \
    LD_LIBRARY_PATH=/opt/gst/lib/x86_64-linux-gnu:/opt/gst/lib \
    GST_PLUGIN_PATH=/opt/gst/lib/x86_64-linux-gnu/gstreamer-1.0 \
    PATH=/opt/gst/bin:/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
    LIBCLANG_PATH=/usr/lib/llvm-18/lib

# --- Rust toolchain (gstreamer-rs 0.25 + cargo-c need >= 1.94) + cargo-c -----
ENV CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal \
    && rustc --version \
    && cargo install cargo-c

WORKDIR /src
CMD ["bash"]
