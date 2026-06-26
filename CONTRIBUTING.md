# Contributing / Building

`gst-plugin-wayland-display` is a GStreamer plugin written in Rust. Its build has
a few hard version floors that no current distro image satisfies out of the box,
so the supported way to build it is the provided container.

## Version floors

| Dependency        | Minimum   | Why                                                                 |
| ----------------- | --------- | ------------------------------------------------------------------- |
| GStreamer (C lib) | **1.28**  | `gstreamer-rs` 0.25 requires it                                     |
| Vulkan headers    | **1.4.317** | `vulkanh264enc` (Vulkan-encode path) only builds against these    |
| libwayland        | **1.23**  | `wl_client_set_max_buffer_size` (raised max buffer size)            |
| rustc             | **1.94**  | `gstreamer-rs` 0.25 (≥1.92) and current `cargo-c` (≥1.94)           |

The Vulkan-encode path additionally needs the bundled encoder patch
(`patches/vkh264enc-dpb-pool-in-new-sequence.patch`), which the container applies
to GStreamer before building it. See `patches/README.md`.

## Build with the container (recommended)

The dev container (`.devcontainer/Dockerfile`) builds patched **GStreamer 1.28.4**
and **libwayland 1.23** from source and installs **Rust 1.94** + `cargo-c`. Open
the folder in VS Code ("Reopen in Container"), or build/run it directly:

```sh
# Build the environment image (gstreamer + libwayland from source; ~10 min first run)
docker build -t gwd-dev -f .devcontainer/Dockerfile .

# Build the plugin inside it
docker run --rm -v "$PWD":/src -w /src gwd-dev \
  cargo build -p gst-plugin-wayland-display

# Install the plugin .so into the image's GStreamer prefix and inspect it
docker run --rm -v "$PWD":/src -w /src gwd-dev bash -lc '
  cargo cinstall -p gst-plugin-wayland-display --prefix=/opt/gst --library-type cdylib &&
  gst-inspect-1.0 waylanddisplaysrc'
```

To exercise the GPU paths (VA / Vulkan / CUDA) at runtime, pass the render node:
`docker run --device /dev/dri ...` (or `--runtime=nvidia` / `--gpus all` for NVIDIA).

## Building GStreamer yourself

GStreamer is installed to `/opt/gst`. To build against an existing GStreamer
checkout, point the toolchain at that prefix:

```sh
export PKG_CONFIG_PATH=/opt/gst/lib/x86_64-linux-gnu/pkgconfig:/opt/gst/lib/pkgconfig
export LD_LIBRARY_PATH=/opt/gst/lib/x86_64-linux-gnu:/opt/gst/lib
export GST_PLUGIN_PATH=/opt/gst/lib/x86_64-linux-gnu/gstreamer-1.0
```

The exact meson configuration used for the patched GStreamer build is documented
in `patches/README.md`.

## CUDA feature

The optional `cuda` feature additionally needs the GStreamer-CUDA library
(`gstcuda-1.0`), which lives under `/usr/local` in CUDA-enabled images:

```sh
export PKG_CONFIG_PATH=/usr/local/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
export RUSTFLAGS="-L /usr/local/lib/x86_64-linux-gnu -l gstcuda-1.0"
cargo build -p gst-plugin-wayland-display --features cuda
```
