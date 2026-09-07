# Build / test / benchmark harness

`ci/harness.sh` is one entry point that builds the workspace, runs the tests,
runs the converter benchmarks, and runs per-vendor encode integration smoke
tests against whatever GPU(s) the machine has. It is the same thing CI should
run and the thing to run locally before pushing.

```bash
ci/harness.sh                 # every phase relevant to the detected hardware
ci/harness.sh detect          # just print what it found
ci/harness.sh build unit      # only those phases
ci/harness.sh --features cuda integration   # force the cuda build + smoke tests
ci/harness.sh -v              # stream sub-command output
```

## Phases

| phase         | what it does |
|---------------|--------------|
| `detect`      | render nodes → vendor, available encoder elements, libclang, XDG_RUNTIME_DIR |
| `lint`        | `cargo fmt --all --check` |
| `build`       | `cargo build` (default **and** `--features cuda` when an Nvidia GPU is present, so a `#[cfg(feature = "cuda")]` caller can't rot unnoticed) |
| `unit`        | `cargo test --workspace` (non-ignored) |
| `gpu`         | `cargo test -- --ignored` with the render node wired into `NV12_TEST_NODE` / `VULKAN_ENC_NODE` (drives the `tests/encode.rs` per-vendor encode tests) |
| `integration` | `waylanddisplaysrc ! <vendor encoder> ! fakesink` to EOS under `G_DEBUG=fatal-criticals` (so a teardown CRITICAL fails the run) |
| `bench`       | applies `benchmark/conv-timing.patch`, runs `run-conv-timing.sh` per GPU, reports per-frame `convert()` cost, reverts the patch |

## Vendor matrix (auto-detected)

| vendor | render node driver | encoder path |
|--------|--------------------|--------------|
| AMD    | `amdgpu`           | `vah265enc` (NV12 dmabuf, LINEAR) |
| Intel  | `i915`/`xe`        | `vah265lpenc` (NV12 dmabuf, Y-tiled; low-power only) |
| Nvidia | `nvidia`           | `dmabuftocuda ! nvh265enc` (needs the `cuda` feature) |
| any    | any with `vulkanh264enc` | `vulkan=true ! vulkanh264enc` (shared-device NV12 `memory:VulkanImage`, zero-copy) |

A vendor with no render node, or whose encoder element is missing, is skipped
(reported, not failed). The Vulkan-encode row is vendor-agnostic — it runs
wherever `vulkanh264enc` registers (an nvidia render node, or Intel with
`ANV_VIDEO_ENCODE=1`); the `integration` phase prefers the nvidia node.

## Host prerequisites

- rust toolchain (`rustup`), `gst-launch-1.0` + gst-plugins-{base,bad,vaapi}
- build deps: `libwayland-dev libinput-dev libxkbcommon-dev libgbm-dev libegl1-mesa-dev libudev-dev libclang-dev pkg-config`
- a C compiler (`gcc`/`clang`) and the Vulkan loader headers (`libvulkan-dev`, or `vulkan-loader-devel` on Fedora) — `build.rs` compiles `vulkan_bridge.c` against the gst Vulkan headers
- a Vulkan driver (`mesa-vulkan-drivers` for AMD/Intel, the proprietary driver for Nvidia) — the converter queries Vulkan for NV12 export modifiers; without it the source advertises nothing and negotiation fails
- `hwdata` (for `/usr/share/hwdata/pci.ids`) so the GPU-name lookup test passes
- Nvidia only: the `cuda` feature links `libgstcuda-1.0`; if your distro ships it without a `.pc`, add a `gstreamer-cuda-1.0.pc` (`Libs: -lgstcuda-1.0`) and a `libgstcuda-1.0.so` dev symlink

## Notes

- `tests/encode.rs` holds the expanded, auto-detecting integration tests. They
  are `#[ignore]` so a GPU-less `cargo test` stays green; the `gpu` phase runs
  them. They make any GLib `CRITICAL` fatal, so teardown refcount bugs fail.
- The `vulkanh265enc`/`vulkanh264enc` Tier-1 path needs a gstreamer `main` build
  with Vulkan-Headers ≥ 1.4.317 and a recent mesa radv; it is not covered by the
  default matrix.
