# NV12 converter benchmark

Quantifies the effect of the `VulkanNv12` pipelining changes (PR #37) on the
**producer's critical path**: the time `VulkanNv12::convert()` spends on the
`waylanddisplaysrc` streaming thread per frame, including any GPU wait.

That time is what the **pipelining lever** removes — pre-change the converter blocks on
`vkWaitForFences` every frame; post-change it submits, hands the export dmabuf an implicit
write fence, and returns while the GPU works asynchronously (on AMD/Intel; Nvidia keeps the
blocking wait because its CUDA consumer can't honor the implicit fence).

## What it measures

`conv-timing.patch` adds a `record_convert_time()` helper to
`wayland-display-core/src/utils/allocator/mod.rs` that times the `convert()` +
`to_gst_buffer()` call and logs running stats (`CONVTIMING n=… avg=…us min=…us max=…us`)
every 60 frames when `CONV_TIMING=1`. The instrumentation lives only here (not in the
shipped source) so the same code can be applied to both the pre-lever and post-lever
builds for an apples-to-apples comparison.

This is a **producer-side** metric, not glass-to-glass latency. The GPU still does the same
conversion work; the win is that producer and encoder now overlap, removing a per-frame
stall (and, on LINEAR, one full-frame copy) and raising the sustainable frame rate.

## How to run

```bash
# 1. apply the instrumentation to the build under test
git apply benchmark/conv-timing.patch

# 2. build (add --features cuda for the Nvidia/dmabuftocuda path)
cargo build -p gst-plugin-wayland-display          # AMD/Intel VA
# cargo build -p gst-plugin-wayland-display --features cuda   # Nvidia

# 3. run a pipeline with timing on (see run-conv-timing.sh for per-GPU examples)
GST_PLUGIN_PATH=target/debug benchmark/run-conv-timing.sh \
  /dev/dri/renderD129 'vah265enc ! fakesink sync=false'

# to compare against the pre-lever build, check out the converter at its parent commit:
#   git show <pre-lever-rev>:wayland-display-core/src/utils/vulkan_nv12.rs > \
#     wayland-display-core/src/utils/vulkan_nv12.rs
# rebuild, and re-run. Revert with: git checkout -- wayland-display-core/...
```

`run-conv-timing.sh` parameters and per-vendor examples are documented at the top of the
script. Note `DRM_FORMAT=NV12` (forces the source's NV12 modifier) is needed for the Nvidia
`dmabuftocuda` path; AMD's `vah265enc` only accepts LINEAR NV12, so run it with
`AMD_DEBUG=nodcc RADV_DEBUG=nodcc`.

## Results

Per-frame source-thread `convert()` cost (average µs over 180–300 frames, **debug** builds,
720p and 4K, 120 fps cap, headless scene). Lower is better.

| GPU | Path exercised | Res | Pre-lever | Post-lever | Δ |
|---|---|---|---:|---:|---|
| Intel UHD 770 | tiled scratch+copy, **lever 1** | 1280×720 | 1265 µs | **59 µs** | **~21× / −1.2 ms** |
| Intel UHD 770 | tiled scratch+copy, **lever 1** | 3840×2160 | 8471 µs | **193 µs** | **~44× / −8.3 ms** |
| AMD RX 7900 XTX | LINEAR direct, **lever 1+2** | 1280×720 | 339 µs | **177 µs** | ~1.9× / −162 µs |
| AMD RX 7900 XTX | LINEAR direct, **lever 1+2** | 3840×2160 | 635 µs | **212 µs** | ~3× / −423 µs |
| Nvidia RTX 5080 | LINEAR direct, blocking (lever 2 only) | 1280×720 | 422 µs | 445 µs | none (within noise) |
| Nvidia RTX 5080 | LINEAR direct, blocking (lever 2 only) | 3840×2160 | 922 µs | 918 µs | none (within noise) |

### Reading the numbers

- **Intel** isolates the pipelining lever (tiled path, both builds scratch+copy). Removing
  the blocking wait takes the producer-thread cost from ~1.3 ms → 60 µs at 720p, and from
  **8.5 ms → 0.2 ms at 4K**. At 120 fps (8.3 ms/frame budget) the pre-lever build spent the
  *entire* budget in the source thread at 4K — it couldn't sustain the rate; post-lever
  leaves it essentially free. The win scales with resolution.
- **AMD** (discrete, fast) shows smaller absolute numbers but a clear 2–3×; it also benefits
  from the LINEAR copy-elision (lever 2).
- **Nvidia** keeps the blocking wait by design (CUDA ignores implicit dma-buf fences), so
  lever 1 doesn't apply; lever 2's copy-elision is negligible against the blocking GPU wait.
  No measurable change — and no regression.

### Caveats

- Producer-thread cost, not end-to-end latency. The GPU work is unchanged; the gain is
  overlap (and raising the sustainable frame rate by un-blocking the source thread).
- Debug builds. Release would lower the post-lever FFI/syscall cost further (pre-lever is
  GPU-wait-dominated and barely moves), widening the ratios.
- AMD ran under `nodcc` (LINEAR) because its `vah265enc` only accepts LINEAR NV12 on
  mesa 25.0.7; the DCC export path wasn't in play.
- `max` spikes are first-frame warm-up / the occasional ring-reuse fence wait; they're rare
  and small relative to the average.

---

## Pre-encoder throughput harness (`bench-e2e.py`)

`bench-e2e.py` measures the **pre-encoder** stage only — source + RGBA→NV12 conversion,
terminating at a `fakesink` right after NV12 (no encoder, which this PR doesn't touch and
which otherwise bottlenecks and hides the difference). It reports sustained throughput at
the `fakesink` (uncapped, `--fps 1000`). `--raw-tail` expresses arbitrary post-source
pipelines (e.g. the Nvidia `dmabuftocuda` path). Run examples are at the top of the script.

### vs main/stable (pre-encoder, true ceiling)

Sustained pre-encoder throughput **with the caches in place**, source uncapped
(`--fps 8000`, since a lower request rate just paces the live source), debug builds.
**stable** = RGBA source → `vapostproc` (AMD/Intel) / `glupload ! glcolorconvert` (Nvidia);
**PR** = in-source Vulkan converter → NV12 (`dmabuftocuda` on Nvidia). Higher is better.

| GPU | Res | stable | PR | |
|---|---|---:|---:|---|
| Intel UHD 770 | 720p | 951 fps | 919 fps | stable +3% |
| Intel UHD 770 | 4K | 234 fps | 174 fps | stable +35% |
| AMD RX 7900 XTX | 720p | 1529 fps | **1682 fps** | PR +10% |
| AMD RX 7900 XTX | 4K | 1004 fps | **1621 fps** | PR +61% |
| Nvidia RTX 5080 | 720p | 2875 fps | **3765 fps** | PR +31% |
| Nvidia RTX 5080 | 4K | 2571 fps | **3065 fps** | PR +19% |

With the per-frame import and EGLImage/CUDA-registration caches in place, the PR is
**faster than stable on AMD and Nvidia** (notably AMD 4K, +61%). It still trails on
**Intel**, where the RGBA-render and the Vulkan convert serialise on the same GPU in the
source thread, whereas stable's `vapostproc` runs the convert on a separate VEBOX engine
that overlaps the 3D render. These are uncapped (academic) ceilings — at real streaming
rates (60–240 fps) every path has ample headroom, so latency and robustness are what matter
in practice; this table just shows the conversion stage isn't a throughput regression except
on Intel.

### Nvidia `dmabuftocuda` EGLImage cache

`dmabuftocuda` originally re-created the EGLImage + CUDA registration every frame, which
made it the dominant per-frame cost. Caching the registration per ring fd nearly doubles
pre-encoder throughput on an RTX 5080:

| | 720p | 4K |
|---|---:|---:|
| before cache | 229 fps | 219 fps |
| **after cache** | **426 fps** | **409 fps** |

Two related notes:
- Without the cache the serial source→`dmabuftocuda` execution (one thread) was the ceiling;
  a `! queue !` between them recovers it too (229→424 fps). With the cache the element is
  cheap enough that the queue is no longer required.
- The blocking Vulkan convert on Nvidia (CUDA can't honor implicit dma-buf fences) is *not*
  the bottleneck — ~445µs of a ~2.3ms source stage. Un-blocking it would need explicit
  Vulkan→CUDA semaphore interop (`cudaImportExternalSemaphore` +
  `cudaWaitExternalSemaphoresAsync`), a low-ROI follow-up once the cache is in place.
