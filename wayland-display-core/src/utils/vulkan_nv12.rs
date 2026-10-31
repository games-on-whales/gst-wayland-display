//! Cross-vendor Vulkan RGBA->NV12 converter.
//!
//! The compositor renders the scene with the (GLES) smithay renderer to an **RGBA
//! dmabuf**; this stage imports that dmabuf into Vulkan (`VK_EXT_external_memory_dma_buf`
//! -- *not* glupload, so none of the second-GL-context / teardown problems), runs a
//! compute shader (`shaders/rgba_to_nv12.comp`, BT.601 limited range) into the planes
//! of an NV12 image, and **exports an NV12 dmabuf** the downstream encoder imports
//! zero-copy:
//!   - AMD/Intel: `vah265enc`/`vaapi` (VAAPI imports the dmabuf directly)
//!   - Nvidia:    `dmabuftocuda -> nvh265enc` (dmabuf->CUDA)
//!
//! ## Two output paths (lever 2: skip the copy on LINEAR)
//!
//! A compute `imageStore` only works on a `STORAGE`-capable image, which on AMD means a
//! **LINEAR** NV12 modifier -- radv reports `STORAGE=false` for the tiled modifiers. But
//! a tiled modifier *does* support `TRANSFER_DST`, and a `vkCmdCopyImage` goes through
//! radv's tiling-maintaining path. So for a **tiled** modifier we:
//!   1. compute RGBA -> NV12 into a LINEAR *scratch* image (storage works there), then
//!   2. copy scratch -> the tiled export image.
//! When the negotiated modifier is **LINEAR** the export image itself supports `STORAGE`,
//! so the compute shader writes it **directly** -- no scratch, no copy (one fewer
//! full-frame write of bandwidth per frame). Selected per instance in [`VulkanNv12::new`].
//!
//! ## Modifier negotiation
//!
//! We advertise the *full* set of NV12 modifiers the GPU's Vulkan can export; gst caps
//! negotiation intersects that with what the downstream encoder accepts and picks the
//! encoder's preferred one. On every GPU tested this lands on LINEAR (7900 XTX) or the
//! Intel Y-tiled modifier when the encoder offers them. **Do not** filter modifiers here:
//! the RX 9070 (GFX12/RDNA4) `vah265enc` advertises *only* a DCC modifier
//! (`NV12:0x0200000000082305`) for DMABuf, so dropping it leaves no common modifier and
//! the pipeline fails `not-negotiated`. (radeonsi-VA still can't import radv's DCC
//! metadata as an encode source on GFX12 -- that surfaces later as "Failed to create the
//! reconstruct picture" -- but that's a driver limitation to solve via the VAMemory path,
//! not by hiding the only modifier the encoder will negotiate.)
//!
//! ## Pipelining (lever 1: no per-frame CPU stall)
//!
//! The export side is an N-deep ring (`RING`): VA caches imported surfaces by dmabuf fd,
//! so a single reused image would alias the current frame with a reference frame. Each
//! ring slot is **self-contained** -- its own command buffer, fence, descriptor set,
//! scratch (tiled path), and import-image holder -- so up to `RING` conversions can be in
//! flight. Instead of blocking the CPU on every frame, on implicit-sync GPUs (AMD/Intel)
//! we export a `SYNC_FD` from the submit's semaphore and `DMA_BUF_IOCTL_IMPORT_SYNC_FILE`
//! it onto the export dmabuf as a **write fence**, so the VA encoder's read is ordered
//! after the compute/copy by the kernel's implicit dma-buf sync -- no `vkWaitForFences`
//! on the producer's critical path. Each slot's import image is reclaimed by waiting that
//! slot's own (RING-frames-old, already-signalled) fence when the slot is next reused.
//!
//! Nvidia's CUDA consumer (`dmabuftocuda`) does **not** honor implicit dma-buf fences
//! (CUDA external-memory import tracks no `dma_resv`), so on an Nvidia device we keep the
//! blocking `vkWaitForFences` -- correct, just not pipelined. The path is chosen
//! automatically from the selected Vulkan device's `vendorID`.
//!
//! Validated on real hardware (RX 7900 XTX): the RGBA->NV12 compute produces the exact
//! BT.601 values (RGB(48,96,192) -> Y=95/Cb=177/Cr=100), and the source advertises the
//! GPU's Vulkan-exportable NV12 modifiers ([`supported_nv12_modifiers`]) so negotiation
//! lands on a modifier the encoder also accepts.

// This module is a thin Vulkan FFI layer: nearly every call is an `unsafe` ash call and
// the helpers are `unsafe fn`, so per-op `unsafe {}` blocks would be pure noise here.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::utils::color::{ContentTransfer, FrameColor};
use ash::vk;
use gst::Buffer as GstBuffer;
use gst_video::{VideoFormat, VideoInfoDmaDrm, VideoMeta};
use gstreamer_allocators::{DmaBufAllocator, DmaBufAllocatorExtManual, FdMemoryFlags};
use smithay::backend::allocator::Buffer as _;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::drm::DrmNode;
use smithay::reexports::drm::buffer::DrmFourcc;
use std::os::fd::{AsFd, AsRawFd, IntoRawFd, RawFd};
use std::sync::{Mutex, OnceLock, PoisonError};

const RGBA_TO_NV12_SPV: &[u8] = include_bytes!("shaders/rgba_to_nv12.spv");
const RGBA_TO_P010_SPV: &[u8] = include_bytes!("shaders/rgba_to_p010.spv");
/// BT.2020 (BT.2100-PQ HDR) variant of the P010 converter: identical topology to
/// [`RGBA_TO_P010_SPV`] but with the BT.2020 luma/chroma matrix, selected when the producer
/// signals HDR output so the matrix matches the `matrix=bt2020` caps tagging.
const RGBA_TO_P010_BT2020_SPV: &[u8] = include_bytes!("shaders/rgba_to_p010_bt2020.spv");
/// HDR render-path spike (`WOLF_HDR_SPIKE`) variant of the BT.2020/PQ P010 converter. Same
/// topology/bindings as [`RGBA_TO_P010_BT2020_SPV`], but its input is the **linear fp16** RGBA
/// render target (1.0 == SDR reference white, may exceed 1.0) instead of 8-bit sRGB, so it skips
/// the sRGB EOTF and tone-maps the already-linear value straight to PQ -- proving a >1.0 value
/// can travel render-target -> converter -> P010 PQ as a brighter-than-white highlight. No
/// pipeline selects it: the spike renders through the regular converters, and this shader is
/// kept only to reproduce that proof by hand. NOTE: the checked-in `.spv` is a compiled
/// shader; regenerate it with `ci/shaders.sh --update` after changing the GLSL.
#[allow(dead_code)]
const RGBA_TO_P010_HDR_SPV: &[u8] = include_bytes!("shaders/rgba_to_p010_hdr.spv");
/// PQ-passthrough variant of the P010 converter (dynamic HDR). Same topology/bindings as
/// [`RGBA_TO_P010_BT2020_SPV`], but it applies ONLY the BT.2020 limited-range Y'CbCr matrix --
/// no sRGB EOTF, no 709->2020 gamut, no PQ OETF -- because its input is already PQ-encoded
/// BT.2020 (a surface that declared PQ, composited into the fp16 target), so it isn't
/// double-PQ'd (washed out). NOTE: the checked-in `.spv` is a compiled shader; regenerate it
/// with `ci/shaders.sh --update` after changing the GLSL.
const RGBA_PQPASS_TO_P010_SPV: &[u8] = include_bytes!("shaders/rgba_pqpass_to_p010.spv");
/// HDR content -> SDR BT.709 P010 (dynamic HDR), for frames tagged SDR while the active surface
/// declares PQ content. Tone-maps instead of showing
/// HDR code values as gamma.
const RGBA_HDR_TO_P010_SDR_SPV: &[u8] = include_bytes!("shaders/rgba_hdr_to_p010_sdr.spv");
/// HDR content -> SDR BT.601 NV12 (dynamic HDR): the NV12 sibling of
/// [`RGBA_HDR_TO_P010_SDR_SPV`]. NV12 output never goes HDR, but an HDR client can still
/// declare PQ content on it; this tone-maps that content instead of showing it washed out.
const RGBA_HDR_TO_NV12_SDR_SPV: &[u8] = include_bytes!("shaders/rgba_hdr_to_nv12_sdr.spv");
const DRM_FORMAT_MOD_LINEAR: u64 = 0;
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
/// NV12 export ring depth (> encoder DPB/pipeline depth so a buffer is free by reuse).
// Ring depth: slots cycled to pipeline convert+encode (array sizing + default active depth).
// 4 is enough fan-out for the downstream encoder's buffer references (1 starves; 4 and 8 are
// indistinguishable in practice). The per-element `ring-depth` property overrides this
// default before the converter is created, clamped to MIN_RING..=RING.
const RING: usize = 4;
/// A slot is only reused once downstream released it, and a busy ring re-emits the latest
/// output; with one slot, anything holding the latest buffer (a sink's last-sample, a queue,
/// an encoder's one frame of latency) would pin it and freeze the stream on one frame.
const MIN_RING: usize = 2;

fn ring_depth(options: &ConversionOptions) -> usize {
    options.ring_depth.clamp(MIN_RING, RING)
}

/// Optional path for a one-shot debug dump of the converter's NV12 output (the LINEAR
/// compute scratch, *before* the tiled copy) as raw NV12. Set
/// `WOLF_VULKAN_DUMP=<file>` to capture one frame -- if it's clean, the RX 9070
/// green-bar/jump corruption is introduced downstream (the tiled copy or the encoder), not
/// our conversion. Read once; off (zero overhead) when unset.
fn dump_path() -> Option<&'static str> {
    use std::sync::OnceLock;
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| {
        std::env::var("WOLF_VULKAN_DUMP")
            .ok()
            .filter(|s| !s.is_empty())
    })
    .as_deref()
}
static DUMP_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Which converted frame to capture when `WOLF_VULKAN_DUMP` is set (default 20, a few
/// frames in so the scene has settled). Override with `WOLF_VULKAN_DUMP_FRAME` to wait
/// longer -- e.g.
/// for a client to connect and paint a known colour before the dump fires. Read once.
fn dump_frame_target() -> u64 {
    use std::sync::OnceLock;
    static F: OnceLock<u64> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("WOLF_VULKAN_DUMP_FRAME")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(20)
    })
}

/// PCI vendor id for Nvidia -- its CUDA consumer ignores implicit dma-buf fences.
const VENDOR_NVIDIA: u32 = 0x10de;
/// `VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR` -- the layout `vulkanh264enc` expects its input.
const VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR: i32 = 1_000_299_001;

type Err = Box<dyn std::error::Error>;
/// An imported RGBA dmabuf (image/memory/view) kept alive until its frame's GPU work ends.
type Import = (vk::Image, vk::DeviceMemory, vk::ImageView);

/// Output pixel format the converter targets. NV12 is the 8-bit 4:2:0 default; P010 is the
/// 10-bit 4:2:0 path (for `vulkanh265enc` Main-10). Identical compute math and pipeline
/// topology -- P010 just swaps the multiplanar image format and per-plane storage views for
/// their 16-bit equivalents and selects the 16-bit shader. The compositor content is 8-bit
/// RGBA, so P010 is a valid 10-bit container of 8-bit-precision content (true HDR later).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixFmt {
    Nv12,
    P010,
}

impl PixFmt {
    /// The multiplanar Vulkan format of the output/scratch image.
    fn image_format(self) -> vk::Format {
        match self {
            PixFmt::Nv12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
            PixFmt::P010 => vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
        }
    }
    /// Per-plane storage-view format the compute shader imageStores into (Y plane). For P010
    /// the plane is `R10X6_UNORM_PACK16`; we view it as `R16_UNORM` -- both are in Vulkan's
    /// 16-bit format-compatibility class, so the view is size/class-compatible -- and a
    /// normalized store lands across all 16 bits (the P010 reader takes the top 10).
    fn y_view_format(self) -> vk::Format {
        match self {
            PixFmt::Nv12 => vk::Format::R8_UNORM,
            PixFmt::P010 => vk::Format::R16_UNORM,
        }
    }
    /// Per-plane storage-view format for the interleaved Cb/Cr plane (`R10X6G10X6` <-> R16G16,
    /// 32-bit class, compatible).
    fn uv_view_format(self) -> vk::Format {
        match self {
            PixFmt::Nv12 => vk::Format::R8G8_UNORM,
            PixFmt::P010 => vk::Format::R16G16_UNORM,
        }
    }
    pub(crate) fn gst_format(self) -> VideoFormat {
        match self {
            PixFmt::Nv12 => VideoFormat::Nv12,
            PixFmt::P010 => VideoFormat::P01010le,
        }
    }
    /// dmabuf fourcc (little-endian 4cc) for the export VideoMeta / VA layout.
    fn fourcc(self) -> u32 {
        match self {
            PixFmt::Nv12 => u32::from_le_bytes(*b"NV12"),
            PixFmt::P010 => u32::from_le_bytes(*b"P010"),
        }
    }
    /// Bytes per luma sample (NV12 = 1, P010 = 2) -- sizes the host-readback debug dump.
    fn y_bytes(self) -> u64 {
        match self {
            PixFmt::Nv12 => 1,
            PixFmt::P010 => 2,
        }
    }
    /// Pick the matching compute shader SPIR-V for a fixed (non-dynamic) output. `bt2020`
    /// selects the BT.2020 (HDR / BT.2100-PQ) sRGB-input tone-map on the P010 path; it never
    /// affects NV12 (8-bit SDR stays BT.601).
    fn shader(self, bt2020: bool) -> &'static [u8] {
        match (self, bt2020) {
            (PixFmt::Nv12, _) => RGBA_TO_NV12_SPV,
            (PixFmt::P010, false) => RGBA_TO_P010_SPV,
            (PixFmt::P010, true) => RGBA_TO_P010_BT2020_SPV,
        }
    }
    /// Derive from a negotiated gst video format (anything but P010 -> NV12).
    pub fn from_gst(format: VideoFormat) -> Self {
        match format {
            VideoFormat::P01010le => PixFmt::P010,
            _ => PixFmt::Nv12,
        }
    }
}

// --- dma-buf implicit-sync ioctl (attach a Vulkan signal as the dmabuf's write fence) ---

#[repr(C)]
struct DmaBufImportSyncFile {
    flags: u32,
    fd: i32,
}
const DMA_BUF_SYNC_READ: u32 = 1 << 0;
const DMA_BUF_SYNC_WRITE: u32 = 1 << 1;
/// How long a colorimetry transition may wait, in total, for a free encode-src slot before
/// it degrades to the busy-ring behaviour (re-emit the previous output) instead of stalling
/// the compositor thread or ending the stream.
const TRANSITION_SLOT_WAIT: std::time::Duration = std::time::Duration::from_millis(100);
/// `_IOW('b', 3, struct dma_buf_import_sync_file)` from `<linux/dma-buf.h>`.
const fn iow(ty: u32, nr: u32, size: u32) -> libc::c_ulong {
    ((1u32 << 30) | (size << 16) | (ty << 8) | nr) as libc::c_ulong
}
const DMA_BUF_IOCTL_IMPORT_SYNC_FILE: libc::c_ulong =
    iow(0x62, 3, std::mem::size_of::<DmaBufImportSyncFile>() as u32);

/// The DRM minor of a render-node path (e.g. `/dev/dri/renderD128` -> 128), so Vulkan
/// device selection can target the same GPU the compositor renders on. `None` if the path
/// can't be stat-ed.
pub fn render_node_minor(path: &str) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    let rdev = std::fs::metadata(path).ok()?.rdev();
    // glibc minor(3): bits 0-7 plus 20+.
    Some(((rdev & 0xff) | ((rdev >> 12) & 0xffff_ff00)) as u32)
}

/// Pick the Vulkan physical device whose DRM render (or primary) minor matches
/// `target_minor`, so on a multi-GPU host we use the same GPU that owns the compositor's
/// render node. A render-node path resolves to the device's *render* minor, a primary
/// (`card*`) path to its *primary* minor -- we accept either. Falls back to the first
/// device, with a warning, when nothing matches (or the minor is unknown).
/// NV12/P010 are 4:2:0 (one chroma sample per 2x2 block), and Vulkan only creates 4:2:0
/// images with even extents; the dispatch and the chroma copies assume it too.
fn check_420_extent(width: u32, height: u32) -> Result<(), Err> {
    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(
            format!("4:2:0 output needs a non-zero even size, got {width}x{height}").into(),
        );
    }
    Ok(())
}

unsafe fn pick_physical_device(
    instance: &ash::Instance,
    target_minor: Option<u32>,
) -> Option<vk::PhysicalDevice> {
    physical_device_for_minor(instance, target_minor, |_| true)
}

/// The physical device backing DRM minor `target_minor`, among the GPUs `usable` accepts.
///
/// Never another GPU than the one named: the compositor renders on `target_minor`, and its
/// dmabuf cannot be imported (or not correctly) into a device on another GPU. A device whose
/// DRM minor can't be read (no `VK_EXT_physical_device_drm`) is only taken when it is the
/// only usable GPU. With no minor (an unparsable node), the first usable GPU. CPU devices
/// (llvmpipe/lavapipe) are never picked.
pub(crate) unsafe fn physical_device_for_minor(
    instance: &ash::Instance,
    target_minor: Option<u32>,
    usable: impl Fn(vk::PhysicalDevice) -> bool,
) -> Option<vk::PhysicalDevice> {
    let devices: Vec<vk::PhysicalDevice> = instance
        .enumerate_physical_devices()
        .ok()?
        .into_iter()
        .filter(|&d| {
            instance.get_physical_device_properties(d).device_type != vk::PhysicalDeviceType::CPU
                && usable(d)
        })
        .collect();
    let Some(minor) = target_minor else {
        return devices.first().copied();
    };
    // PhysicalDeviceDrmPropertiesEXT may only be chained when the device exposes
    // VK_EXT_physical_device_drm.
    let drm_minors = |d: vk::PhysicalDevice| {
        let has_drm = instance
            .enumerate_device_extension_properties(d)
            .is_ok_and(|exts| {
                exts.iter()
                    .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::physical_device_drm::NAME))
            });
        has_drm.then(|| {
            let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
            instance.get_physical_device_properties2(d, &mut p2);
            drm
        })
    };
    let mut unknown = Vec::new();
    for &d in &devices {
        match drm_minors(d) {
            Some(drm)
                if (drm.has_render != 0 && drm.render_minor == i64::from(minor))
                    || (drm.has_primary != 0 && drm.primary_minor == i64::from(minor)) =>
            {
                return Some(d);
            }
            Some(_) => {}
            None => unknown.push(d),
        }
    }
    if devices.len() == 1 && unknown.len() == 1 {
        tracing::info!(
            "no Vulkan device reports DRM minor {minor}; using the only GPU, which cannot report it"
        );
        return Some(unknown[0]);
    }
    tracing::warn!(
        "no usable Vulkan device matches DRM minor {minor}; not falling back to another GPU"
    );
    None
}

/// DRM modifiers `target_minor`'s GPU can export NV12 with (queried once per GPU, cached;
/// a failed query is retried a few seconds later rather than cached as "no modifiers").
/// The source advertises all of these so caps negotiation lands on a modifier the encoder
/// also accepts -- the converter then exports that exact modifier (LINEAR directly, tiled
/// via a transfer copy), so no LINEAR-vs-tiled mismatch and no guessing.
pub fn supported_nv12_modifiers(target_minor: Option<u32>) -> &'static [u64] {
    supported_modifiers(target_minor, PixFmt::Nv12)
}

/// DRM modifiers `target_minor`'s GPU can export P010 with -- the 10-bit sibling of
/// [`supported_nv12_modifiers`], advertised so a P010 negotiation lands on a Vulkan-exportable
/// modifier.
pub fn supported_p010_modifiers(target_minor: Option<u32>) -> &'static [u64] {
    supported_modifiers(target_minor, PixFmt::P010)
}

fn supported_modifiers(target_minor: Option<u32>, fmt: PixFmt) -> &'static [u64] {
    /// A failed probe is retried after this long: it may be transient, but caps queries are
    /// frequent and each probe creates a Vulkan instance under the cache lock.
    const RETRY_FAILED: std::time::Duration = std::time::Duration::from_secs(5);
    type Probe = Result<&'static [u64], std::time::Instant>;
    static CACHE: OnceLock<Mutex<Vec<(Option<u32>, PixFmt, Probe)>>> = OnceLock::new();
    let mut guard = CACHE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    let cached = guard
        .iter()
        .position(|(m, f, _)| *m == target_minor && *f == fmt);
    match cached.map(|i| guard[i].2) {
        Some(Ok(mods)) => return mods,
        Some(Err(failed_at)) if failed_at.elapsed() < RETRY_FAILED => return &[],
        _ => {}
    }
    let probe = match unsafe { query_modifiers(target_minor, fmt.image_format()) } {
        Some(mods) => Ok(&*Box::leak(mods.into_boxed_slice())),
        None => Err(std::time::Instant::now()),
    };
    match cached {
        Some(i) => guard[i].2 = probe,
        None => guard.push((target_minor, fmt, probe)),
    }
    probe.unwrap_or(&[])
}

unsafe fn query_modifiers(target_minor: Option<u32>, format: vk::Format) -> Option<Vec<u64>> {
    crate::utils::renderer::keep_vulkan_drivers_loaded();
    let entry = ash::Entry::load().ok()?;
    let instance = entry
        .create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2)),
            None,
        )
        .ok()?;
    let Some(pd) = pick_physical_device(&instance, target_minor) else {
        instance.destroy_instance(None);
        return None;
    };
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut p2 = vk::FormatProperties2::default().push_next(&mut list);
    instance.get_physical_device_format_properties2(pd, format, &mut p2);
    let mut props = vec![
        vk::DrmFormatModifierPropertiesEXT::default();
        list.drm_format_modifier_count as usize
    ];
    let mut list2 = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut props);
    let mut p2b = vk::FormatProperties2::default().push_next(&mut list2);
    instance.get_physical_device_format_properties2(pd, format, &mut p2b);
    // Keep only modifiers that support TRANSFER_DST (the copy target) -- every export
    // modifier the encoder might want is filled via vkCmdCopyImage. We advertise the full
    // set (incl. AMD DCC): gst negotiation picks the encoder's preferred modifier, and on
    // every GPU tested it lands on LINEAR/Y-tiled when the encoder offers it; on a GPU
    // whose encoder *only* accepts DCC (RX 9070 / GFX12), DCC is the sole option, so we
    // must not drop it or negotiation fails outright. Only modifiers with exactly the two
    // format planes as memory planes, though: the export describes planes 0 and 1 only, so a
    // modifier with extra metadata planes (GFX9-11 DCC, Intel CCS) would reach the consumer
    // with an incomplete layout.
    let mods = props
        .iter()
        .filter(|m| {
            m.drm_format_modifier_plane_count == 2
                && m.drm_format_modifier_tiling_features
                    .contains(vk::FormatFeatureFlags::TRANSFER_DST)
        })
        .map(|m| m.drm_format_modifier)
        .collect();
    instance.destroy_instance(None);
    Some(mods)
}

/// The physical device matching `target_minor` and a logical device on it with the
/// converter's extensions. Nothing is left to free on failure (the caller owns `instance`).
struct OpenedDevice {
    pd: vk::PhysicalDevice,
    device: ash::Device,
    qfi: u32,
    /// The device can export a submit's signal as a SYNC_FD (and is not Nvidia), so the
    /// converter pipelines through dmabuf fences instead of blocking.
    implicit_sync: bool,
    queue_family_foreign: bool,
}

unsafe fn open_device(instance: &ash::Instance, target_minor: u32) -> Result<OpenedDevice, Err> {
    // Match by DRM render/primary minor (VK_EXT_physical_device_drm); warns and falls
    // back to the first device if no driver reports a matching node.
    let pd = pick_physical_device(instance, Some(target_minor)).ok_or("no vulkan device")?;

    // Implicit dma-buf sync (kernel write-fence on the export buffer) is honored by
    // the VA consumer on AMD/Intel but not by CUDA on Nvidia -- so only pipeline
    // without a CPU stall when the selected device is not Nvidia.
    let dev_props = instance.get_physical_device_properties(pd);
    let implicit_sync = dev_props.vendor_id != VENDOR_NVIDIA;
    tracing::debug!(
        "VulkanNv12: vendor={:#x} implicit_sync(pipelined)={implicit_sync}",
        dev_props.vendor_id
    );

    // We export exactly the negotiated modifier (the caps layer already picked one the
    // encoder imports -- LINEAR for the interpipe/vapostproc path, the encoder's own
    // modifier for a direct `! vah265enc`). No env-gated override.

    let qfi = instance
        .get_physical_device_queue_family_properties(pd)
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .ok_or("no compute queue")? as u32;
    let prio = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qfi)
        .queue_priorities(&prio)];
    // Only enable what the device supports: required extensions fail the converter with
    // a clear error, optional ones degrade (blocking fence wait, EXTERNAL family).
    let available: Vec<std::ffi::CString> = instance
        .enumerate_device_extension_properties(pd)?
        .iter()
        .filter_map(|e| e.extension_name_as_c_str().ok().map(|n| n.to_owned()))
        .collect();
    let has = |name: &std::ffi::CStr| available.iter().any(|a| a.as_c_str() == name);
    let mut exts = Vec::new();
    for name in [
        ash::khr::external_memory_fd::NAME,
        ash::ext::external_memory_dma_buf::NAME,
        ash::ext::image_drm_format_modifier::NAME,
    ] {
        if !has(name) {
            return Err(format!("Vulkan device lacks {name:?}").into());
        }
        exts.push(name.as_ptr());
    }
    // Needed to export the submit's signal as a SYNC_FD we attach to the dmabuf; without
    // it the converter blocks on its fence like the Nvidia path.
    let implicit_sync = implicit_sync && has(ash::khr::external_semaphore_fd::NAME);
    if implicit_sync {
        exts.push(ash::khr::external_semaphore_fd::NAME.as_ptr());
    }
    let queue_family_foreign = has(ash::ext::queue_family_foreign::NAME);
    if queue_family_foreign {
        exts.push(ash::ext::queue_family_foreign::NAME.as_ptr());
    }
    let mut supported_ycbcr = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut supported_ycbcr);
    instance.get_physical_device_features2(pd, &mut features);
    let mut ycbcr = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
        .sampler_ycbcr_conversion(supported_ycbcr.sampler_ycbcr_conversion != 0);
    let device = instance.create_device(
        pd,
        &vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&exts)
            .push_next(&mut ycbcr),
        None,
    )?;
    Ok(OpenedDevice {
        pd,
        device,
        qfi,
        implicit_sync,
        queue_family_foreign,
    })
}

/// A converter's own instance and device, returned when the converter drops so the next
/// converter on the same GPU (the next session, or this one after a renegotiation) reuses
/// them instead of destroying and recreating them.
///
/// Destroying a device is not safe in a process that renders with NVIDIA's EGL:
/// `vkDestroyDevice` holds a driver lock while it joins the device's worker thread, the
/// worker's exit waits on a driver mutex that a thread inside an EGL call (any session's
/// compositor) holds, and that thread waits on the destroy's lock. The core GPU tests hung
/// in about 1 run in 120 on a 615.71 driver; reused devices are never destroyed.
struct PooledDevice {
    minor: u32,
    entry: ash::Entry,
    instance: ash::Instance,
    opened: OpenedDevice,
}

static DEVICE_POOL: Mutex<Vec<PooledDevice>> = Mutex::new(Vec::new());

/// An idle device for `target_minor` from the pool, or a new one.
unsafe fn take_device(target_minor: u32) -> Result<(ash::Entry, ash::Instance, OpenedDevice), Err> {
    crate::utils::renderer::keep_vulkan_drivers_loaded();
    let pooled = {
        let mut pool = DEVICE_POOL.lock().unwrap_or_else(PoisonError::into_inner);
        pool.iter()
            .position(|d| d.minor == target_minor)
            .map(|i| pool.swap_remove(i))
    };
    if let Some(d) = pooled {
        return Ok((d.entry, d.instance, d.opened));
    }
    let entry = ash::Entry::load()?;
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let instance = entry.create_instance(
        &vk::InstanceCreateInfo::default().application_info(&app),
        None,
    )?;
    match open_device(&instance, target_minor) {
        Ok(opened) => Ok((entry, instance, opened)),
        Err(e) => {
            instance.destroy_instance(None);
            Err(e)
        }
    }
}

/// Map a compositor RGBA dmabuf fourcc to the Vulkan format used for the import.
fn rgba_vk_format(fourcc: DrmFourcc) -> Option<vk::Format> {
    Some(match fourcc {
        DrmFourcc::Abgr8888 | DrmFourcc::Xbgr8888 => vk::Format::R8G8B8A8_UNORM,
        DrmFourcc::Argb8888 | DrmFourcc::Xrgb8888 => vk::Format::B8G8R8A8_UNORM,
        // Dynamic HDR: the compositor renders into an fp16 RGBA dmabuf so HDR client values
        // (including PQ code values) survive composition unclamped.
        DrmFourcc::Abgr16161616f => vk::Format::R16G16B16A16_SFLOAT,
        _ => return None,
    })
}

/// One exported NV12 image in the output ring, plus everything needed to fill it
/// independently of the other slots (so `RING` frames can be in flight at once).
struct Nv12Out {
    image: vk::Image,
    mem: vk::DeviceMemory,
    // Built once over the exported dmabuf fd; re-emitted (ref-counted) every time this
    // slot is used so the VA encoder reuses one cached surface per slot instead of
    // creating a new VA surface every frame (which exhausts the driver's surface budget).
    buffer: GstBuffer,
    /// A dup of the export dmabuf fd, kept so we can attach a write fence (sync_file) to
    /// the underlying BO every frame for implicit-sync consumers.
    export_fd: RawFd,
    /// Tiled/DCC path only: LINEAR compute scratch (storage), copied into `image`.
    /// `None` on the LINEAR direct path (compute writes `image` directly).
    scratch: Option<(vk::Image, vk::DeviceMemory)>,
    /// Per-plane storage views the compute shader writes: scratch planes (tiled) or
    /// `image` planes (LINEAR direct).
    y_view: vk::ImageView,
    uv_view: vk::ImageView,
    /// Per-slot recording/sync state, so slots don't contend (lever 1 pipelining).
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_set: vk::DescriptorSet,
    /// Cached RGBA import (keyed by the source dmabuf device/inode): the VkImage/memory/view over the
    /// compositor's RGBA dmabuf. The compositor renders into a single stable buffer, so
    /// this is created once per slot and reused every frame -- avoiding a per-frame
    /// create_image + dma-buf memory import + view. The import views the dmabuf the
    /// compositor overwrites in place, so a reused import reads the latest content. Rebuilt
    /// only if the backing object changes; freed at drop.
    import: Option<((u64, u64), Import)>,
    /// Whether `fence` has un-reclaimed GPU work (i.e. `cmd` is still in use).
    in_flight: bool,
}

impl Nv12Out {
    /// A slot with nothing created yet (null handles, no export fd), filled in place.
    fn empty() -> Self {
        Nv12Out {
            image: vk::Image::null(),
            mem: vk::DeviceMemory::null(),
            buffer: GstBuffer::new(),
            export_fd: -1,
            scratch: None,
            y_view: vk::ImageView::null(),
            uv_view: vk::ImageView::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            desc_set: vk::DescriptorSet::null(),
            import: None,
            in_flight: false,
        }
    }
}

/// Owns the Vulkan device, the compute pipeline, and the NV12 export ring.
pub struct VulkanNv12 {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    /// gst's `GstVulkanQueue` behind `queue` on the shared-device path: every submit holds
    /// its submit lock, since the encoders on that device submit to the same `VkQueue`.
    gst_queue: Option<gstreamer_vulkan::VulkanQueue>,
    /// Our queue family, the destination of dmabuf acquires and source of releases.
    qfi: u32,
    /// The family on the other side of those transfers: QUEUE_FAMILY_FOREIGN_EXT when the
    /// device has VK_EXT_queue_family_foreign, else QUEUE_FAMILY_EXTERNAL.
    foreign_qf: u32,
    cmd_pool: vk::CommandPool,
    pipeline: vk::Pipeline,
    /// Per-frame pipelines for dynamic HDR (fp16 input only); `convert` picks one from the
    /// frame's [`FrameColor`] instead of `pipeline`.
    dynamic: Option<DynamicPipelines>,
    pipeline_layout: vk::PipelineLayout,
    desc_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    memp: vk::PhysicalDeviceMemoryProperties,
    /// Binary semaphore signalled by every submit; exported as a SYNC_FD and attached to
    /// the export dmabuf as a write fence. Only used on the implicit-sync (non-Nvidia)
    /// path.
    sync_sem: vk::Semaphore,
    sem_fd: Option<ash::khr::external_semaphore_fd::Device>,
    /// `sync_sem` still holds a signal whose payload could not be exported; it is recreated
    /// once the blocking wait has retired that submit.
    sync_sem_stuck: bool,
    /// AMD/Intel: hand downstream a dmabuf write-fence and don't stall. Nvidia: block.
    implicit_sync: bool,
    /// LINEAR direct path (compute writes the export image; no scratch/copy).
    direct: bool,
    /// Shared-device encode path: outputs are NV12 `GstVulkanImageMemory` from the encoder's
    /// own pool (single multiplanar `VIDEO_ENCODE_SRC` images); compute writes a storage
    /// scratch then copies into the pool image, which is left in `VIDEO_ENCODE_SRC` layout
    /// for the encoder to view zero-copy. No dmabuf export, no implicit-sync fence.
    encode_src: bool,
    /// Keeps the shared `GstVulkanDevice` alive for the converter's lifetime (the encode-src
    /// images are allocated on it).
    _shared_device: Option<gstreamer_vulkan::VulkanDevice>,
    /// Where our own device goes back to when we drop (see [`PooledDevice`]): its render
    /// minor and the device as opened. `None` on the shared-device path.
    pool_return: Option<(u32, OpenedDevice)>,
    outputs: Vec<Nv12Out>,
    next: usize, // next ring slot to write
    cur: usize,  // last slot written (the one to_gst_buffer returns)
    /// True once at least one frame has completed conversion, so a busy-drop has a previous
    /// output to safely re-emit rather than reusing a slot the encoder still references.
    have_output: bool,
    last_output_hdr: bool,
    /// Count of frames dropped because every ring slot was still referenced downstream at
    /// reuse time; drives rate-limited logging only.
    busy_drops: u64,
    /// When the pending colorimetry transition started waiting for a free slot; bounds the
    /// total wait to [`TRANSITION_SLOT_WAIT`] across frames.
    transition_wait: Option<std::time::Instant>,
    width: u32,
    height: u32,
    /// Target output format (NV12 8-bit or P010 10-bit); selects the compute shader, image
    /// format, and storage-view formats.
    fmt: PixFmt,
}

impl std::fmt::Debug for VulkanNv12 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanNv12({}x{})", self.width, self.height)
    }
}

/// Per-source conversion settings. Legacy environment variables supply defaults;
/// element properties override them before the pipeline starts.
#[derive(Clone, Copy, Debug)]
pub struct ConversionOptions {
    pub ring_depth: usize,
    pub linear_encode_src: bool,
    pub sdr_reference_white: f32,
    /// Peak luminance (cd/m²) of the HDR display presented to clients under dynamic HDR
    /// (the element's `hdr-peak-luminance`).
    pub hdr_peak_nits: u32,
    /// Dynamic HDR color management (the element's `dynamic-hdr` property):
    /// builds the fp16 render target, both SDR and PQ-passthrough converter pipelines,
    /// and gates the compositor's HDR-state signalling. Read once at converter creation
    /// and at compositor start; the element property is the source of truth.
    pub hdr_cm: bool,
    /// Under the element's `stable-color-transport`: the output state every frame is
    /// converted for, fixed at start from the static `hdr` setting, instead of following
    /// the compositor's HDR state. `None` = follow it (dynamic caps).
    pub stable_output_hdr: Option<bool>,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            ring_depth: RING,
            linear_encode_src: false,
            // Wolf sets WOLF_SDR_REFERENCE_WHITE from its config; the sdr-reference-white
            // property overrides it, like WOLF_HDR_PEAK_NITS below.
            sdr_reference_white: std::env::var("WOLF_SDR_REFERENCE_WHITE")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|n| (1.0..=10000.0).contains(n))
                .unwrap_or(203.0),
            hdr_peak_nits: std::env::var("WOLF_HDR_PEAK_NITS")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(1000)
                .min(65535),
            hdr_cm: false,
            stable_output_hdr: None,
        }
    }
}

fn dma_buf_identity(fd: RawFd) -> Result<(u64, u64), std::io::Error> {
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok((stat.st_dev, stat.st_ino))
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use std::os::fd::AsRawFd;

    #[test]
    fn recycled_fd_invalidates_import_identity() {
        let first = tempfile::tempfile().unwrap();
        let second = tempfile::tempfile().unwrap();
        let fd = first.as_raw_fd();
        let original = dma_buf_identity(fd).unwrap();
        assert_eq!(unsafe { libc::dup2(second.as_raw_fd(), fd) }, fd);
        assert_ne!(dma_buf_identity(fd).unwrap(), original);
    }
}

impl VulkanNv12 {
    /// Bring up Vulkan, build the compute pipeline, and allocate the NV12 export ring for
    /// `video_info`'s resolution, exporting with the negotiated modifier.
    pub fn new(
        render_node: DrmNode,
        video_info: VideoInfoDmaDrm,
        fmt: PixFmt,
        bt2020: bool,
        fp16_input: bool,
        va_share: &crate::utils::va_share::VaShare,
        options: ConversionOptions,
    ) -> Option<Self> {
        let width = video_info.width();
        let height = video_info.height();
        // Pick the Vulkan device whose DRM render/primary minor matches the render node,
        // so on multi-GPU hosts we import the compositor's dmabuf on the *same* GPU that
        // produced it (e.g. import an nvidia block-linear buffer on the nvidia device,
        // not on the Intel device that happens to be physical-device 0).
        let target_minor = libc::minor(render_node.dev_id());
        let modifier = match video_info.modifier() {
            DRM_FORMAT_MOD_INVALID => DRM_FORMAT_MOD_LINEAR,
            m => m,
        };
        match unsafe {
            Self::new_inner(
                width,
                height,
                modifier,
                target_minor,
                fmt,
                bt2020,
                fp16_input,
                va_share,
                options,
            )
        } {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::error!("VulkanNv12::new_inner failed: {e}");
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn new_inner(
        width: u32,
        height: u32,
        export_modifier: u64,
        target_minor: u32,
        fmt: PixFmt,
        bt2020: bool,
        fp16_input: bool,
        va_share: &crate::utils::va_share::VaShare,
        options: ConversionOptions,
    ) -> Result<Self, Err> {
        check_420_extent(width, height)?;
        let (entry, instance, opened) = take_device(target_minor)?;
        let queue = opened.device.get_device_queue(opened.qfi, 0);
        let memp = instance.get_physical_device_memory_properties(opened.pd);
        // From here on `this` owns everything created: if a later step fails, its Drop frees
        // whatever was created so far (destroying a null handle is a no-op) and returns the
        // device and the instance to the pool.
        let mut this = Self::skeleton(
            entry,
            instance,
            opened.device.clone(),
            queue,
            opened.qfi,
            if opened.queue_family_foreign {
                vk::QUEUE_FAMILY_FOREIGN_EXT
            } else {
                vk::QUEUE_FAMILY_EXTERNAL
            },
            memp,
            width,
            height,
            fmt,
        );
        let (qfi, implicit_sync) = (opened.qfi, opened.implicit_sync);
        this.pool_return = Some((target_minor, opened));
        this.build_compute(fmt, bt2020, fp16_input, &options, qfi)?;

        // ---- NV12 export ring ----
        // Prefer the LINEAR direct path (compute writes the export image; no scratch/copy)
        // when the negotiated modifier is LINEAR; if the driver won't give a STORAGE-able
        // LINEAR NV12 export image, fall back to the tiled scratch+copy path.
        let gst_allocator = DmaBufAllocator::new();
        let want_direct = export_modifier == DRM_FORMAT_MOD_LINEAR;
        this.direct = want_direct;
        if let Err(e) = this.build_export_ring(&gst_allocator, export_modifier, va_share, &options)
        {
            if !want_direct {
                return Err(e);
            }
            tracing::warn!("VulkanNv12: LINEAR direct path unavailable ({e}); using scratch+copy");
            // Free the slots built so far and their descriptor sets (the pool holds exactly
            // one set per slot) before building the tiled ring.
            this.destroy_outputs();
            this.device
                .reset_descriptor_pool(this.desc_pool, vk::DescriptorPoolResetFlags::empty())?;
            this.direct = false;
            this.build_export_ring(&gst_allocator, export_modifier, va_share, &options)?;
        }

        // Binary semaphore exported per submit as the dmabuf write-fence (implicit path).
        if implicit_sync {
            let mut exp = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            this.sync_sem = this.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut exp),
                None,
            )?;
            this.sem_fd = Some(ash::khr::external_semaphore_fd::Device::new(
                &this.instance,
                &this.device,
            ));
            this.implicit_sync = true;
        }
        Ok(this)
    }

    /// A converter owning `device`/`instance` with no other objects yet: every other handle
    /// is null, so dropping it part-way through construction frees exactly what was built.
    #[allow(clippy::too_many_arguments)]
    fn skeleton(
        entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        queue: vk::Queue,
        qfi: u32,
        foreign_qf: u32,
        memp: vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
        fmt: PixFmt,
    ) -> Self {
        VulkanNv12 {
            _entry: entry,
            instance,
            device,
            queue,
            gst_queue: None,
            qfi,
            foreign_qf,
            cmd_pool: vk::CommandPool::null(),
            pipeline: vk::Pipeline::null(),
            dynamic: None,
            pipeline_layout: vk::PipelineLayout::null(),
            desc_layout: vk::DescriptorSetLayout::null(),
            desc_pool: vk::DescriptorPool::null(),
            sampler: vk::Sampler::null(),
            memp,
            sync_sem: vk::Semaphore::null(),
            sem_fd: None,
            sync_sem_stuck: false,
            implicit_sync: false,
            direct: false,
            encode_src: false,
            _shared_device: None,
            pool_return: None,
            outputs: Vec::new(),
            next: 0,
            have_output: false,
            last_output_hdr: false,
            busy_drops: 0,
            transition_wait: None,
            cur: 0,
            width,
            height,
            fmt,
        }
    }

    /// The compute pipeline(s), descriptor layout/pool, sampler and command pool, shared by
    /// the dmabuf and encode-src paths (one descriptor set and command buffer per ring slot).
    unsafe fn build_compute(
        &mut self,
        fmt: PixFmt,
        bt2020: bool,
        fp16_input: bool,
        options: &ConversionOptions,
        cmd_queue_family: u32,
    ) -> Result<(), Err> {
        let binds = [
            dsl_bind(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            dsl_bind(1, vk::DescriptorType::STORAGE_IMAGE),
            dsl_bind(2, vk::DescriptorType::STORAGE_IMAGE),
        ];
        self.desc_layout = self.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
            None,
        )?;
        let dsls = [self.desc_layout];
        let pcr = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        self.pipeline_layout = self.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&dsls)
                .push_constant_ranges(&pcr),
            None,
        )?;
        let (pipeline, dynamic) = build_pipelines(
            &self.device,
            self.pipeline_layout,
            fmt,
            bt2020,
            fp16_input,
            options,
        )?;
        self.pipeline = pipeline;
        self.dynamic = dynamic;

        let psizes = [
            pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, RING as u32),
            pool_size(vk::DescriptorType::STORAGE_IMAGE, 2 * RING as u32),
        ];
        self.desc_pool = self.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(RING as u32)
                .pool_sizes(&psizes),
            None,
        )?;
        self.sampler = self
            .device
            .create_sampler(&vk::SamplerCreateInfo::default(), None)?;
        self.cmd_pool = self.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(cmd_queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        Ok(())
    }

    /// Fill the dmabuf export ring (`self.direct` selects the LINEAR direct or tiled path).
    /// On failure the slots built so far stay in `self.outputs` for the caller to free.
    unsafe fn build_export_ring(
        &mut self,
        gst_allocator: &DmaBufAllocator,
        export_modifier: u64,
        va_share: &crate::utils::va_share::VaShare,
        options: &ConversionOptions,
    ) -> Result<(), Err> {
        tracing::debug!(
            "VulkanNv12: creating {} export images {}x{} modifier={export_modifier:#x} direct={}",
            ring_depth(options),
            self.width,
            self.height,
            self.direct
        );
        for _ in 0..ring_depth(options) {
            let out = create_output(
                &self.instance,
                &self.device,
                &self.memp,
                gst_allocator,
                self.desc_pool,
                self.desc_layout,
                self.cmd_pool,
                self.width,
                self.height,
                export_modifier,
                self.direct,
                self.fmt,
                va_share,
            )?;
            self.outputs.push(out);
        }
        Ok(())
    }

    /// Shared-device encode path: wrap the downstream encoder's `GstVulkanDevice` and build
    /// an output ring of NV12 `GstVulkanImageMemory` buffers from its encode-src pool. The
    /// compositor's RGBA dmabuf is imported + converted (compute -> storage scratch) and
    /// `vkCmdCopyImage`'d into the pool's encode-src image, left in `VIDEO_ENCODE_SRC`
    /// layout for `vulkanh264enc` to view zero-copy. Same device as the encoder, so ordering
    /// is a plain fence wait (the encoder does its own input acquire).
    #[allow(clippy::too_many_arguments)]
    pub fn new_on_shared(
        device_gst: gstreamer_vulkan::VulkanDevice,
        raw: crate::utils::vulkan_share::RawVk,
        nv12_caps: &gst::Caps,
        targets: &[crate::utils::video_info::EncodeTarget],
        width: u32,
        height: u32,
        fmt: PixFmt,
        bt2020: bool,
        fp16_input: bool,
        options: ConversionOptions,
    ) -> Option<Self> {
        match unsafe {
            Self::new_on_shared_inner(
                device_gst, raw, nv12_caps, targets, width, height, fmt, bt2020, fp16_input,
                options,
            )
        } {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::error!("VulkanNv12::new_on_shared failed: {e}");
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn new_on_shared_inner(
        device_gst: gstreamer_vulkan::VulkanDevice,
        raw: crate::utils::vulkan_share::RawVk,
        _nv12_caps: &gst::Caps,
        targets: &[crate::utils::video_info::EncodeTarget],
        width: u32,
        height: u32,
        fmt: PixFmt,
        bt2020: bool,
        fp16_input: bool,
        options: ConversionOptions,
    ) -> Result<Self, Err> {
        check_420_extent(width, height)?;
        // An absorbed device may lack the import extensions; ash would then call stubs.
        let missing = crate::utils::vulkan_share::missing_extensions(&device_gst);
        if !missing.is_empty() {
            return Err(format!("shared GstVulkanDevice lacks {missing:?}").into());
        }
        // Drive ash on the encoder's existing instance/device (do NOT create our own), and
        // submit to gst's own graphics queue under its submit lock.
        let entry = ash::Entry::load()?;
        let instance = ash::Instance::load(entry.static_fn(), raw.instance);
        let device = ash::Device::load(instance.fp_v1_0(), raw.device);
        let queue = raw.gfx_vk_queue();
        let mut queue_families = vec![raw.gfx_queue_family];
        if let Some(enc) = raw
            .encode_queue_family
            .filter(|f| *f != raw.gfx_queue_family)
        {
            queue_families.push(enc);
        }
        let memp = instance.get_physical_device_memory_properties(raw.physical);
        // As on the dmabuf path, `this` frees whatever was built if a later step fails; with
        // `encode_src` set it never destroys the encoder's device or instance.
        let mut this = Self::skeleton(
            entry,
            instance,
            device,
            queue,
            raw.gfx_queue_family,
            if raw.queue_family_foreign {
                vk::QUEUE_FAMILY_FOREIGN_EXT
            } else {
                vk::QUEUE_FAMILY_EXTERNAL
            },
            memp,
            width,
            height,
            fmt,
        );
        this.encode_src = true;
        this.gst_queue = Some(raw.gfx_queue);
        this._shared_device = Some(device_gst.clone());
        this.build_compute(fmt, bt2020, fp16_input, &options, raw.gfx_queue_family)?;

        // ---- encode-src output ring: NV12 VIDEO_ENCODE_SRC GstVulkanImageMemory images
        // allocated directly on the shared device (bypasses the pool's generic-format check).
        for _ in 0..ring_depth(&options) {
            let out = create_encode_output(
                &this.device,
                &this.memp,
                &device_gst,
                targets,
                &queue_families,
                this.desc_pool,
                this.desc_layout,
                this.cmd_pool,
                width,
                height,
                fmt,
                options.linear_encode_src,
            )?;
            this.outputs.push(out);
        }
        Ok(this)
    }

    /// Whether ring slot `i` may be overwritten: nothing downstream references it any more.
    /// The encode-src child headers keep the slot buffer itself referenced (through
    /// `GstParentBufferMeta`); a dmabuf slot is handed out as a ref of the slot buffer, and a
    /// copy-on-write header copy of it (BaseSrc's DISCONT `make_writable`, or any downstream
    /// element) shares its memory instead, so the memory must be unshared too.
    fn slot_free(&self, i: usize) -> bool {
        let buffer = &self.outputs[i].buffer;
        buffer.is_writable() && (self.encode_src || buffer.is_all_memory_writable())
    }

    /// Import the compositor's `rgba` dmabuf and convert it RGBA->NV12 into the next
    /// export ring slot. Under dynamic HDR (fp16 input) `color` selects the per-frame
    /// shader, so the pixels match the colorimetry the element tags the frame with;
    /// otherwise it is ignored and the shader fixed at creation runs.
    pub fn convert(&mut self, rgba: &Dmabuf, color: FrameColor) -> Result<(), Err> {
        unsafe { self.convert_inner(rgba, color) }
    }

    unsafe fn convert_inner(&mut self, rgba: &Dmabuf, color: FrameColor) -> Result<(), Err> {
        // fd numbers are recycled. Identify the backing object, not its descriptor.
        let fd = rgba.handles().next().ok_or("no fd")?.as_fd().as_raw_fd();
        let key = dma_buf_identity(fd)?;

        // Reuse safety: a produced buffer can still be referenced downstream -- held by an
        // encoder (B-frames, lookahead), a queue, or several consumers at once (interpipe
        // fan-out). Our own fence only proves *our* last write to a slot finished; it says
        // nothing about the consumers' reads, which carry no sync back to us (the dmabuf
        // write-fence orders their read after our write, not our next write after their
        // read). Overwriting a slot while it is still read tears frames, corrupts encoder
        // input, or wedges a Vulkan encoder, so only a slot every consumer has released may
        // be reused. Take the next free slot in ring order; if none is free, drop this frame
        // and re-emit the last completed output (`to_gst_buffer` returns `cur`).
        //
        // A colorimetry transition waits a little for a fresh slot, since repeating SDR pixels
        // under HDR caps (or vice versa) is wrong; past TRANSITION_SLOT_WAIT it degrades to the
        // same re-emit rather than stalling the compositor thread or ending the stream.
        let n = self.outputs.len();
        let changing_color = self
            .dynamic
            .as_ref()
            .is_some_and(|d| self.last_output_hdr != d.output_hdr(color));
        let mut free = (0..n)
            .map(|i| (self.next + i) % n)
            .find(|&i| self.slot_free(i));
        if !changing_color {
            self.transition_wait = None;
        }
        if free.is_none() && changing_color {
            let since = *self
                .transition_wait
                .get_or_insert_with(std::time::Instant::now);
            while free.is_none() && since.elapsed() < TRANSITION_SLOT_WAIT {
                std::thread::sleep(std::time::Duration::from_millis(1));
                free = (0..n)
                    .map(|i| (self.next + i) % n)
                    .find(|&i| self.slot_free(i));
            }
        }
        let idx = match free {
            Some(i) => i,
            None if !self.have_output => {
                return Err("VulkanNv12: output ring busy before first completed frame".into());
            }
            None => {
                self.busy_drops = self.busy_drops.saturating_add(1);
                if self.busy_drops == 1 || self.busy_drops % 60 == 0 {
                    tracing::warn!(
                        busy_drops = self.busy_drops,
                        changing_color,
                        "VulkanNv12: every output slot is still referenced downstream; dropping new frame, re-emitting previous output"
                    );
                }
                return Ok(());
            }
        };
        self.transition_wait = None;
        self.next = (idx + 1) % self.outputs.len();
        // Reclaim the slot's previous (RING-frames-old) work. The fence was signalled long
        // ago in steady state, so this wait is a no-op that lets us recycle the cmd buffer.
        if self.outputs[idx].in_flight {
            self.device
                .wait_for_fences(&[self.outputs[idx].fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.outputs[idx].fence])?;
            self.outputs[idx].in_flight = false;
        }
        // No timestamp reset on the parent: to_gst_buffer hands downstream a fresh child
        // header whose PTS is already NONE, and nothing stamps the parent header.

        // Ensure this slot's cached RGBA import matches the current dmabuf. Built once per
        // slot (the slot is idle here -- we waited its fence above -- so freeing/recreating
        // is safe). Reused every frame since the dmabuf is stable; the import is a distinct
        // VkImage per slot (not shared), so its per-frame layout transition stays
        // serialised by the slot fence and never races another in-flight frame.
        let stale = !matches!(&self.outputs[idx].import, Some((identity, _)) if *identity == key);
        if stale {
            self.free_import(idx);
            let import = self.create_import(rgba)?;
            self.outputs[idx].import = Some((key, import));
        }
        let (import_img, _import_mem, view) = self.outputs[idx].import.as_ref().unwrap().1;

        let slot = &self.outputs[idx];
        let out_img = slot.image;
        let desc_set = slot.desc_set;
        let cmd = slot.cmd;
        let fence = slot.fence;
        let direct = self.direct;
        // Per-frame shader under dynamic HDR, else the one fixed at creation.
        let pipeline = match &self.dynamic {
            Some(dynamic) => dynamic.pick(color).unwrap_or(self.pipeline),
            None => self.pipeline,
        };
        // Where the compute shader writes: scratch (tiled) or the export image (direct).
        let compute_target = match slot.scratch {
            Some((s, _)) => s,
            None => out_img,
        };

        // Bind this slot's source (binding 0) to the cached import view. Bindings 1,2 (the
        // compute output planes) were written once at slot creation and are constant. Safe
        // to update now: this slot's previous submit is complete (waited above).
        let src_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        self.device.update_descriptor_sets(
            &[write_img(
                desc_set,
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                &src_info,
            )],
            &[],
        );

        self.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        self.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

        // The RGBA import is written by GLES outside Vulkan: acquire it from the foreign
        // family in GENERAL, the layout it is released back in after every frame. An
        // UNDEFINED old layout would let the driver discard the contents (and DCC/metadata
        // of a compressed modifier) before we sample them.
        let (qfi, foreign) = (self.qfi, self.foreign_qf);
        let to_read = img_barrier(
            import_img,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
        )
        .src_queue_family_index(foreign)
        .dst_queue_family_index(qfi);
        // The compute target is fully overwritten, so its old contents may be discarded. On
        // the direct path it is the exported image: acquire it from the consumer.
        let (t_src, t_dst) = if direct {
            (foreign, qfi)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };
        let t_y = img_barrier(
            compute_target,
            vk::ImageAspectFlags::PLANE_0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
        )
        .src_queue_family_index(t_src)
        .dst_queue_family_index(t_dst);
        let t_uv = img_barrier(
            compute_target,
            vk::ImageAspectFlags::PLANE_1,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
        )
        .src_queue_family_index(t_src)
        .dst_queue_family_index(t_dst);
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_read, t_y, t_uv],
        );
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &[desc_set],
            &[],
        );
        let pc: [i32; 2] = [self.width as i32, self.height as i32];
        self.device.cmd_push_constants(
            cmd,
            self.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            std::slice::from_raw_parts(pc.as_ptr() as *const u8, 8),
        );
        self.device
            .cmd_dispatch(cmd, (self.width / 2 + 7) / 8, (self.height / 2 + 7) / 8, 1);

        // Hand the RGBA import back to GLES once the compute has read it.
        let release_import = img_barrier(
            import_img,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::empty(),
        )
        .src_queue_family_index(qfi)
        .dst_queue_family_index(foreign);
        if direct {
            // LINEAR direct: the compute output *is* the export image. Flush the shader
            // writes and release it to the dmabuf consumer (ordering to the consumer is the
            // implicit dma-buf write-fence / the CPU wait below).
            let f_y = img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_0,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::empty(),
            )
            .src_queue_family_index(qfi)
            .dst_queue_family_index(foreign);
            let f_uv = img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::empty(),
            )
            .src_queue_family_index(qfi)
            .dst_queue_family_index(foreign);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[f_y, f_uv, release_import],
            );
        } else {
            // Tiled/DCC: copy NV12 scratch -> NV12 export slot (radv fills DCC metadata).
            // A dmabuf export slot is acquired from its consumer; the encode-src image is on
            // our device, shared CONCURRENTly with the encode queue, so it needs no transfer.
            let (o_src, o_dst) = if self.encode_src {
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
            } else {
                (foreign, qfi)
            };
            let s_y_src = img_barrier(
                compute_target,
                vk::ImageAspectFlags::PLANE_0,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            let s_uv_src = img_barrier(
                compute_target,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            let o_y = img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_0,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            )
            .src_queue_family_index(o_src)
            .dst_queue_family_index(o_dst);
            let o_uv = img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            )
            .src_queue_family_index(o_src)
            .dst_queue_family_index(o_dst);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[s_y_src, s_uv_src, o_y, o_uv, release_import],
            );
            let regions = [
                vk::ImageCopy::default()
                    .src_subresource(plane_layers(vk::ImageAspectFlags::PLANE_0))
                    .dst_subresource(plane_layers(vk::ImageAspectFlags::PLANE_0))
                    .extent(vk::Extent3D {
                        width: self.width,
                        height: self.height,
                        depth: 1,
                    }),
                vk::ImageCopy::default()
                    .src_subresource(plane_layers(vk::ImageAspectFlags::PLANE_1))
                    .dst_subresource(plane_layers(vk::ImageAspectFlags::PLANE_1))
                    .extent(vk::Extent3D {
                        width: self.width / 2,
                        height: self.height / 2,
                        depth: 1,
                    }),
            ];
            self.device.cmd_copy_image(
                cmd,
                compute_target,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                out_img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );

            // Hand the encoder its input already in VIDEO_ENCODE_SRC_KHR (ordered after this
            // submit by the slot fence wait below, same device), or release the dmabuf export
            // slot to its consumer in GENERAL.
            let (final_layout, r_src, r_dst) = if self.encode_src {
                (
                    vk::ImageLayout::from_raw(VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR),
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::QUEUE_FAMILY_IGNORED,
                )
            } else {
                (vk::ImageLayout::GENERAL, qfi, foreign)
            };
            let e_y = img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_0,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                final_layout,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
            )
            .src_queue_family_index(r_src)
            .dst_queue_family_index(r_dst);
            let e_uv = img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                final_layout,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
            )
            .src_queue_family_index(r_src)
            .dst_queue_family_index(r_dst);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[e_y, e_uv],
            );
        }

        self.device.end_command_buffer(cmd)?;
        let cbs = [cmd];
        let mut submit = vk::SubmitInfo::default().command_buffers(&cbs);
        let sems = [self.sync_sem];
        if self.implicit_sync {
            submit = submit.signal_semaphores(&sems);
        }
        self.submit(&[submit], fence)?;
        // In flight from here until its fence is waited on, so an error below still leaves
        // teardown (`destroy_outputs`) waiting for it before freeing the command buffer.
        self.outputs[idx].in_flight = true;

        let fenced = self.implicit_sync && self.attach_fences(idx, fd);
        if fenced {
            // The export dmabuf's BO carries the submit's completion as a *write* fence, so
            // the VA encoder's import-read waits for it via the kernel's implicit sync, and
            // the RGBA dmabuf carries it as a *read* fence, so the compositor's next render
            // into it waits for this compute to finish sampling it -- no CPU stall on the
            // producer's critical path (lever 1). The cached import stays alive (it's
            // reused next time this slot comes round).
        } else {
            // Nvidia/CUDA ignores implicit dma-buf fences (and the implicit path falls back
            // here when the fences could not be attached): block until the write is done
            // before the buffer is handed downstream and the RGBA buffer is rendered again.
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.reset_fences(&[fence])?;
            self.outputs[idx].in_flight = false;
            if self.implicit_sync && self.sync_sem_stuck {
                self.recreate_sync_sem();
            }
        }

        // Debug: one-shot readback of the converter's NV12 output to a file. `compute_target`
        // (the LINEAR scratch on the encode-src path) holds exactly what the compute wrote,
        // before the tiled encode-src copy -- so a clean dump localises the green-bar/jump to
        // the tiled copy or the encoder. Off unless WOLF_VULKAN_DUMP is set; once, a
        // few frames in (let the scene settle). Failures are logged, never fatal. The
        // direct export is reacquired for readback and released again before hand-off.
        if let Some(path) = dump_path() {
            // Fire either at the fixed frame target (one-shot) OR on demand whenever a
            // `<path>.now` trigger file exists -- consumed each time so every touch captures
            // exactly one fresh frame. The trigger lets a specific live gameplay frame be
            // snapped (the fixed-frame path lands on deterministic loading screens).
            let trigger = format!("{path}.now");
            let by_trigger = std::path::Path::new(&trigger).exists();
            let by_frame = DUMP_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                == dump_frame_target();
            if by_trigger || by_frame {
                if by_trigger {
                    let _ = std::fs::remove_file(&trigger);
                }
                // Our submit only: device_wait_idle would need every queue of a shared
                // device (the encoders') externally synchronised.
                if self.outputs[idx].in_flight {
                    self.device.wait_for_fences(&[fence], true, u64::MAX).ok();
                }
                let pix = match self.fmt {
                    PixFmt::Nv12 => "nv12",
                    PixFmt::P010 => "p010le",
                };
                match self.dump_planar(compute_target, self.width, self.height, path) {
                    Ok(()) => tracing::info!(
                        "VulkanNv12: dumped converter {:?} {}x{} (raw planar) to {path} -- view: \
                         ffmpeg -f rawvideo -pix_fmt {pix} -s {}x{} -i {path} -frames 1 out.png",
                        self.fmt,
                        self.width,
                        self.height,
                        self.width,
                        self.height
                    ),
                    Err(e) => tracing::warn!("VulkanNv12: {:?} dump failed: {e}", self.fmt),
                }
            }
        }

        self.cur = idx;
        self.have_output = true;
        self.last_output_hdr = self.dynamic.as_ref().is_some_and(|d| d.output_hdr(color));
        Ok(())
    }

    /// One-shot debug readback of a multiplanar NV12/P010 image (must be in
    /// TRANSFER_SRC_OPTIMAL for scratch, GENERAL/foreign for direct exports) into
    /// a host buffer, then write tight raw planar (`w*h*bpp` Y plane + `w*h/2*bpp` interleaved
    /// UV) to `path`. `bpp` is 1 for NV12, 2 for P010 (16-bit samples, value in the MSBs).
    unsafe fn dump_planar(&self, img: vk::Image, w: u32, h: u32, path: &str) -> Result<(), Err> {
        std::fs::write(path, self.read_planar(img, w, h)?)?;
        Ok(())
    }

    /// The last converted frame as tight planar samples (as [`Self::dump_planar`] writes
    /// them), read back on the GPU.
    #[cfg(test)]
    pub(crate) fn read_output(&self) -> Result<Vec<u8>, Err> {
        let slot = &self.outputs[self.cur];
        let img = match slot.scratch {
            Some((s, _)) => s,
            None => slot.image,
        };
        unsafe { self.read_planar(img, self.width, self.height) }
    }

    /// Copy `img` (layouts as for [`Self::dump_planar`]) into tight planar bytes.
    unsafe fn read_planar(&self, img: vk::Image, w: u32, h: u32) -> Result<Vec<u8>, Err> {
        let bpp = self.fmt.y_bytes();
        let y_size = w as u64 * h as u64 * bpp;
        let total = y_size + (w as u64 * h as u64 / 2 * bpp);
        let buf = self.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(total)
                .usage(vk::BufferUsageFlags::TRANSFER_DST),
            None,
        )?;
        let mr = self.device.get_buffer_memory_requirements(buf);
        let mt = mem_type(
            &self.memp,
            mr.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mem = self.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(mr.size)
                .memory_type_index(mt),
            None,
        )?;
        self.device.bind_buffer_memory(buf, mem, 0)?;

        let cmd = self.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(self.cmd_pool)
                .command_buffer_count(1),
        )?[0];
        self.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
        if self.direct {
            let barriers =
                [vk::ImageAspectFlags::PLANE_0, vk::ImageAspectFlags::PLANE_1].map(|aspect| {
                    img_barrier(
                        img,
                        aspect,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_READ,
                    )
                    .src_queue_family_index(self.foreign_qf)
                    .dst_queue_family_index(self.qfi)
                });
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
        let regions = [
            vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                }),
            vk::BufferImageCopy::default()
                .buffer_offset(y_size)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: w / 2,
                    height: h / 2,
                    depth: 1,
                }),
        ];
        self.device.cmd_copy_image_to_buffer(
            cmd,
            img,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buf,
            &regions,
        );
        if self.direct {
            let barriers =
                [vk::ImageAspectFlags::PLANE_0, vk::ImageAspectFlags::PLANE_1].map(|aspect| {
                    img_barrier(
                        img,
                        aspect,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::AccessFlags::empty(),
                    )
                    .src_queue_family_index(self.qfi)
                    .dst_queue_family_index(self.foreign_qf)
                });
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
        self.device.end_command_buffer(cmd)?;
        let fence = self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cbs = [cmd];
        self.submit(&[vk::SubmitInfo::default().command_buffers(&cbs)], fence)?;
        self.device.wait_for_fences(&[fence], true, u64::MAX)?;

        let ptr = self
            .device
            .map_memory(mem, 0, total, vk::MemoryMapFlags::empty())? as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, total as usize).to_vec();
        self.device.unmap_memory(mem);

        self.device.destroy_fence(fence, None);
        self.device.free_command_buffers(self.cmd_pool, &[cmd]);
        self.device.destroy_buffer(buf, None);
        self.device.free_memory(mem, None);
        Ok(bytes)
    }

    /// Submit to our queue, holding gst's submit lock on the shared-device path.
    unsafe fn submit(
        &self,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> ash::prelude::VkResult<()> {
        let _lock = self
            .gst_queue
            .as_ref()
            .map(crate::utils::vulkan_share::QueueSubmitGuard::new);
        self.device.queue_submit(self.queue, submits, fence)
    }

    /// Build a sampled VkImage over the source RGBA dmabuf (explicit per-plane layout, so
    /// multi-plane DCC modifiers import too). The Vulkan device matches the render node
    /// (see `new`), so the import runs on the GPU that produced the dmabuf -- this is what
    /// lets nvidia import its own block-linear RGBA. Cached per ring slot by the caller.
    unsafe fn create_import(&self, rgba: &Dmabuf) -> Result<Import, Err> {
        let pd_format = rgba_vk_format(rgba.format().code).ok_or("unsupported RGBA fourcc")?;
        let modifier: u64 = rgba.format().modifier.into();
        // One layout per memory plane (main + any DCC-metadata plane). Planes share the
        // single underlying dmabuf allocation, so each is an offset/pitch into the same
        // imported memory (non-disjoint); offsets beyond what `offsets()` reports are 0.
        let strides: Vec<u64> = rgba.strides().map(|s| s as u64).collect();
        let offsets: Vec<u64> = rgba.offsets().map(|o| o as u64).collect();
        if strides.is_empty() {
            return Err("no stride".into());
        }
        // Diagnostic for cross-API import layout bugs (e.g. RX 9070 DCC "jumping"): the
        // actual modifier + per-plane strides/offsets the import builds the VkImage from.
        tracing::debug!(
            "VulkanNv12 import: format={:?} modifier={modifier:#x} extent={}x{} strides={strides:?} offsets={offsets:?} planes={}",
            rgba.format().code,
            self.width,
            self.height,
            strides.len(),
        );
        let src_fd = rgba
            .handles()
            .next()
            .ok_or("no fd")?
            .as_fd()
            .try_clone_to_owned()?;
        let plane_layout: Vec<vk::SubresourceLayout> = strides
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                vk::SubresourceLayout::default()
                    .offset(offsets.get(i).copied().unwrap_or(0))
                    .row_pitch(s)
            })
            .collect();
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(modifier)
            .plane_layouts(&plane_layout);
        let mut extmem = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let img = self.device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(pd_format)
                .extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut extmem)
                .push_next(&mut explicit),
            None,
        )?;
        // Everything below frees what it created so far on failure; `src_fd` is closed by its
        // OwnedFd unless a successful import took ownership of it.
        let mem = match self.import_memory(img, src_fd) {
            Ok(mem) => mem,
            Err(e) => {
                self.device.destroy_image(img, None);
                return Err(e);
            }
        };
        let bound = self
            .device
            .bind_image_memory(img, mem, 0)
            .map_err(Err::from)
            .and_then(|()| plane_view(&self.device, img, pd_format, vk::ImageAspectFlags::COLOR));
        match bound {
            Ok(view) => Ok((img, mem, view)),
            Err(e) => {
                self.device.destroy_image(img, None);
                self.device.free_memory(mem, None);
                Err(e)
            }
        }
    }

    /// Import `fd` as memory for `img`. Vulkan owns the fd only once the allocation
    /// succeeds; on failure it is closed when `fd` drops.
    unsafe fn import_memory(
        &self,
        img: vk::Image,
        fd: std::os::fd::OwnedFd,
    ) -> Result<vk::DeviceMemory, Err> {
        let mr = self.device.get_image_memory_requirements(img);
        let ext_fd = ash::khr::external_memory_fd::Device::new(&self.instance, &self.device);
        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        ext_fd.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            fd.as_raw_fd(),
            &mut fd_props,
        )?;
        let allowed = mr.memory_type_bits & fd_props.memory_type_bits;
        let mt = (0..self.memp.memory_type_count)
            .find(|&i| allowed & (1 << i) != 0)
            .ok_or("no import-compatible memory type")?;
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd.as_raw_fd());
        let mem = self.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(mr.size)
                .memory_type_index(mt)
                .push_next(&mut import_info),
            None,
        )?;
        // A successful import transfers ownership of the fd to the driver.
        let _ = fd.into_raw_fd();
        Ok(mem)
    }

    /// Export the submit's signal semaphore as a SYNC_FD and import it as a write fence onto
    /// slot `idx`'s export dmabuf (the consumer's read waits for our write) and as a read
    /// fence onto the source RGBA dmabuf `rgba_fd` (the compositor's next write waits for our
    /// read). Returns `false` unless both landed; the caller then waits the submit's fence
    /// on the CPU instead. A semaphore whose payload could not be exported still holds a
    /// pending signal and must not be signalled again: it is flagged for recreation once
    /// that wait has retired it.
    unsafe fn attach_fences(&mut self, idx: usize, rgba_fd: RawFd) -> bool {
        use std::os::fd::{FromRawFd, OwnedFd};
        let Some(loader) = self.sem_fd.as_ref() else {
            return false;
        };
        let sync_fd = match loader.get_semaphore_fd(
            &vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(self.sync_sem)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
        ) {
            Ok(fd) if fd >= 0 => OwnedFd::from_raw_fd(fd),
            Ok(_) => return true, // -1 means already signalled; nothing to wait on
            Err(e) => {
                tracing::warn!("VulkanNv12: get_semaphore_fd failed: {e}; waiting on the CPU");
                self.sync_sem_stuck = true;
                return false;
            }
        };
        let import = |target: RawFd, flags: u32| {
            let arg = DmaBufImportSyncFile {
                flags,
                fd: sync_fd.as_raw_fd(),
            };
            if libc::ioctl(target, DMA_BUF_IOCTL_IMPORT_SYNC_FILE, &arg as *const _) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        };
        match import(self.outputs[idx].export_fd, DMA_BUF_SYNC_WRITE)
            .and_then(|()| import(rgba_fd, DMA_BUF_SYNC_READ))
        {
            Ok(()) => true,
            // The kernel predates the ioctl (Linux < 6.0): it will fail every frame, so
            // switch to the blocking path for good instead of failing and warning per frame.
            Err(e) if matches!(e.raw_os_error(), Some(libc::ENOTTY | libc::EINVAL)) => {
                tracing::warn!(
                    "VulkanNv12: IMPORT_SYNC_FILE unsupported ({e}); waiting on the CPU from now on"
                );
                self.implicit_sync = false;
                false
            }
            Err(e) => {
                tracing::warn!("VulkanNv12: IMPORT_SYNC_FILE failed ({e}); waiting on the CPU");
                false
            }
        }
    }

    /// Replace the per-submit signal semaphore after a failed export left it signalled. The
    /// caller has waited the submit that signalled it. Without a new one the implicit path is
    /// abandoned for the blocking fence wait.
    unsafe fn recreate_sync_sem(&mut self) {
        self.device.destroy_semaphore(self.sync_sem, None);
        self.sync_sem_stuck = false;
        let mut exp = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        match self.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut exp),
            None,
        ) {
            Ok(sem) => self.sync_sem = sem,
            Err(e) => {
                tracing::warn!(
                    "VulkanNv12: cannot recreate the sync semaphore ({e}); blocking per frame"
                );
                self.sync_sem = vk::Semaphore::null();
                self.implicit_sync = false;
            }
        }
    }

    unsafe fn free_import(&mut self, idx: usize) {
        if let Some((_, (img, mem, view))) = self.outputs[idx].import.take() {
            self.device.destroy_image_view(view, None);
            self.device.destroy_image(img, None);
            self.device.free_memory(mem, None);
        }
    }

    /// The just-converted NV12 export slot as a gst buffer.
    ///
    /// **VA / RGBx / DMABuf (`encode_src == false`):** the slot's cached buffer, ref-counted,
    /// so the encoder reuses one stable surface per slot. Unchanged.
    ///
    /// **Vulkan encode path (`encode_src == true`):** a fresh CHILD header over the slot,
    /// carrying `GstParentBufferMeta`, with PTS/DTS/duration cleared so BaseSrc's
    /// `do_timestamp` stamps it with the current running-time each frame. The parent's own
    /// timestamps are never read.
    pub fn to_gst_buffer(&self) -> Result<GstBuffer, Err> {
        let parent = &self.outputs[self.cur].buffer;
        if !self.encode_src {
            return Ok(parent.clone());
        }
        // Encode-src: hand downstream a FRESH child header over the slot, carrying
        // GstParentBufferMeta that references the cached slot buffer.
        //
        // A bare clone is defeated by copy-on-write. BaseSrc calls
        // gst_buffer_make_writable() on the DISCONT frame (gstbasesrc.c, "marking pending
        // DISCONT"), and any downstream element that makes the header writable does the
        // same. At refcount 2 that copies the header, so the cached slot buffer drops back
        // to refcount 1 while the copy -- and the encoder reading through it -- still points
        // at the same VkImage. The reuse gate in convert_inner then sees the slot as free
        // and lets the producer overwrite an image an encode is still reading.
        //
        // ParentBufferMeta survives that: its COPY transform re-adds the meta and re-refs
        // the parent, so every later header copy keeps the slot non-writable until the last
        // child is dropped. Resetting the timestamps on the fresh child also gives BaseSrc's
        // do_timestamp a PTS of NONE to stamp, without touching the shared slot.
        let mut child = parent.copy();
        let child_ref = child
            .get_mut()
            .ok_or("VulkanNv12: fresh output child unexpectedly shared")?;
        child_ref.set_pts(gst::ClockTime::NONE);
        child_ref.set_dts(gst::ClockTime::NONE);
        child_ref.set_duration(gst::ClockTime::NONE);
        gst::ParentBufferMeta::add(child_ref, parent);
        Ok(child)
    }
}

impl VulkanNv12 {
    /// Free every ring slot (after its in-flight work) and its cached import. The slots'
    /// descriptor sets go back with the pool (reset or destroyed by the caller).
    unsafe fn destroy_outputs(&mut self) {
        // Wait for our own submits only: on the shared device, device_wait_idle would race
        // the encoders' unsynchronised use of their queues.
        let pending: Vec<vk::Fence> = self
            .outputs
            .iter()
            .filter(|o| o.in_flight)
            .map(|o| o.fence)
            .collect();
        if !pending.is_empty() {
            let _ = self.device.wait_for_fences(&pending, true, u64::MAX);
        }
        for idx in 0..self.outputs.len() {
            self.free_import(idx);
        }
        // On the encode-src path the output image + memory belong to the gst pool (freed
        // when the buffer/pool drop) and there is no export fd to close.
        let owns_image = !self.encode_src;
        for o in std::mem::take(&mut self.outputs) {
            destroy_output(&self.device, self.cmd_pool, o, owns_image);
        }
    }
}

impl Drop for VulkanNv12 {
    fn drop(&mut self) {
        unsafe {
            self.destroy_outputs();
            if self.sync_sem != vk::Semaphore::null() {
                self.device.destroy_semaphore(self.sync_sem, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            if let Some(dynamic) = self.dynamic.take() {
                dynamic.destroy(&self.device);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            // On the shared/encode-src path the VkDevice + VkInstance are owned by the
            // downstream encoder's GstVulkanDevice (held alive via `_shared_device`); the
            // encoder still frees its own DPB image views at teardown, so destroying them
            // here would yank the device out from under it. Only tear down what we created.
            if !self.encode_src {
                // Everything we created on the device is gone; once our submits are done it
                // is as good as new. A device that can't even go idle is not reused.
                match self.pool_return.take() {
                    Some((minor, opened)) if self.device.device_wait_idle().is_ok() => {
                        DEVICE_POOL
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push(PooledDevice {
                                minor,
                                entry: self._entry.clone(),
                                instance: self.instance.clone(),
                                opened,
                            });
                    }
                    _ => {
                        self.device.destroy_device(None);
                        self.instance.destroy_instance(None);
                    }
                }
            }
        }
    }
}

// --- helpers ---

/// The per-frame pipelines of the dynamic-HDR fp16 path, one per (output state, content
/// transfer) pair the output format can carry.
struct DynamicPipelines {
    /// HDR output pipelines; `None` on NV12, which never carries HDR.
    hdr_out: Option<HdrOutPipelines>,
    /// SDR output: PQ content tone-mapped to the SDR output matrix.
    pq_to_sdr: vk::Pipeline,
}

#[derive(Clone, Copy)]
struct HdrOutPipelines {
    /// HDR output: SDR sRGB content -> PQ at the SDR reference white.
    sdr_to_pq: vk::Pipeline,
    /// HDR output: PQ content, matrix only.
    pq_pass: vk::Pipeline,
}

impl DynamicPipelines {
    /// Whether `color`'s frame is converted for an HDR output. An output that cannot carry
    /// HDR (NV12) is SDR whatever the requested state.
    fn output_hdr(&self, color: FrameColor) -> bool {
        color.output_hdr && self.hdr_out.is_some()
    }

    /// The pipeline for `color`, or `None` for SDR content on an SDR frame, which is the
    /// converter's regular pipeline.
    fn pick(&self, color: FrameColor) -> Option<vk::Pipeline> {
        let hdr_out = self.hdr_out.filter(|_| color.output_hdr);
        Some(match (hdr_out, color.content) {
            (Some(h), ContentTransfer::Sdr) => h.sdr_to_pq,
            (Some(h), ContentTransfer::Pq) => h.pq_pass,
            (None, ContentTransfer::Pq) => self.pq_to_sdr,
            (None, ContentTransfer::Sdr) => return None,
        })
    }

    unsafe fn destroy(self, device: &ash::Device) {
        if let Some(h) = self.hdr_out {
            device.destroy_pipeline(h.sdr_to_pq, None);
            device.destroy_pipeline(h.pq_pass, None);
        }
        device.destroy_pipeline(self.pq_to_sdr, None);
    }
}

/// Build the converter's pipeline(s). Under dynamic HDR on the fp16 path the regular
/// pipeline is the format's SDR shader and [`DynamicPipelines`] covers every other
/// (output, content) pair, so a mid-stream bt709 <-> bt2100-pq flip or a PQ client on an
/// NV12 stream needs no rebuild. Otherwise the single pipeline follows the negotiated caps
/// (`bt2020`). The SDR reference white feeds specialization constant 0 of the shaders that
/// declare it.
unsafe fn build_pipelines(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    fmt: PixFmt,
    bt2020: bool,
    fp16_input: bool,
    options: &ConversionOptions,
) -> Result<(vk::Pipeline, Option<DynamicPipelines>), Err> {
    let ref_white = options.sdr_reference_white;
    if !(options.hdr_cm && fp16_input) {
        let pipeline = build_compute_pipeline(device, layout, fmt.shader(bt2020), ref_white)?;
        return Ok((pipeline, None));
    }
    let shaders: &[&[u8]] = match fmt {
        PixFmt::P010 => &[
            RGBA_TO_P010_SPV,
            RGBA_HDR_TO_P010_SDR_SPV,
            RGBA_TO_P010_BT2020_SPV,
            RGBA_PQPASS_TO_P010_SPV,
        ],
        PixFmt::Nv12 => &[RGBA_TO_NV12_SPV, RGBA_HDR_TO_NV12_SDR_SPV],
    };
    let mut built = Vec::with_capacity(shaders.len());
    for spv in shaders {
        match build_compute_pipeline(device, layout, spv, ref_white) {
            Ok(p) => built.push(p),
            Err(e) => {
                for p in built {
                    device.destroy_pipeline(p, None);
                }
                return Err(e);
            }
        }
    }
    let dynamic = DynamicPipelines {
        hdr_out: (built.len() == 4).then(|| HdrOutPipelines {
            sdr_to_pq: built[2],
            pq_pass: built[3],
        }),
        pq_to_sdr: built[1],
    };
    tracing::debug!("VulkanNv12: dynamic-HDR pipelines ready ({fmt:?})");
    Ok((built[0], Some(dynamic)))
}

/// Build one RGBA->NV12/P010 compute pipeline from `spv` on `device` with `layout`, binding the
/// SDR reference-white value to specialization constant 0 (constant_id 0; Vulkan ignores it for
/// a shader that doesn't reference it). The shader module is freed before returning.
unsafe fn build_compute_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    spv: &[u8],
    sdr_ref_white: f32,
) -> Result<vk::Pipeline, Err> {
    let module = device.create_shader_module(
        &vk::ShaderModuleCreateInfo {
            code_size: spv.len(),
            p_code: spv.as_ptr() as *const u32,
            ..Default::default()
        },
        None,
    )?;
    let spec_data = sdr_ref_white.to_ne_bytes();
    let spec_entries = [vk::SpecializationMapEntry::default()
        .constant_id(0)
        .offset(0)
        .size(std::mem::size_of::<f32>())];
    let spec_info = vk::SpecializationInfo::default()
        .map_entries(&spec_entries)
        .data(&spec_data);
    let entry_name = c"main";
    let result = device.create_compute_pipelines(
        vk::PipelineCache::null(),
        &[vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(entry_name)
                    .specialization_info(&spec_info),
            )
            .layout(layout)],
        None,
    );
    device.destroy_shader_module(module, None);
    Ok(result.map_err(|(_, e)| e)?[0])
}

/// LINEAR NV12/P010 image with per-plane storage views (R8/R8G8 for NV12, R16/R16G16 for
/// P010). Storage works on LINEAR; it does not on DCC modifiers. Used as the tiled path's
/// compute scratch and, on the direct path, as the export image itself (with `usage`
/// extended for export). The image is `MUTABLE_FORMAT` with a view-format list so the planes
/// can be stored through the size/class-compatible single-component views.
unsafe fn create_storage(
    device: &ash::Device,
    memp: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
    export: bool,
    fmt: PixFmt,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::ImageView), Err> {
    let mods = [DRM_FORMAT_MOD_LINEAR];
    let mut modlist =
        vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&mods);
    let view_formats = [
        fmt.image_format(),
        fmt.y_view_format(),
        fmt.uv_view_format(),
    ];
    let mut flist = vk::ImageFormatListCreateInfo::default().view_formats(&view_formats);
    let mut extmem = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE)
        .image_type(vk::ImageType::TYPE_2D)
        .format(fmt.image_format())
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut modlist)
        .push_next(&mut flist);
    if export {
        info = info.push_next(&mut extmem);
    }
    let image = device.create_image(&info, None)?;
    let mut planes = StoragePlanes {
        image,
        ..Default::default()
    };
    match fill_storage(device, memp, export, &mut planes, fmt) {
        Ok(()) => Ok((planes.image, planes.mem, planes.y_view, planes.uv_view)),
        Err(e) => {
            planes.destroy(device);
            Err(e)
        }
    }
}

/// A storage image and the objects made over it; null handles are not yet created.
#[derive(Default)]
struct StoragePlanes {
    image: vk::Image,
    mem: vk::DeviceMemory,
    y_view: vk::ImageView,
    uv_view: vk::ImageView,
}

impl StoragePlanes {
    unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_image_view(self.y_view, None);
        device.destroy_image_view(self.uv_view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.mem, None);
    }
}

/// Back `planes.image` with memory (exportable when `export`) and make its plane views.
unsafe fn fill_storage(
    device: &ash::Device,
    memp: &vk::PhysicalDeviceMemoryProperties,
    export: bool,
    planes: &mut StoragePlanes,
    fmt: PixFmt,
) -> Result<(), Err> {
    let image = planes.image;
    let mr = device.get_image_memory_requirements(image);
    let mut exp = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(mr.size)
        .memory_type_index(mem_type(
            memp,
            mr.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?);
    if export {
        alloc = alloc.push_next(&mut exp);
    }
    planes.mem = device.allocate_memory(&alloc, None)?;
    device.bind_image_memory(image, planes.mem, 0)?;
    planes.y_view = plane_view(
        device,
        image,
        fmt.y_view_format(),
        vk::ImageAspectFlags::PLANE_0,
    )?;
    planes.uv_view = plane_view(
        device,
        image,
        fmt.uv_view_format(),
        vk::ImageAspectFlags::PLANE_1,
    )?;
    Ok(())
}

/// Build one self-contained export ring slot (export image + dmabuf, compute target +
/// storage views, command buffer, fence, descriptor set).
#[allow(clippy::too_many_arguments)]
unsafe fn create_output(
    instance: &ash::Instance,
    device: &ash::Device,
    memp: &vk::PhysicalDeviceMemoryProperties,
    allocator: &DmaBufAllocator,
    desc_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    cmd_pool: vk::CommandPool,
    width: u32,
    height: u32,
    modifier: u64,
    direct: bool,
    fmt: PixFmt,
    va_share: &crate::utils::va_share::VaShare,
) -> Result<Nv12Out, Err> {
    let mut out = Nv12Out::empty();
    match fill_output(
        &mut out,
        instance,
        device,
        memp,
        allocator,
        desc_pool,
        desc_layout,
        cmd_pool,
        width,
        height,
        modifier,
        direct,
        fmt,
        va_share,
    ) {
        Ok(()) => Ok(out),
        Err(e) => {
            destroy_output(device, cmd_pool, out, true);
            Err(e)
        }
    }
}

/// Build `out` in place, storing each object as soon as it exists so a failure part-way
/// leaves `out` holding exactly what must be freed.
#[allow(clippy::too_many_arguments)]
unsafe fn fill_output(
    out: &mut Nv12Out,
    instance: &ash::Instance,
    device: &ash::Device,
    memp: &vk::PhysicalDeviceMemoryProperties,
    allocator: &DmaBufAllocator,
    desc_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    cmd_pool: vk::CommandPool,
    width: u32,
    height: u32,
    modifier: u64,
    direct: bool,
    fmt: PixFmt,
    va_share: &crate::utils::va_share::VaShare,
) -> Result<(), Err> {
    // The export image: direct (LINEAR, compute-writable) carries STORAGE; tiled is a
    // pure TRANSFER_DST target filled by vkCmdCopyImage.
    if direct {
        let (image, mem, y_view, uv_view) = create_storage(
            device,
            memp,
            width,
            height,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST
                | if dump_path().is_some() {
                    vk::ImageUsageFlags::TRANSFER_SRC
                } else {
                    vk::ImageUsageFlags::empty()
                },
            true,
            fmt,
        )?;
        (out.image, out.mem, out.y_view, out.uv_view) = (image, mem, y_view, uv_view);
    } else {
        let mods = [modifier];
        let mut modlist =
            vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&mods);
        let mut extmem = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let image = device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(fmt.image_format())
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut extmem)
                .push_next(&mut modlist),
            None,
        )?;
        out.image = image;
        let mr = device.get_image_memory_requirements(image);
        let mut exp = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        out.mem = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(mr.size)
                .memory_type_index(mem_type(
                    memp,
                    mr.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?)
                .push_next(&mut exp),
            None,
        )?;
        device.bind_image_memory(image, out.mem, 0)?;
        // Per-slot LINEAR scratch the compute shader writes; copied into `image`.
        let (s_img, s_mem, s_y, s_uv) = create_storage(
            device,
            memp,
            width,
            height,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            false,
            fmt,
        )?;
        out.scratch = Some((s_img, s_mem));
        (out.y_view, out.uv_view) = (s_y, s_uv);
    }
    let (image, mem) = (out.image, out.mem);

    let ext_fd = ash::khr::external_memory_fd::Device::new(instance, device);
    // `export_fd` stays ours: it attaches the per-frame write fence (the sync_file attaches to
    // the shared underlying BO/dma_resv) and is closed in Drop. Each gst memory gets its own
    // dup and closes it when the last buffer referencing it is freed, so neither the fds nor
    // the BO they pin outlive their users across renegotiations.
    let export_fd = ext_fd.get_memory_fd(
        &vk::MemoryGetFdInfoKHR::default()
            .memory(mem)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
    )?;
    out.export_fd = export_fd;
    let dup_export = || -> Result<std::os::fd::OwnedFd, Err> {
        Ok(std::os::fd::BorrowedFd::borrow_raw(export_fd).try_clone_to_owned()?)
    };
    let l0 = device.get_image_subresource_layout(
        image,
        vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT),
    );
    let l1 = device.get_image_subresource_layout(
        image,
        vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_1_EXT),
    );

    // Build the gst buffer once over this fd; it is re-emitted (ref-counted) per use.
    // Preferred: a VA-allocator buffer carrying a VA surface on the shared (encoder)
    // display, so the encoder *reuses* it (zero per-frame imports). Fallback: a plain
    // dmabuf buffer (non-VA encoders, or no VA display shared yet).
    let mr = device.get_image_memory_requirements(image);
    let layout = crate::utils::va_share::Nv12Layout {
        format: fmt.gst_format(),
        width,
        height,
        fourcc: fmt.fourcc(),
        modifier,
        y_offset: l0.offset as usize,
        y_stride: l0.row_pitch as i32,
        uv_offset: l1.offset as usize,
        uv_stride: l1.row_pitch as i32,
    };
    // VA surface sharing, for NV12 and P010 alike: without a VA-allocator buffer the encoder
    // imports a new surface for every frame and runs out of reconstruct surfaces within
    // seconds ("Failed to create the reconstruct picture").
    let shared = va_share.build_shared_buffer(dup_export()?, mr.size as usize, &layout);
    out.buffer = match shared {
        Some(b) => b,
        None => {
            let mut buffer = GstBuffer::new();
            {
                let b = buffer.get_mut().unwrap();
                let gmem = allocator.alloc_dmabuf_with_flags(
                    dup_export()?.into_raw_fd(),
                    mr.size as usize,
                    FdMemoryFlags::NONE,
                )?;
                b.append_memory(gmem);
                VideoMeta::add_full(
                    b,
                    gst_video::VideoFrameFlags::empty(),
                    fmt.gst_format(),
                    width,
                    height,
                    &[l0.offset as usize, l1.offset as usize],
                    &[l0.row_pitch as i32, l1.row_pitch as i32],
                )?;
            }
            buffer
        }
    };

    // Per-slot command buffer, fence, descriptor set; bind the (constant) compute output
    // planes now -- the source (binding 0) is rebound each frame in convert().
    fill_slot_sync(out, device, desc_pool, desc_layout, cmd_pool)
}

/// Allocate `out`'s command buffer, fence and descriptor set, and bind its (constant)
/// compute output planes; the source (binding 0) is rebound each frame in convert().
unsafe fn fill_slot_sync(
    out: &mut Nv12Out,
    device: &ash::Device,
    desc_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    cmd_pool: vk::CommandPool,
) -> Result<(), Err> {
    out.cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .command_buffer_count(1),
    )?[0];
    out.fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
    let dsls = [desc_layout];
    out.desc_set = device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&dsls),
    )?[0];
    let y_info = [image_info(out.y_view)];
    let uv_info = [image_info(out.uv_view)];
    device.update_descriptor_sets(
        &[
            write_img(out.desc_set, 1, vk::DescriptorType::STORAGE_IMAGE, &y_info),
            write_img(out.desc_set, 2, vk::DescriptorType::STORAGE_IMAGE, &uv_info),
        ],
        &[],
    );
    Ok(())
}

/// Free one ring slot. `owns_image`: the export image, its memory and export fd are ours
/// (dmabuf path), not the encoder pool's (encode-src path). The cached import is freed by
/// the caller; the descriptor set goes back with its pool.
unsafe fn destroy_output(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    out: Nv12Out,
    owns_image: bool,
) {
    device.destroy_image_view(out.y_view, None);
    device.destroy_image_view(out.uv_view, None);
    if let Some((s_img, s_mem)) = out.scratch {
        device.destroy_image(s_img, None);
        device.free_memory(s_mem, None);
    }
    if owns_image {
        device.destroy_image(out.image, None);
        device.free_memory(out.mem, None);
        if out.export_fd >= 0 {
            libc::close(out.export_fd);
        }
    }
    device.destroy_fence(out.fence, None);
    if out.cmd != vk::CommandBuffer::null() {
        device.free_command_buffers(cmd_pool, &[out.cmd]);
    }
}

/// Build one encode-src ring slot: a buffer from the encoder's `GstVulkanImageBufferPool`
/// (a single multiplanar NV12 `VIDEO_ENCODE_SRC` image), a LINEAR storage scratch the
/// compute shader writes, and the per-slot cmd/fence/descriptor set. No dmabuf export --
/// the pool owns the output image's memory.
#[allow(clippy::too_many_arguments)]
unsafe fn create_encode_output(
    device: &ash::Device,
    memp: &vk::PhysicalDeviceMemoryProperties,
    gst_device: &gstreamer_vulkan::VulkanDevice,
    targets: &[crate::utils::video_info::EncodeTarget],
    queue_families: &[u32],
    desc_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    cmd_pool: vk::CommandPool,
    width: u32,
    height: u32,
    fmt: PixFmt,
    linear_encode_src: bool,
) -> Result<Nv12Out, Err> {
    let mut out = Nv12Out::empty();
    match fill_encode_output(
        &mut out,
        device,
        memp,
        gst_device,
        targets,
        queue_families,
        desc_pool,
        desc_layout,
        cmd_pool,
        width,
        height,
        fmt,
        linear_encode_src,
    ) {
        Ok(()) => Ok(out),
        Err(e) => {
            destroy_output(device, cmd_pool, out, false);
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn fill_encode_output(
    out: &mut Nv12Out,
    device: &ash::Device,
    memp: &vk::PhysicalDeviceMemoryProperties,
    gst_device: &gstreamer_vulkan::VulkanDevice,
    targets: &[crate::utils::video_info::EncodeTarget],
    queue_families: &[u32],
    desc_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    cmd_pool: vk::CommandPool,
    width: u32,
    height: u32,
    fmt: PixFmt,
    linear_encode_src: bool,
) -> Result<(), Err> {
    let buffer = crate::utils::vulkan_share::alloc_encode_src_buffer(
        gst_device,
        width,
        height,
        targets,
        fmt,
        linear_encode_src,
        queue_families,
    )
    .ok_or("encode-src image allocation failed")?;
    let out_img = crate::utils::vulkan_share::recover_vk_image(&buffer)
        .ok_or("encode-src buffer is not a single GstVulkanImageMemory")?;
    // The pool owns the encode-src image memory; `mem` stays null.
    out.buffer = buffer;
    out.image = out_img;

    // LINEAR storage scratch (compute writes here; copied into the encode-src image).
    let (s_img, s_mem, y_view, uv_view) = create_storage(
        device,
        memp,
        width,
        height,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        false,
        fmt,
    )?;
    out.scratch = Some((s_img, s_mem));
    (out.y_view, out.uv_view) = (y_view, uv_view);

    // Diagnostic for the RX 9070 green-bar/jump report: if the encoder's pool gives us an
    // encode-src image padded beyond width*height*3/2 (e.g. RDNA4 row-alignment), our copy
    // fills only `height` rows and the padding stays zeroed -> green at the bottom.
    let tight_nv12 = width as u64 * height as u64 * 3 / 2;
    tracing::debug!(
        "encode-src slot: req={width}x{height} NV12; out_img mem={} scratch mem={} tight={tight_nv12}",
        device.get_image_memory_requirements(out_img).size,
        device.get_image_memory_requirements(s_img).size,
    );

    fill_slot_sync(out, device, desc_pool, desc_layout, cmd_pool)
}

fn mem_type(
    memp: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, Err> {
    (0..memp.memory_type_count)
        .find(|&i| {
            bits & (1 << i) != 0 && memp.memory_types[i as usize].property_flags.contains(flags)
        })
        .ok_or_else(|| "no suitable memory type".into())
}

unsafe fn plane_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView, Err> {
    Ok(device.create_image_view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }),
        None,
    )?)
}

fn plane_layers(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(aspect)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
}

fn dsl_bind(binding: u32, ty: vk::DescriptorType) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(ty)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn pool_size(ty: vk::DescriptorType, count: u32) -> vk::DescriptorPoolSize {
    vk::DescriptorPoolSize::default()
        .ty(ty)
        .descriptor_count(count)
}

fn image_info(view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::GENERAL)
}

fn write_img<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    info: &'a [vk::DescriptorImageInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(ty)
        .image_info(info)
}

fn img_barrier(
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(from)
        .new_layout(to)
        .src_access_mask(src)
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

#[cfg(test)]
mod legacy_environment_tests {
    #[test]
    fn legacy_diagnostics_and_peak_default_are_preserved() {
        if std::env::var_os("GST_LEGACY_ENV_TEST_CHILD").is_some() {
            assert_eq!(super::ConversionOptions::default().hdr_peak_nits, 1600);
            assert_eq!(
                super::ConversionOptions::default().sdr_reference_white,
                250.0
            );
            assert_eq!(super::dump_path(), Some("/tmp/legacy-frame"));
            assert_eq!(super::dump_frame_target(), 31);
            return;
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "utils::vulkan_nv12::legacy_environment_tests::legacy_diagnostics_and_peak_default_are_preserved"])
            .env("GST_LEGACY_ENV_TEST_CHILD", "1").env("WOLF_HDR_PEAK_NITS", "1600")
            .env("WOLF_SDR_REFERENCE_WHITE", "250")
            .env("WOLF_VULKAN_DUMP", "/tmp/legacy-frame").env("WOLF_VULKAN_DUMP_FRAME", "31")
            .status().unwrap();
        assert!(status.success());
    }
}

#[cfg(test)]
mod conversion_tests {
    //! Pixel-exact checks of the converter on a real GPU: a solid fp16 RGBA dmabuf goes in,
    //! the exported NV12/P010 planes are read back and compared with the expected codes.
    use super::*;
    use smithay::backend::allocator::dmabuf::DmabufAllocator;
    use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags};
    use smithay::backend::allocator::{Allocator, Modifier};

    macro_rules! skip {
        ($($a:tt)*) => {{ eprintln!("skip: {}", format!($($a)*)); return; }};
    }

    const W: u32 = 64;
    const H: u32 = 64;

    /// SMPTE ST 2084 inverse EOTF: normalised linear light (1.0 = 10000 nits) -> PQ code.
    fn pq_oetf(l: f64) -> f64 {
        let (m1, m2) = (0.1593017578125, 78.84375);
        let (c1, c2, c3) = (0.8359375, 18.8515625, 18.6875);
        let lm = l.powf(m1);
        ((c1 + c2 * lm) / (1.0 + c3 * lm)).powf(m2)
    }

    /// A LINEAR fp16 RGBA dmabuf on `node` filled with `rgb` (alpha 1), drawn by GLES on the
    /// same GPU as the compositor draws its frames. Filling it with the CPU instead lets the
    /// GPU sample a stale cache sector of whatever the memory held before (one in a few
    /// hundred conversions on NVIDIA), which the compositor's GPU-drawn frames never hit.
    fn solid_fp16(node: DrmNode, rgb: [f32; 3]) -> Option<Dmabuf> {
        use smithay::backend::renderer::{Bind, Frame, Renderer};
        use smithay::utils::{Physical, Rectangle, Size, Transform};
        let gbm = crate::utils::allocator::new_gbm_device(node)?;
        let mut alloc = DmabufAllocator(GbmAllocator::new(gbm, GbmBufferFlags::RENDERING));
        let dmabuf = alloc
            .create_buffer(W, H, DrmFourcc::Abgr16161616f, &[Modifier::Linear])
            .ok()?;
        let mut renderer = crate::utils::renderer::setup_renderer(Some(node));
        let size: Size<i32, Physical> = (W as i32, H as i32).into();
        let mut bound = dmabuf.clone();
        let mut target = renderer.bind(&mut bound).ok()?;
        let mut frame = renderer.render(&mut target, size, Transform::Normal).ok()?;
        frame
            .clear(
                [rgb[0], rgb[1], rgb[2], 1.0].into(),
                &[Rectangle::from_size(size)],
            )
            .ok()?;
        frame.finish().ok()?.wait().ok()?;
        drop(target);
        Some(dmabuf)
    }

    /// Convert a solid `rgb` frame with `color` and return every (Y, Cb, Cr) sample code
    /// (P010: the 10-bit code, after checking the low 6 bits are zero).
    fn convert_solid(
        node: DrmNode,
        fmt: PixFmt,
        rgb: [f32; 3],
        color: FrameColor,
    ) -> Option<(Vec<u16>, Vec<(u16, u16)>)> {
        let src = solid_fp16(node, rgb)?;
        let drm_format = match fmt {
            PixFmt::Nv12 => "NV12",
            PixFmt::P010 => "P010",
        };
        let caps = gst_video::VideoCapsBuilder::new()
            .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .format(gst_video::VideoFormat::DmaDrm)
            .field("drm-format", drm_format)
            .width(W as i32)
            .height(H as i32)
            .framerate(gst::Fraction::new(30, 1))
            .build();
        let info = VideoInfoDmaDrm::from_caps(&caps).ok()?;
        let options = ConversionOptions {
            hdr_cm: true,
            ..Default::default()
        };
        let va_share = crate::utils::va_share::VaShare::default();
        let mut conv = VulkanNv12::new(node, info, fmt, false, true, &va_share, options)?;
        conv.convert(&src, color).expect("convert");
        let buffer = conv.to_gst_buffer().expect("output buffer");
        let mem = buffer.peek_memory(0);
        let dma = mem
            .downcast_memory_ref::<gstreamer_allocators::DmaBufMemory>()
            .expect("dmabuf memory");
        // The export carries our write fence: wait for it like an implicit-sync reader.
        let mut pfd = libc::pollfd {
            fd: dma.fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        assert_eq!(
            unsafe { libc::poll(&mut pfd, 1, 5000) },
            1,
            "write fence never signalled"
        );
        let bpp = fmt.y_bytes() as usize;
        // Read what the shader wrote through Vulkan, as the GPU consumers do: RADV's export
        // is in memory the CPU can't map.
        let data = conv.read_output().expect("GPU readback");
        let row = W as usize * bpp;
        let (offsets, strides) = ([0, row * H as usize], [row, row]);
        let sample = |off: usize| -> u16 {
            if bpp == 1 {
                data[off] as u16
            } else {
                let v = u16::from_le_bytes([data[off], data[off + 1]]);
                assert_eq!(v & 0x3f, 0, "P010 sample {v:#06x} has low bits set");
                v >> 6
            }
        };
        let mut ys = Vec::new();
        for y in 0..H as usize {
            for x in 0..W as usize {
                ys.push(sample(offsets[0] + y * strides[0] + x * bpp));
            }
        }
        let mut cs = Vec::new();
        for y in 0..H as usize / 2 {
            for x in 0..W as usize / 2 {
                let o = offsets[1] + y * strides[1] + x * 2 * bpp;
                cs.push((sample(o), sample(o + bpp)));
            }
        }
        Some((ys, cs))
    }

    /// The converters these tests open share the process-wide device pool; one test at a
    /// time, or another test can take the device a test expects back.
    static POOL: Mutex<()> = Mutex::new(());

    fn node() -> Option<DrmNode> {
        gst::init().unwrap();
        let path = std::env::var("NV12_TEST_NODE").ok()?;
        DrmNode::from_path(path).ok()
    }

    fn assert_solid(
        label: &str,
        (ys, cs): &(Vec<u16>, Vec<(u16, u16)>),
        y: u16,
        cb: u16,
        cr: u16,
        tol: u16,
    ) {
        let near = |a: u16, b: u16| a.abs_diff(b) <= tol;
        if let Some(i) = ys.iter().position(|&v| !near(v, y)) {
            panic!(
                "{label}: Y[{i}] (x {}, y {}) = {} (expected {y}); {} of {} samples wrong",
                i % W as usize,
                i / W as usize,
                ys[i],
                ys.iter().filter(|&&v| !near(v, y)).count(),
                ys.len()
            );
        }
        if let Some(i) = cs.iter().position(|&(b, r)| !(near(b, cb) && near(r, cr))) {
            panic!(
                "{label}: CbCr[{i}] = {:?} (expected {cb},{cr}); {} of {} samples wrong",
                cs[i],
                cs.iter()
                    .filter(|&&(b, r)| !(near(b, cb) && near(r, cr)))
                    .count(),
                cs.len()
            );
        }
    }

    // Converters on one GPU reuse a device instead of destroying it (see `PooledDevice`).
    #[test]
    #[ignore = "needs a GPU render node in NV12_TEST_NODE; run via ci/harness.sh gpu"]
    fn a_dropped_converter_hands_its_device_to_the_next_one() {
        let _pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(node) = node() else {
            skip!("NV12_TEST_NODE unset");
        };
        let caps = gst_video::VideoCapsBuilder::new()
            .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .format(gst_video::VideoFormat::DmaDrm)
            .field("drm-format", "NV12")
            .width(W as i32)
            .height(H as i32)
            .framerate(gst::Fraction::new(30, 1))
            .build();
        let info = VideoInfoDmaDrm::from_caps(&caps).unwrap();
        let va_share = crate::utils::va_share::VaShare::default();
        let open = || {
            VulkanNv12::new(
                node,
                info.clone(),
                PixFmt::Nv12,
                false,
                false,
                &va_share,
                ConversionOptions::default(),
            )
            .expect("converter")
        };
        let first = open();
        let device = first.device.handle();
        drop(first);
        let second = open();
        assert_eq!(second.device.handle(), device, "the device was not reused");
        // A converter alive at the same time gets a device of its own.
        let third = open();
        assert_ne!(third.device.handle(), device);
    }

    #[test]
    #[ignore = "needs a GPU render node in NV12_TEST_NODE; run via ci/harness.sh gpu"]
    fn p010_samples_are_exact_10_bit_limited_range_codes() {
        let _pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(node) = node() else {
            skip!("NV12_TEST_NODE unset");
        };
        let sdr = FrameColor::default();
        let Some(white) = convert_solid(node, PixFmt::P010, [1.0; 3], sdr) else {
            skip!("fp16 source or P010 converter unavailable on this GPU");
        };
        // BT.709 limited range: white 940, black 64, neutral chroma 512, exact.
        assert_solid("p010 white", &white, 940, 512, 512, 0);
        let black = convert_solid(node, PixFmt::P010, [0.0; 3], sdr).unwrap();
        assert_solid("p010 black", &black, 64, 512, 512, 0);
        // Red: Y = 64 + 876 * 0.2126, Cb = 512 - 896 * 0.2126 / 1.8556, Cr = 960.
        let red = convert_solid(node, PixFmt::P010, [1.0, 0.0, 0.0], sdr).unwrap();
        assert_solid("p010 red", &red, 250, 409, 960, 0);
        // HDR output, PQ content: BT.2020 matrix only.
        let pq = FrameColor {
            output_hdr: true,
            content: ContentTransfer::Pq,
        };
        let grey = convert_solid(node, PixFmt::P010, [0.5; 3], pq).unwrap();
        assert_solid("p010 pq grey", &grey, 502, 512, 512, 0);
        let pq_red = convert_solid(node, PixFmt::P010, [0.5, 0.0, 0.0], pq).unwrap();
        assert_solid("p010 pq red", &pq_red, 179, 449, 736, 0);
    }

    #[test]
    #[ignore = "needs a GPU render node in NV12_TEST_NODE; run via ci/harness.sh gpu"]
    fn nv12_tone_maps_pq_content_instead_of_showing_it_as_gamma() {
        let _pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(node) = node() else {
            skip!("NV12_TEST_NODE unset");
        };
        // PQ-encoded SDR reference white (203 nits): tone-mapped, it lands near the top of
        // the SDR range; passed through the sRGB shader it would read as mid grey (~143).
        let ref_white = pq_oetf(203.0 / 10000.0) as f32;
        let roll = 0.8 + 0.2 * (1.0 - (-(1.0f64 - 0.8) / 0.2).exp());
        let srgb = 1.055 * roll.powf(1.0 / 2.4) - 0.055;
        let expect_y = (255.0 * (0.859 * srgb + 0.0625)).round() as u16;
        for output_hdr in [false, true] {
            let color = FrameColor {
                output_hdr,
                content: ContentTransfer::Pq,
            };
            let Some(out) = convert_solid(node, PixFmt::Nv12, [ref_white; 3], color) else {
                skip!("fp16 source or NV12 converter unavailable on this GPU");
            };
            // NV12 never carries HDR: a requested HDR output is still tone-mapped to SDR.
            assert_solid(
                &format!("nv12 pq white (output_hdr={output_hdr})"),
                &out,
                expect_y,
                128,
                128,
                1,
            );
        }
        // SDR content keeps the regular BT.601 shader.
        let white = convert_solid(node, PixFmt::Nv12, [1.0; 3], FrameColor::default()).unwrap();
        assert_solid("nv12 sdr white", &white, 235, 128, 128, 1);
    }
}
