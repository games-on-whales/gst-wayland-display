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

use ash::vk;
use gst::Buffer as GstBuffer;
use gst_video::{VideoFormat, VideoInfoDmaDrm, VideoMeta};
use gstreamer_allocators::{DmaBufAllocator, DmaBufAllocatorExtManual, FdMemoryFlags};
use smithay::backend::allocator::Buffer as _;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::drm::DrmNode;
use smithay::reexports::drm::buffer::DrmFourcc;
use std::os::fd::{AsFd, AsRawFd, IntoRawFd, RawFd};
use std::sync::{Mutex, OnceLock};

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
/// can travel render-target -> converter -> P010 PQ as a brighter-than-white highlight. Selected
/// only on the P010+bt2020 path when `WOLF_HDR_SPIKE` is set. NOTE: the checked-in `.spv` is an
/// empty placeholder; compile `shaders/rgba_to_p010_hdr.comp` with glslc before using the spike.
// Retained for the WOLF_HDR_SPIKE linear-bars proof; the real SDR path uses the sRGB
// tone-map and HDR content uses the passthrough pipeline, so this is otherwise unused.
#[allow(dead_code)]
const RGBA_TO_P010_HDR_SPV: &[u8] = include_bytes!("shaders/rgba_to_p010_hdr.spv");
/// PQ-passthrough variant of the P010 converter (`WOLF_HDR_CM`). Same topology/bindings as
/// [`RGBA_TO_P010_BT2020_SPV`], but it applies ONLY the BT.2020 limited-range Y'CbCr matrix --
/// no sRGB EOTF, no 709->2020 gamut, no PQ OETF -- because its input is already PQ-encoded
/// BT.2020 (gamescope's 10-bit XB30/AB30/XR30/AR30 HDR output, composited into the fp16
/// target). Selected per frame via `convert(pq_passthrough=true)` so a 10-bit client frame
/// isn't double-PQ'd (washed out) by the tone-mapping shader. NOTE: the checked-in `.spv` is an
/// empty placeholder; compile `shaders/rgba_pqpass_to_p010.comp` with glslc before using it.
const RGBA_PQPASS_TO_P010_SPV: &[u8] = include_bytes!("shaders/rgba_pqpass_to_p010.spv");
const DRM_FORMAT_MOD_LINEAR: u64 = 0;
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
/// NV12 export ring depth (> encoder DPB/pipeline depth so a buffer is free by reuse).
// Ring depth: slots cycled to pipeline convert+encode (array sizing + default active depth).
// 4 is enough fan-out for the downstream encoder's buffer references (1 starves; 4 and 8 are
// indistinguishable in practice). Override the active depth at runtime with WOLF_VULKAN_RING
// (1..=RING) via ring_used().
const RING: usize = 4;

/// Optional path for a one-shot debug dump of the converter's NV12 output (the LINEAR
/// compute scratch, *before* the tiled encode-src copy) as raw NV12. Set
/// `WOLF_VULKAN_DUMP=<file>` to capture one frame -- if it's clean, the RX 9070 green-bar/
/// jump corruption is introduced downstream (the tiled copy or the encoder), not our
/// conversion. Read once; off (zero overhead) when unset.
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

/// Which converted frame to capture when `WOLF_VULKAN_DUMP` is set (default 20, a few frames
/// in so the scene has settled). Override with `WOLF_VULKAN_DUMP_FRAME` to wait longer -- e.g.
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

/// Number of ring slots actually cycled (`<= RING`; the per-slot arrays stay sized at
/// `RING`). Override with `WOLF_VULKAN_RING` for diagnosis -- e.g. `WOLF_VULKAN_RING=1` pins
/// a single encode-src slot, isolating whether per-slot encode-src images (address-dependent
/// tiling/swizzle) cause the RX 9070 / GFX12 "jumping" image. The compositor renders into one
/// stable RGBA buffer, so every slot converts identical content -- if a single slot is stable
/// but 8 jump, the encode-src pool images differ per slot. Read once.
fn ring_used() -> usize {
    use std::sync::OnceLock;
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| {
        let n = std::env::var("WOLF_VULKAN_RING")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| (1..=RING).contains(&n))
            .unwrap_or(RING);
        tracing::debug!("VulkanNv12: ring_used={n} (RING={RING})");
        n
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
    fn gst_format(self) -> VideoFormat {
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
    /// Pick the matching compute shader SPIR-V. `bt2020` selects the BT.2020 (HDR / BT.2100-PQ)
    /// matrix variant on the P010 path; `fp16_input` is true when the RGBA render target the
    /// converter samples is the linear fp16 (`Abgr16161616f` -> `R16G16B16A16_SFLOAT`) HDR
    /// target instead of an 8-bit sRGB target. On the P010+bt2020 path the input format -- not
    /// any env var -- chooses the shader: a linear fp16 input (real HDR client content, or the
    /// `WOLF_HDR_SPIKE` bars) feeds the linear-input PQ shader (no sRGB EOTF, values may exceed
    /// 1.0); an 8-bit sRGB input feeds the sRGB-input BT.2020/PQ tone-map. Neither flag affects
    /// NV12 (8-bit SDR stays BT.601).
    fn shader(self, bt2020: bool, fp16_input: bool) -> &'static [u8] {
        match (self, bt2020) {
            (PixFmt::Nv12, _) => RGBA_TO_NV12_SPV,
            (PixFmt::P010, false) => RGBA_TO_P010_SPV,
            (PixFmt::P010, true) => {
                // The "normal" (non-passthrough) pipeline. SDR client content -- whether on an
                // 8-bit target or stored sRGB-gamma in the fp16 target (Smithay does NOT
                // linearise when sampling SDR clients) -- needs the sRGB-EOTF + tone-map shader.
                // The linear-input shader (RGBA_TO_P010_HDR_SPV) only suited the synthetic
                // WOLF_HDR_SPIKE bars (genuinely linear); real gamescope/Steam SDR is sRGB-gamma,
                // so feeding it to the linear shader PQ-encoded gamma-as-linear -> washed out.
                // Already-PQ 10-bit content takes the separate passthrough pipeline instead.
                let _ = fp16_input;
                RGBA_TO_P010_BT2020_SPV
            }
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
const DMA_BUF_SYNC_WRITE: u32 = 1 << 1;
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
unsafe fn pick_physical_device(
    instance: &ash::Instance,
    target_minor: Option<u32>,
) -> Option<vk::PhysicalDevice> {
    let devices = instance.enumerate_physical_devices().ok()?;
    if let Some(minor) = target_minor {
        let matched = devices.iter().copied().find(|&d| {
            let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
            instance.get_physical_device_properties2(d, &mut p2);
            (drm.has_render != 0 && drm.render_minor as i64 == minor as i64)
                || (drm.has_primary != 0 && drm.primary_minor as i64 == minor as i64)
        });
        if matched.is_some() {
            return matched;
        }
        tracing::warn!(
            "VulkanNv12: no Vulkan device matches DRM minor {minor}; falling back to the \
             first device -- wrong GPU on a multi-GPU host"
        );
    }
    devices.first().copied()
}

/// DRM modifiers `target_minor`'s GPU can export NV12 with (queried once per GPU, cached).
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
    static CACHE: OnceLock<Mutex<Vec<(Option<u32>, PixFmt, &'static [u64])>>> = OnceLock::new();
    let mut guard = CACHE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    if let Some((_, _, mods)) = guard
        .iter()
        .find(|(m, f, _)| *m == target_minor && *f == fmt)
    {
        return mods;
    }
    let mods = unsafe { query_modifiers(target_minor, fmt.image_format()) }.unwrap_or_default();
    let leaked: &'static [u64] = Box::leak(mods.into_boxed_slice());
    guard.push((target_minor, fmt, leaked));
    leaked
}

unsafe fn query_modifiers(target_minor: Option<u32>, format: vk::Format) -> Option<Vec<u64>> {
    let entry = ash::Entry::load().ok()?;
    let instance = entry
        .create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2)),
            None,
        )
        .ok()?;
    let pd = pick_physical_device(&instance, target_minor)?;
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
    // must not drop it or negotiation fails outright.
    let mods = props
        .iter()
        .filter(|m| {
            m.drm_format_modifier_tiling_features
                .contains(vk::FormatFeatureFlags::TRANSFER_DST)
        })
        .map(|m| m.drm_format_modifier)
        .collect();
    instance.destroy_instance(None);
    Some(mods)
}

/// Map a compositor RGBA dmabuf fourcc to the Vulkan format used for the import.
fn rgba_vk_format(fourcc: DrmFourcc) -> Option<vk::Format> {
    Some(match fourcc {
        DrmFourcc::Abgr8888 | DrmFourcc::Xbgr8888 => vk::Format::R8G8B8A8_UNORM,
        DrmFourcc::Argb8888 | DrmFourcc::Xrgb8888 => vk::Format::B8G8R8A8_UNORM,
        // HDR render-path spike (WOLF_HDR_SPIKE): the compositor renders into an fp16 RGBA
        // dmabuf so highlights can exceed 1.0. The sampler then reads linear fp16 (no clamp),
        // which the rgba_to_p010_hdr shader expects (1.0 == SDR reference white).
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
    /// Cached RGBA import (keyed by the source dmabuf fd): the VkImage/memory/view over the
    /// compositor's RGBA dmabuf. The compositor renders into a single stable buffer, so
    /// this is created once per slot and reused every frame -- avoiding a per-frame
    /// create_image + dma-buf memory import + view. The import views the dmabuf the
    /// compositor overwrites in place, so a reused import reads the latest content. Rebuilt
    /// only if the fd ever changes; freed at drop.
    import: Option<(i32, Import)>,
    /// Whether `fence` has un-reclaimed GPU work (i.e. `cmd` is still in use).
    in_flight: bool,
}

/// Owns the Vulkan device, the compute pipeline, and the NV12 export ring.
pub struct VulkanNv12 {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    pipeline: vk::Pipeline,
    /// Per-frame PQ-passthrough pipeline (matrix-only, already-PQ BT.2020 input). `Some` only on
    /// the HDR fp16 P010 path; `convert(pq_passthrough=true)` binds it instead of `pipeline`.
    pipeline_pq: Option<vk::Pipeline>,
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
    outputs: Vec<Nv12Out>,
    next: usize, // next ring slot to write
    cur: usize,  // last slot written (the one to_gst_buffer returns)
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

impl VulkanNv12 {
    /// Bring up Vulkan, build the compute pipeline, and allocate the NV12 export ring for
    /// `video_info`'s resolution, exporting with the negotiated modifier.
    pub fn new(
        render_node: DrmNode,
        video_info: VideoInfoDmaDrm,
        fmt: PixFmt,
        bt2020: bool,
        fp16_input: bool,
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
    ) -> Result<Self, Err> {
        let entry = ash::Entry::load()?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )?;
        // Match by DRM render/primary minor (VK_EXT_physical_device_drm); warns and falls
        // back to the first device if no driver reports a matching node.
        let pd = pick_physical_device(&instance, Some(target_minor)).ok_or("no vulkan device")?;

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
        let mut exts = vec![
            ash::khr::external_memory_fd::NAME.as_ptr(),
            ash::ext::external_memory_dma_buf::NAME.as_ptr(),
            ash::ext::image_drm_format_modifier::NAME.as_ptr(),
        ];
        if implicit_sync {
            // Needed to export the submit's signal as a SYNC_FD we attach to the dmabuf.
            exts.push(ash::khr::external_semaphore_fd::NAME.as_ptr());
        }
        let mut ycbcr = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
            .sampler_ycbcr_conversion(true);
        let device = instance.create_device(
            pd,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&qci)
                .enabled_extension_names(&exts)
                .push_next(&mut ycbcr),
            None,
        )?;
        let queue = device.get_device_queue(qfi, 0);
        let memp = instance.get_physical_device_memory_properties(pd);

        // ---- compute pipeline(s) (shader selected by output format + input format) ----
        let binds = [
            dsl_bind(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            dsl_bind(1, vk::DescriptorType::STORAGE_IMAGE),
            dsl_bind(2, vk::DescriptorType::STORAGE_IMAGE),
        ];
        let desc_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
            None,
        )?;
        let dsls = [desc_layout];
        let pcr = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&dsls)
                .push_constant_ranges(&pcr),
            None,
        )?;
        // SDR reference white (nits) -> specialization constant 0 of the BT.2020/PQ shader,
        // so it's tunable via Wolf's [gstreamer.video] sdr_reference_white (passed as the
        // WOLF_SDR_REFERENCE_WHITE env) without recompiling. The other shaders don't declare
        // constant_id 0, and Vulkan ignores a spec entry an unused shader doesn't reference.
        let sdr_ref_white = sdr_reference_white();
        let pipeline = build_compute_pipeline(
            &device,
            pipeline_layout,
            normal_shader(fmt, bt2020, fp16_input),
            sdr_ref_white,
        )?;
        let pipeline_pq = build_pq_passthrough(
            &device,
            pipeline_layout,
            fmt,
            bt2020,
            fp16_input,
            sdr_ref_white,
        );

        // One descriptor set + command buffer per ring slot (pipelined, no contention).
        let psizes = [
            pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, RING as u32),
            pool_size(vk::DescriptorType::STORAGE_IMAGE, 2 * RING as u32),
        ];
        let desc_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(RING as u32)
                .pool_sizes(&psizes),
            None,
        )?;
        let sampler = device.create_sampler(&vk::SamplerCreateInfo::default(), None)?;
        let cmd_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(qfi)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;

        // ---- NV12 export ring ----
        // Prefer the LINEAR direct path (compute writes the export image; no scratch/copy)
        // when the negotiated modifier is LINEAR; if the driver won't give a STORAGE-able
        // LINEAR NV12 export image, fall back to the tiled scratch+copy path.
        let gst_allocator = DmaBufAllocator::new();
        let want_direct = export_modifier == DRM_FORMAT_MOD_LINEAR;
        let build = |direct: bool| -> Result<Vec<Nv12Out>, Err> {
            tracing::debug!(
                "VulkanNv12: creating {RING} export images {width}x{height} \
                 modifier={export_modifier:#x} direct={direct}"
            );
            let mut outputs = Vec::with_capacity(RING);
            for _ in 0..RING {
                outputs.push(create_output(
                    &instance,
                    &device,
                    &memp,
                    &gst_allocator,
                    desc_pool,
                    desc_layout,
                    cmd_pool,
                    width,
                    height,
                    export_modifier,
                    direct,
                    fmt,
                )?);
            }
            Ok(outputs)
        };
        let (outputs, direct) = match (want_direct, build(want_direct)) {
            (_, Ok(o)) => (o, want_direct),
            (true, Err(e)) => {
                tracing::warn!(
                    "VulkanNv12: LINEAR direct path unavailable ({e}); using scratch+copy"
                );
                (build(false)?, false)
            }
            (false, Err(e)) => return Err(e),
        };

        // Binary semaphore exported per submit as the dmabuf write-fence (implicit path).
        let (sync_sem, sem_fd) = if implicit_sync {
            let mut exp = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            let sem = device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut exp),
                None,
            )?;
            let loader = ash::khr::external_semaphore_fd::Device::new(&instance, &device);
            (sem, Some(loader))
        } else {
            (vk::Semaphore::null(), None)
        };

        Ok(VulkanNv12 {
            _entry: entry,
            instance,
            device,
            queue,
            cmd_pool,
            pipeline,
            pipeline_pq,
            pipeline_layout,
            desc_layout,
            desc_pool,
            sampler,
            memp,
            sync_sem,
            sem_fd,
            implicit_sync,
            direct,
            encode_src: false,
            _shared_device: None,
            outputs,
            next: 0,
            cur: 0,
            width,
            height,
            fmt,
        })
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
        profile: &str,
        width: u32,
        height: u32,
        fmt: PixFmt,
        bt2020: bool,
        fp16_input: bool,
    ) -> Option<Self> {
        match unsafe {
            Self::new_on_shared_inner(
                device_gst, raw, nv12_caps, profile, width, height, fmt, bt2020, fp16_input,
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
        profile: &str,
        width: u32,
        height: u32,
        fmt: PixFmt,
        bt2020: bool,
        fp16_input: bool,
    ) -> Result<Self, Err> {
        // Drive ash on the encoder's existing instance/device (do NOT create our own).
        let entry = ash::Entry::load()?;
        let instance = ash::Instance::load(entry.static_fn(), raw.instance);
        let device = ash::Device::load(instance.fp_v1_0(), raw.device);
        let queue = device.get_device_queue(raw.gfx_queue_family, 0);
        let memp = instance.get_physical_device_memory_properties(raw.physical);

        // ---- compute pipeline(s) (same as the dmabuf path, on the shared device) ----
        let binds = [
            dsl_bind(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            dsl_bind(1, vk::DescriptorType::STORAGE_IMAGE),
            dsl_bind(2, vk::DescriptorType::STORAGE_IMAGE),
        ];
        let desc_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
            None,
        )?;
        let dsls = [desc_layout];
        let pcr = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&dsls)
                .push_constant_ranges(&pcr),
            None,
        )?;
        // SDR reference white (nits) -> specialization constant 0 of the BT.2020/PQ shader,
        // so it's tunable via Wolf's [gstreamer.video] sdr_reference_white (passed as the
        // WOLF_SDR_REFERENCE_WHITE env) without recompiling. The other shaders don't declare
        // constant_id 0, and Vulkan ignores a spec entry an unused shader doesn't reference.
        let sdr_ref_white = sdr_reference_white();
        let pipeline = build_compute_pipeline(
            &device,
            pipeline_layout,
            normal_shader(fmt, bt2020, fp16_input),
            sdr_ref_white,
        )?;
        let pipeline_pq = build_pq_passthrough(
            &device,
            pipeline_layout,
            fmt,
            bt2020,
            fp16_input,
            sdr_ref_white,
        );
        let psizes = [
            pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, RING as u32),
            pool_size(vk::DescriptorType::STORAGE_IMAGE, 2 * RING as u32),
        ];
        let desc_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(RING as u32)
                .pool_sizes(&psizes),
            None,
        )?;
        let sampler = device.create_sampler(&vk::SamplerCreateInfo::default(), None)?;
        let cmd_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(raw.gfx_queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;

        // ---- encode-src output ring: NV12 VIDEO_ENCODE_SRC GstVulkanImageMemory images
        // allocated directly on the shared device (bypasses the pool's generic-format check).
        let mut outputs = Vec::with_capacity(RING);
        for _ in 0..RING {
            outputs.push(create_encode_output(
                &device,
                &memp,
                &device_gst,
                profile,
                desc_pool,
                desc_layout,
                cmd_pool,
                width,
                height,
                fmt,
            )?);
        }

        Ok(VulkanNv12 {
            _entry: entry,
            instance,
            device,
            queue,
            cmd_pool,
            pipeline,
            pipeline_pq,
            pipeline_layout,
            desc_layout,
            desc_pool,
            sampler,
            memp,
            sync_sem: vk::Semaphore::null(),
            sem_fd: None,
            implicit_sync: false,
            direct: false,
            encode_src: true,
            _shared_device: Some(device_gst),
            outputs,
            next: 0,
            cur: 0,
            width,
            height,
            fmt,
        })
    }

    /// Import the compositor's `rgba` dmabuf and convert it RGBA->NV12 into the next
    /// export ring slot. `pq_passthrough` selects the per-frame PQ-passthrough shader (the
    /// frame's content came from a 10-bit, already-PQ BT.2020 client buffer) when that pipeline
    /// is available (HDR fp16 P010 path); otherwise the normal tone-mapping shader runs.
    pub fn convert(&mut self, rgba: &Dmabuf, pq_passthrough: bool) -> Result<(), Err> {
        unsafe { self.convert_inner(rgba, pq_passthrough) }
    }

    unsafe fn convert_inner(&mut self, rgba: &Dmabuf, pq_passthrough: bool) -> Result<(), Err> {
        // Cache key: the source dmabuf's primary fd. The compositor renders into a single
        // stable RGBA buffer, so this is constant across frames and each slot's import is
        // built only once.
        let key_fd = rgba.handles().next().ok_or("no fd")?.as_fd().as_raw_fd();

        // pick the next export ring slot and reclaim its previous (RING-frames-old) work
        let idx = self.next;
        self.next = (self.next + 1) % ring_used().min(self.outputs.len());
        if self.outputs[idx].in_flight {
            // This fence was signalled RING frames ago, so this is a no-op wait in steady
            // state -- it just lets us safely recycle the slot's cmd buffer.
            self.device
                .wait_for_fences(&[self.outputs[idx].fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.outputs[idx].fence])?;
            self.outputs[idx].in_flight = false;
        }

        // Fan-out safety (encode-src path): one produced buffer can be referenced by several
        // downstream encoders at once (interpipe delivers it to every consumer). Our own
        // graphics fence above only proves *our* last write to this slot finished -- it says
        // nothing about the consumers' encode *reads*, which run on the encode queue with no
        // shared sync to us. Overwriting the slot's image while an encode still reads it is a
        // GPU data hazard that wedges the encoder. Block until every consumer has dropped its
        // ref (the buffer is writable again, refcount back to 1) before reusing the slot.
        if self.encode_src {
            let mut waited = 0u32;
            while self.outputs[idx].buffer.get_mut().is_none() {
                // ~1s cap so a paused/stalled consumer can't deadlock the producer forever.
                if waited >= 10_000 {
                    tracing::warn!(
                        "VulkanNv12: encode-src slot {idx} still referenced after 1s; reusing anyway"
                    );
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
                waited += 1;
            }
            // The slot's persistent GstBuffer is reused every RING frames and still carries
            // the timestamps stamped onto it on its previous use. BaseSrc's do_timestamp only
            // stamps a buffer whose PTS is NONE, so without a reset the exported PTS degenerate
            // to the ring's ~4 recurring values (non-monotonic, duplicated), which downstream
            // RTP payloading turns into timestamps libwebrtc cannot assemble. Reset the timing
            // metadata here, right after the reuse gate, where the slot is uniquely owned on
            // the normal path -- resetting in to_gst_buffer via make_mut() would copy the
            // refcount-2 buffer and blind the gate above (the copy keeps outputs[idx].buffer
            // at refcount 1 forever). On the gate's 1s-timeout escape the slot is NOT uniquely
            // owned; get_mut() returns None and the reset is skipped (the stream is already
            // degraded there, and an in-place write to a shared header would be worse).
            if let Some(b) = self.outputs[idx].buffer.get_mut() {
                b.set_pts(gst::ClockTime::NONE);
                b.set_dts(gst::ClockTime::NONE);
                b.set_duration(gst::ClockTime::NONE);
            } else {
                tracing::warn!(
                    "VulkanNv12: encode-src slot {idx} PTS reset skipped (buffer still shared)"
                );
            }
        }

        // Ensure this slot's cached RGBA import matches the current dmabuf. Built once per
        // slot (the slot is idle here -- we waited its fence above -- so freeing/recreating
        // is safe). Reused every frame since the dmabuf is stable; the import is a distinct
        // VkImage per slot (not shared), so its per-frame layout transition stays
        // serialised by the slot fence and never races another in-flight frame.
        let stale = !matches!(&self.outputs[idx].import, Some((fd, _)) if *fd == key_fd);
        if stale {
            self.free_import(idx);
            let import = self.create_import(rgba)?;
            self.outputs[idx].import = Some((key_fd, import));
        }
        let (import_img, _import_mem, view) = self.outputs[idx].import.as_ref().unwrap().1;

        let slot = &self.outputs[idx];
        let out_img = slot.image;
        let desc_set = slot.desc_set;
        let cmd = slot.cmd;
        let fence = slot.fence;
        let direct = self.direct;
        // Per-frame shader: the PQ-passthrough pipeline when this frame is already-PQ 10-bit
        // content and that pipeline exists, else the normal (tone-mapping / BT.601) pipeline.
        let pipeline = match (pq_passthrough, self.pipeline_pq) {
            (true, Some(p)) => p,
            _ => self.pipeline,
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

        // compute: RGBA(src) -> NV12 compute_target (storage)
        let to_read = img_barrier(
            import_img,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
        );
        let t_y = img_barrier(
            compute_target,
            vk::ImageAspectFlags::PLANE_0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
        );
        let t_uv = img_barrier(
            compute_target,
            vk::ImageAspectFlags::PLANE_1,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
        );
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

        if direct {
            // LINEAR direct: the compute output *is* the export image. Flush the shader
            // writes so the dmabuf consumer sees them (ordering to the consumer is the
            // implicit dma-buf write-fence / the CPU wait below).
            let f_y = img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_0,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::empty(),
            );
            let f_uv = img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::empty(),
            );
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[f_y, f_uv],
            );
        } else {
            // Tiled/DCC: copy NV12 scratch -> NV12 export slot (radv fills DCC metadata).
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
            );
            let o_uv = img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[s_y_src, s_uv_src, o_y, o_uv],
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

            if self.encode_src {
                // Hand the encoder its input already in VIDEO_ENCODE_SRC_KHR. The encode
                // queue read is ordered after this submit by the slot fence wait below
                // (same device as the encoder).
                let enc_layout = vk::ImageLayout::from_raw(VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR);
                let e_y = img_barrier(
                    out_img,
                    vk::ImageAspectFlags::PLANE_0,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    enc_layout,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::empty(),
                );
                let e_uv = img_barrier(
                    out_img,
                    vk::ImageAspectFlags::PLANE_1,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    enc_layout,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::empty(),
                );
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
        }

        self.device.end_command_buffer(cmd)?;
        let cbs = [cmd];
        let mut submit = vk::SubmitInfo::default().command_buffers(&cbs);
        let sems = [self.sync_sem];
        if self.implicit_sync {
            submit = submit.signal_semaphores(&sems);
        }
        self.device.queue_submit(self.queue, &[submit], fence)?;

        if self.implicit_sync {
            // Hand the export dmabuf's BO the submit's completion as a *write* fence, so
            // the VA encoder's import-read waits for it via the kernel's implicit sync --
            // no CPU stall on the producer's critical path (lever 1). The cached import
            // stays alive (it's reused next time this slot comes round).
            self.attach_write_fence(idx);
            self.outputs[idx].in_flight = true;
        } else {
            // Nvidia/CUDA ignores implicit dma-buf fences: block until the write is done
            // before the buffer is handed downstream.
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.reset_fences(&[fence])?;
        }

        // Debug: one-shot readback of the converter's NV12 output to a file. `compute_target`
        // (the LINEAR scratch on the encode-src path) holds exactly what the compute wrote,
        // before the tiled encode-src copy -- so a clean dump localises the green-bar/jump to
        // the tiled copy or the encoder. Off unless WOLF_VULKAN_DUMP is set; once, a few
        // frames in (let the scene settle). Failures are logged, never fatal.
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
                self.device.device_wait_idle().ok();
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
        Ok(())
    }

    /// One-shot debug readback of a multiplanar NV12/P010 image (must be in
    /// TRANSFER_SRC_OPTIMAL and have TRANSFER_SRC usage -- true for the compute scratch) into
    /// a host buffer, then write tight raw planar (`w*h*bpp` Y plane + `w*h/2*bpp` interleaved
    /// UV) to `path`. `bpp` is 1 for NV12, 2 for P010 (16-bit samples, value in the MSBs).
    unsafe fn dump_planar(&self, img: vk::Image, w: u32, h: u32, path: &str) -> Result<(), Err> {
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
        self.device.end_command_buffer(cmd)?;
        let fence = self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cbs = [cmd];
        self.device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::default().command_buffers(&cbs)],
            fence,
        )?;
        self.device.wait_for_fences(&[fence], true, u64::MAX)?;

        let ptr = self
            .device
            .map_memory(mem, 0, total, vk::MemoryMapFlags::empty())? as *const u8;
        let res = std::fs::write(path, std::slice::from_raw_parts(ptr, total as usize));
        self.device.unmap_memory(mem);

        self.device.destroy_fence(fence, None);
        self.device.free_command_buffers(self.cmd_pool, &[cmd]);
        self.device.destroy_buffer(buf, None);
        self.device.free_memory(mem, None);
        res?;
        Ok(())
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
            .try_clone_to_owned()?
            .into_raw_fd();
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
        let mr = self.device.get_image_memory_requirements(img);
        let ext_fd = ash::khr::external_memory_fd::Device::new(&self.instance, &self.device);
        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        ext_fd.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            src_fd,
            &mut fd_props,
        )?;
        let allowed = mr.memory_type_bits & fd_props.memory_type_bits;
        let mt = (0..self.memp.memory_type_count)
            .find(|&i| allowed & (1 << i) != 0)
            .ok_or("no import-compatible memory type")?;
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(src_fd);
        let mem = self.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(mr.size)
                .memory_type_index(mt)
                .push_next(&mut import_info),
            None,
        )?;
        self.device.bind_image_memory(img, mem, 0)?;
        let view = plane_view(&self.device, img, pd_format, vk::ImageAspectFlags::COLOR)?;
        Ok((img, mem, view))
    }

    /// Export the submit's signal semaphore as a SYNC_FD and import it onto slot `idx`'s
    /// export dmabuf as a write fence (best-effort: a failure just means the consumer
    /// falls back to whatever ordering it had, so we log and continue).
    unsafe fn attach_write_fence(&self, idx: usize) {
        let Some(loader) = self.sem_fd.as_ref() else {
            return;
        };
        let sync_fd = match loader.get_semaphore_fd(
            &vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(self.sync_sem)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
        ) {
            Ok(fd) if fd >= 0 => fd,
            Ok(_) => return, // -1 means already signalled; nothing to wait on
            Err(e) => {
                tracing::warn!("VulkanNv12: get_semaphore_fd failed: {e}");
                return;
            }
        };
        let arg = DmaBufImportSyncFile {
            flags: DMA_BUF_SYNC_WRITE,
            fd: sync_fd,
        };
        let rc = libc::ioctl(
            self.outputs[idx].export_fd,
            DMA_BUF_IOCTL_IMPORT_SYNC_FILE,
            &arg as *const _,
        );
        if rc != 0 {
            tracing::warn!(
                "VulkanNv12: IMPORT_SYNC_FILE failed (errno {})",
                std::io::Error::last_os_error()
            );
        }
        libc::close(sync_fd);
    }

    unsafe fn free_import(&mut self, idx: usize) {
        if let Some((_, (img, mem, view))) = self.outputs[idx].import.take() {
            self.device.destroy_image_view(view, None);
            self.device.destroy_image(img, None);
            self.device.free_memory(mem, None);
        }
    }

    /// The just-converted NV12 export slot as a gst buffer. Returns the slot's cached
    /// buffer (ref-counted) so the VA encoder reuses one stable surface per slot.
    ///
    /// On the `encode_src` path the returned buffer's PTS is `NONE` in normal steady state:
    /// `convert_inner` resets the recycled slot's timing metadata right after the reuse gate,
    /// so BaseSrc's `do_timestamp` re-stamps it with the current running-time each frame
    /// (matching the RGBx/DMABuf paths, which build a fresh `GstBuffer` every frame). On the
    /// gate's 1s-timeout escape the reset is skipped (buffer still shared) and the PTS may be
    /// stale; `convert_inner` warns when that happens.
    pub fn to_gst_buffer(&self) -> Result<GstBuffer, Err> {
        Ok(self.outputs[self.cur].buffer.clone())
    }
}

impl Drop for VulkanNv12 {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for idx in 0..self.outputs.len() {
                self.free_import(idx);
            }
            for o in self.outputs.drain(..) {
                self.device.destroy_image_view(o.y_view, None);
                self.device.destroy_image_view(o.uv_view, None);
                if let Some((s_img, s_mem)) = o.scratch {
                    self.device.destroy_image(s_img, None);
                    self.device.free_memory(s_mem, None);
                }
                // On the encode-src path the output image + memory belong to the gst pool
                // (freed when the buffer/pool drop) and there is no export fd to close.
                if !self.encode_src {
                    self.device.destroy_image(o.image, None);
                    self.device.free_memory(o.mem, None);
                    libc::close(o.export_fd);
                }
                self.device.destroy_fence(o.fence, None);
            }
            if self.sync_sem != vk::Semaphore::null() {
                self.device.destroy_semaphore(self.sync_sem, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            if let Some(p) = self.pipeline_pq {
                self.device.destroy_pipeline(p, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            // On the shared/encode-src path the VkDevice + VkInstance are owned by the
            // downstream encoder's GstVulkanDevice (held alive via `_shared_device`); the
            // encoder still frees its own DPB image views at teardown, so destroying them
            // here would yank the device out from under it. Only tear down what we created.
            if !self.encode_src {
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }
}

// --- helpers ---

/// SDR diffuse white (nits) for the BT.2020/PQ tone-map: Wolf's `[gstreamer.video]
/// sdr_reference_white` passed as `WOLF_SDR_REFERENCE_WHITE`, default 203 (BT.2408 graphics
/// white). Bound to specialization constant 0 of every converter pipeline (ignored by shaders
/// that don't declare it).
fn sdr_reference_white() -> f32 {
    std::env::var("WOLF_SDR_REFERENCE_WHITE")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(203.0)
}

/// `WOLF_HDR_CM`: dynamic-colorimetry HDR mode. When set, the fp16 P010 converter builds BOTH
/// the SDR (`RGBA_TO_P010_SPV`, BT.709 matrix, no PQ) and PQ-passthrough
/// (`RGBA_PQPASS_TO_P010_SPV`) pipelines up front -- independent of the negotiated caps
/// colorimetry -- and `convert()` picks per frame by `pq_passthrough`, so a producer's
/// mid-stream bt709<->bt2100-pq caps flip never rebuilds the converter (which would stall the
/// compositor thread). Read once; unset == byte-identical prior behavior.
fn wolf_hdr_cm() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("WOLF_HDR_CM").is_ok())
}

/// The "normal" (`pq_passthrough = false`) converter shader. Under `WOLF_HDR_CM` on the fp16
/// P010 path this is the BT.709-matrix `RGBA_TO_P010_SPV` (true SDR, no PQ) regardless of the
/// negotiated caps `bt2020` flag, so the pipeline survives a mid-stream bt709<->bt2100-pq
/// colorimetry flip without a rebuild; the per-frame PQ-passthrough pipeline handles already-PQ
/// HDR content. With `WOLF_HDR_CM` unset this is exactly `PixFmt::shader` (caps-driven).
fn normal_shader(fmt: PixFmt, bt2020: bool, fp16_input: bool) -> &'static [u8] {
    if wolf_hdr_cm() && fmt == PixFmt::P010 && fp16_input {
        RGBA_TO_P010_SPV
    } else {
        fmt.shader(bt2020, fp16_input)
    }
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

/// Build the per-frame PQ-passthrough pipeline, or `None` when it isn't applicable. Needed on the
/// HDR fp16 P010 path, where a frame from a 10-bit already-PQ client buffer must skip the
/// tone-map. The static HDR path builds it when the caps are `bt2020`; under `WOLF_HDR_CM` it's
/// built for ANY fp16 P010 negotiation (independent of the caps `bt2020` flag) so it survives a
/// bt709-tagged negotiation / a mid-stream colorimetry flip. Best-effort: a build failure (e.g.
/// the placeholder `.spv` hasn't been compiled with glslc yet) leaves it `None` so the normal
/// pipeline still runs -- byte-identical to the prior behavior.
unsafe fn build_pq_passthrough(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    fmt: PixFmt,
    bt2020: bool,
    fp16_input: bool,
    sdr_ref_white: f32,
) -> Option<vk::Pipeline> {
    let want = fmt == PixFmt::P010 && fp16_input && (bt2020 || wolf_hdr_cm());
    if !want {
        return None;
    }
    match build_compute_pipeline(device, layout, RGBA_PQPASS_TO_P010_SPV, sdr_ref_white) {
        Ok(p) => {
            tracing::debug!("VulkanNv12: PQ-passthrough pipeline ready (per-frame 10-bit input)");
            Some(p)
        }
        Err(e) => {
            tracing::warn!(
                "VulkanNv12: PQ-passthrough pipeline unavailable ({e}); 10-bit frames will use \
                 the tone-mapping shader (double-PQ). Compile shaders/rgba_pqpass_to_p010.comp."
            );
            None
        }
    }
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
    let mem = device.allocate_memory(&alloc, None)?;
    device.bind_image_memory(image, mem, 0)?;
    let y_view = plane_view(
        device,
        image,
        fmt.y_view_format(),
        vk::ImageAspectFlags::PLANE_0,
    )?;
    let uv_view = plane_view(
        device,
        image,
        fmt.uv_view_format(),
        vk::ImageAspectFlags::PLANE_1,
    )?;
    Ok((image, mem, y_view, uv_view))
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
) -> Result<Nv12Out, Err> {
    // The export image: direct (LINEAR, compute-writable) carries STORAGE; tiled is a
    // pure TRANSFER_DST target filled by vkCmdCopyImage.
    let (image, mem, y_view, uv_view, scratch) = if direct {
        let (image, mem, y_view, uv_view) = create_storage(
            device,
            memp,
            width,
            height,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            true,
            fmt,
        )?;
        (image, mem, y_view, uv_view, None)
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
        let mr = device.get_image_memory_requirements(image);
        let mut exp = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mem = device.allocate_memory(
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
        device.bind_image_memory(image, mem, 0)?;
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
        (image, mem, s_y, s_uv, Some((s_img, s_mem)))
    };

    let ext_fd = ash::khr::external_memory_fd::Device::new(instance, device);
    let fd = ext_fd.get_memory_fd(
        &vk::MemoryGetFdInfoKHR::default()
            .memory(mem)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
    )?;
    // Keep our own dup so we can attach a per-frame write fence even after `fd` is owned
    // by the gst buffer (the sync_file attaches to the shared underlying BO/dma_resv).
    let export_fd = libc::dup(fd);
    if export_fd < 0 {
        return Err("dup(export dmabuf fd) failed".into());
    }
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
        width,
        height,
        fourcc: fmt.fourcc(),
        modifier,
        y_offset: l0.offset as usize,
        y_stride: l0.row_pitch as i32,
        uv_offset: l1.offset as usize,
        uv_stride: l1.row_pitch as i32,
    };
    // VA surface sharing is the AMD/Intel NV12 dmabuf-encode optimisation; the P010 path
    // targets the Vulkan encoder, so only attempt the VA-allocator buffer for NV12.
    let shared = if fmt == PixFmt::Nv12 {
        crate::utils::va_share::build_shared_buffer(fd, mr.size as usize, &layout)
    } else {
        None
    };
    let buffer = match shared {
        Some(b) => b,
        None => {
            let mut buffer = GstBuffer::new();
            {
                let b = buffer.get_mut().unwrap();
                let gmem = allocator.alloc_dmabuf_with_flags(
                    fd,
                    mr.size as usize,
                    FdMemoryFlags::DONT_CLOSE,
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
    let cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .command_buffer_count(1),
    )?[0];
    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
    let dsls = [desc_layout];
    let desc_set = device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&dsls),
    )?[0];
    let y_info = [image_info(y_view)];
    let uv_info = [image_info(uv_view)];
    device.update_descriptor_sets(
        &[
            write_img(desc_set, 1, vk::DescriptorType::STORAGE_IMAGE, &y_info),
            write_img(desc_set, 2, vk::DescriptorType::STORAGE_IMAGE, &uv_info),
        ],
        &[],
    );

    Ok(Nv12Out {
        image,
        mem,
        buffer,
        export_fd,
        scratch,
        y_view,
        uv_view,
        cmd,
        fence,
        desc_set,
        import: None,
        in_flight: false,
    })
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
    profile: &str,
    desc_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    cmd_pool: vk::CommandPool,
    width: u32,
    height: u32,
    fmt: PixFmt,
) -> Result<Nv12Out, Err> {
    let buffer = crate::utils::vulkan_share::alloc_encode_src_buffer(
        gst_device, width, height, profile, fmt,
    )
    .ok_or("encode-src image allocation failed")?;
    let out_img = crate::utils::vulkan_share::recover_vk_image(&buffer)
        .ok_or("encode-src buffer is not a single GstVulkanImageMemory")?;

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

    // Diagnostic for the RX 9070 green-bar/jump report: if the encoder's pool gives us an
    // encode-src image padded beyond width*height*3/2 (e.g. RDNA4 row-alignment), our copy
    // fills only `height` rows and the padding stays zeroed -> green at the bottom.
    let tight_nv12 = width as u64 * height as u64 * 3 / 2;
    tracing::debug!(
        "encode-src slot: req={width}x{height} NV12; out_img mem={} scratch mem={} tight={tight_nv12}",
        device.get_image_memory_requirements(out_img).size,
        device.get_image_memory_requirements(s_img).size,
    );

    let cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .command_buffer_count(1),
    )?[0];
    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
    let dsls = [desc_layout];
    let desc_set = device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&dsls),
    )?[0];
    let y_info = [image_info(y_view)];
    let uv_info = [image_info(uv_view)];
    device.update_descriptor_sets(
        &[
            write_img(desc_set, 1, vk::DescriptorType::STORAGE_IMAGE, &y_info),
            write_img(desc_set, 2, vk::DescriptorType::STORAGE_IMAGE, &uv_info),
        ],
        &[],
    );

    Ok(Nv12Out {
        image: out_img,
        mem: vk::DeviceMemory::null(), // the pool owns the encode-src image memory
        buffer,
        export_fd: -1,
        scratch: Some((s_img, s_mem)),
        y_view,
        uv_view,
        cmd,
        fence,
        desc_set,
        import: None,
        in_flight: false,
    })
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
