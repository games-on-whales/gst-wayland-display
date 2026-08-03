//! Share the downstream encoder's `GstVulkanDevice` with our producer, and mint
//! encode-ready NV12 `GstVulkanImageMemory` buffers on it.
//!
//! For the Vulkan-encode path (`waylanddisplaysrc ! interpipesink` ⇒ `interpipesrc !
//! vulkanh264enc`, both in one process) the producer must output NV12 images on the *same*
//! `GstVulkanDevice` the encoder uses, created as a *single multiplanar*
//! `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM` image with `VIDEO_ENCODE_SRC` usage and the
//! encoder's video profile chained in — then `vulkanh264enc` image-views it with no copy.
//!
//! The shared device arrives via `GstContext` (`gst.vulkan.device`), exactly like the
//! CUDA path's context sharing in Wolf. This module extracts + stashes the shared device
//! (set from `ElementImpl::set_context`), exposes the raw `VkInstance`/`VkPhysicalDevice`/
//! `VkDevice` + a graphics queue family so our `ash` compute/copy runs on the encoder's
//! device, and builds a `GstVulkanImageBufferPool` of encode-src NV12 images whose
//! `VkImage` we recover.
//!
//! `gstreamer-vulkan` (safe) leaves the Vulkan-typed calls unbound (gir skips vk types), so
//! we call them through `gstreamer-vulkan-sys` (whose `vulkan::*` are ash `vk::*`
//! re-exports) and read the two public struct fields we need (`VkDevice`, `VkImage`) via
//! `repr(C)` overlays anchored on `gst::ffi::{GstObject, GstMemory}` (correct ABI prefix).

#![allow(unsafe_op_in_unsafe_fn)]

use crate::utils::vulkan_nv12::PixFmt;
use ash::vk;
use gst::glib::translate::{ToGlibPtr, from_glib_full};
use gst::prelude::*;
use gstreamer_vulkan::prelude::*;
use gstreamer_vulkan::{VulkanDevice, VulkanInstance, VulkanPhysicalDevice};
use gstreamer_vulkan_sys as gstvk;
use std::os::raw::c_void;
use std::sync::{Mutex, OnceLock};

// VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR / VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR via raw
// values (stable, and avoids depending on the named ash constants existing).
const VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_KHR: u32 = 0x0000_2000;
const VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR: i32 = 1_000_299_001;

// --- repr(C) overlays of the public gst-vulkan structs (fields the -sys crate omits). ---

/// `struct _GstVulkanDevice { GstObject parent; GstVulkanInstance *instance;
/// GstVulkanPhysicalDevice *physical_device; VkDevice device; ... }`
#[repr(C)]
struct GstVulkanDeviceOverlay {
    parent: gst::ffi::GstObject,
    instance: *mut c_void,        // GstVulkanInstance*
    physical_device: *mut c_void, // GstVulkanPhysicalDevice*
    device: vk::Device,           // ABI: a dispatchable handle (pointer)
}

/// `struct _GstVulkanInstance { GstObject parent; VkInstance instance; ... }`
#[repr(C)]
struct GstVulkanInstanceOverlay {
    parent: gst::ffi::GstObject,
    instance: vk::Instance,
}

/// `struct _GstVulkanQueue { GstObject parent; GstVulkanDevice *device; VkQueue queue;
/// guint32 family; guint32 index; }`
#[repr(C)]
struct GstVulkanQueueOverlay {
    parent: gst::ffi::GstObject,
    device: *mut c_void,
    queue: vk::Queue,
    family: u32,
    index: u32,
}

/// `struct _GstVulkanImageMemory { GstMemory parent; GstVulkanDevice *device;
/// VkImage image; ... }`
#[repr(C)]
struct GstVulkanImageMemoryOverlay {
    parent: gst::ffi::GstMemory,
    device: *mut c_void,
    image: vk::Image,
}

/// Raw Vulkan handles + queue family pulled out of the shared `GstVulkanDevice`, ready to
/// drive an `ash`-loaded device.
#[derive(Clone, Copy, Debug)]
pub struct RawVk {
    pub instance: vk::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: vk::Device,
    pub gfx_queue_family: u32,
}

/// Process-wide slot for the shared device, filled from `set_context` and read by the
/// converter when it builds its output ring (mirrors `va_share`'s display slot).
fn device_slot() -> &'static Mutex<Option<VulkanDevice>> {
    static SLOT: OnceLock<Mutex<Option<VulkanDevice>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Pull the shared `GstVulkanDevice` out of a received `GstContext` and stash it. Call from
/// `ElementImpl::set_context` for the `gst.vulkan.device` context type. Returns true if a
/// device is now shared.
pub fn handle_set_context(context: &gst::Context) -> bool {
    let ctx_ptr = context.to_glib_none().0;
    let mut dev_ptr: *mut gstvk::GstVulkanDevice = std::ptr::null_mut();
    let got = unsafe { gstvk::gst_context_get_vulkan_device(ctx_ptr, &mut dev_ptr) };
    if got == gst::glib::ffi::GFALSE || dev_ptr.is_null() {
        return false;
    }
    let device: VulkanDevice = unsafe { from_glib_full(dev_ptr) };
    tracing::debug!("vulkan_share: absorbed shared GstVulkanDevice {dev_ptr:?}");
    *device_slot().lock().unwrap() = Some(device);
    true
}

/// The shared device, if one has been created (or absorbed).
pub fn shared_device() -> Option<VulkanDevice> {
    device_slot().lock().unwrap().clone()
}

/// Process-wide slot for the `GstVulkanInstance` we own (keeps it alive + lets us answer
/// `gst.vulkan.instance` context queries).
fn instance_slot() -> &'static Mutex<Option<VulkanInstance>> {
    static SLOT: OnceLock<Mutex<Option<VulkanInstance>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

// VK_QUEUE_VIDEO_ENCODE_BIT_KHR — the encoder (gst_vulkan_encoder_create_from_queue) requires a
// queue family with this bit, and gst_vulkan_device_choose_queues only opens an encode queue if the
// chosen physical device has one. Software devices (llvmpipe) do NOT, so a device without this bit
// is useless for the encode path and makes vulkanh264enc fail to link.
const VK_QUEUE_VIDEO_ENCODE_BIT_KHR: vk::QueueFlags = vk::QueueFlags::from_raw(0x0000_0040);

/// Whether `dev` exposes a video-encode queue family (i.e. it can host `vulkanh264enc`).
unsafe fn has_video_encode_queue(ash_inst: &ash::Instance, dev: vk::PhysicalDevice) -> bool {
    ash_inst
        .get_physical_device_queue_family_properties(dev)
        .iter()
        .any(|q| q.queue_flags.contains(VK_QUEUE_VIDEO_ENCODE_BIT_KHR))
}

unsafe fn device_name(ash_inst: &ash::Instance, dev: vk::PhysicalDevice) -> String {
    let mut p2 = vk::PhysicalDeviceProperties2::default();
    ash_inst.get_physical_device_properties2(dev, &mut p2);
    let bytes = &p2.properties.device_name;
    let len = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
    let slice = std::slice::from_raw_parts(bytes.as_ptr() as *const u8, len);
    String::from_utf8_lossy(slice).into_owned()
}

/// Pick the physical device to back our shared encode device. We must NEVER hand the encoder a
/// device that can't video-encode (e.g. llvmpipe), so encode capability is a hard requirement:
/// prefer the encode-capable GPU whose DRM render/primary minor matches the compositor's render
/// node (`target_minor`); otherwise the first encode-capable GPU. Only if none can encode do we
/// fall back to index 0. (The old code fell back to a bare index 0, which — since Vulkan
/// enumeration order isn't guaranteed — could land on llvmpipe and make `vulkanh264enc` fail to
/// link, intermittently across restarts.)
unsafe fn physical_index_for_minor(
    ash_inst: &ash::Instance,
    target_minor: Option<u32>,
) -> Option<u32> {
    let devices = ash_inst.enumerate_physical_devices().ok()?;
    if devices.is_empty() {
        return None;
    }
    let encode_capable: Vec<usize> = (0..devices.len())
        .filter(|&i| has_video_encode_queue(ash_inst, devices[i]))
        .collect();

    // Prefer an encode-capable device matching the compositor's render minor.
    if let Some(minor) = target_minor {
        for &i in &encode_capable {
            let d = devices[i];
            let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
            ash_inst.get_physical_device_properties2(d, &mut p2);
            if (drm.has_render != 0 && drm.render_minor as i64 == minor as i64)
                || (drm.has_primary != 0 && drm.primary_minor as i64 == minor as i64)
            {
                tracing::info!(
                    "vulkan_share: selected physical device {i} '{}' (matches render minor {minor}, video-encode capable)",
                    device_name(ash_inst, d)
                );
                return Some(i as u32);
            }
        }
        tracing::warn!(
            "vulkan_share: no encode-capable physical device matched render minor {minor}; \
             falling back to first encode-capable GPU"
        );
    }
    if let Some(&i) = encode_capable.first() {
        tracing::info!(
            "vulkan_share: selected physical device {i} '{}' (first video-encode capable)",
            device_name(ash_inst, devices[i])
        );
        return Some(i as u32);
    }
    tracing::warn!(
        "vulkan_share: NO video-encode-capable physical device found; using index 0 (encode will likely fail)"
    );
    Some(0)
}

/// Create (once) the `GstVulkanInstance` + `GstVulkanDevice` that *we* own, on the GPU
/// backing `target_minor`, with the external-memory extensions the RGBA-dmabuf import needs
/// (`VK_KHR_external_memory_fd` etc.) enabled — which gst-vulkan's own device does not. This
/// is the device we hand the encoder (see [`provide_context`]) so producer and encoder share
/// one device with no zero-copy gap *and* no gstreamer fork. Idempotent.
pub fn ensure_owned_device(target_minor: Option<u32>) -> Option<VulkanDevice> {
    let mut slot = device_slot().lock().unwrap();
    if let Some(d) = slot.clone() {
        return Some(d);
    }
    let instance = VulkanInstance::new();
    if let Err(e) = instance.open() {
        tracing::error!("vulkan_share: GstVulkanInstance open failed: {e}");
        return None;
    }
    let vk_instance = unsafe { (*(instance.as_ptr() as *const GstVulkanInstanceOverlay)).instance };
    let index = unsafe {
        let entry = match ash::Entry::load() {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("vulkan_share: ash entry load failed: {e}");
                return None;
            }
        };
        let ash_inst = ash::Instance::load(entry.static_fn(), vk_instance);
        physical_index_for_minor(&ash_inst, target_minor)?
    };
    let physical = VulkanPhysicalDevice::new(&instance, index);
    let device = VulkanDevice::new(&physical);
    for ext in [
        "VK_KHR_external_memory_fd",
        "VK_EXT_external_memory_dma_buf",
        "VK_EXT_image_drm_format_modifier",
        "VK_KHR_external_semaphore_fd",
    ] {
        device.enable_extension(ext);
    }
    if let Err(e) = device.open() {
        tracing::error!("vulkan_share: GstVulkanDevice open failed: {e}");
        return None;
    }
    *instance_slot().lock().unwrap() = Some(instance);
    *slot = Some(device.clone());
    tracing::info!(
        "vulkan_share: created shared GstVulkanDevice (phys idx {index}) with external-memory extensions"
    );
    Some(device)
}

/// Answer a `gst.vulkan.{instance,device}` context query on `element` with the device we own,
/// creating it on `target_minor`'s GPU on first ask. The downstream encoder's
/// `gst_vulkan_ensure_element_data` then adopts *our* device instead of minting its own.
pub fn provide_context(
    element: &gst::Element,
    query: &mut gst::QueryRef,
    target_minor: Option<u32>,
) -> bool {
    if ensure_owned_device(target_minor).is_none() {
        return false;
    }
    let inst = instance_slot().lock().unwrap().clone();
    let dev = device_slot().lock().unwrap().clone();
    unsafe {
        gstvk::gst_vulkan_handle_context_query(
            element.as_ptr() as *mut _,
            query.as_mut_ptr() as *mut _,
            std::ptr::null_mut(),
            inst.as_ref().map_or(std::ptr::null_mut(), |i| i.as_ptr()) as *mut _,
            dev.as_ref().map_or(std::ptr::null_mut(), |d| d.as_ptr()) as *mut _,
        ) != gst::glib::ffi::GFALSE
    }
}

/// Wait up to `timeout` for our shared `GstVulkanDevice` to be available.
///
/// The encoder shares its device via a `GstContext` delivered to `set_context` on the
/// streaming thread, which races the compositor thread that allocates our Vulkan output
/// buffer. Polling here lets the allocation wait for the device to arrive instead of
/// failing when it merely hasn't been shared *yet*. Returns `None` if it never arrives.
pub fn wait_for_shared_device(timeout: std::time::Duration) -> Option<VulkanDevice> {
    let start = std::time::Instant::now();
    loop {
        if let Some(dev) = shared_device() {
            return Some(dev);
        }
        if start.elapsed() >= timeout {
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Extract the raw `VkInstance`/`VkPhysicalDevice`/`VkDevice` + a graphics-capable queue
/// family from the (shared) `GstVulkanDevice`.
pub fn raw_handles(device: &VulkanDevice) -> Option<RawVk> {
    let dev_ptr: *mut gstvk::GstVulkanDevice = device.to_glib_none().0;
    if dev_ptr.is_null() {
        return None;
    }
    unsafe {
        let overlay = &*(dev_ptr as *const GstVulkanDeviceOverlay);
        if overlay.device == vk::Device::null() || overlay.instance.is_null() {
            return None;
        }
        let vk_instance = (*(overlay.instance as *const GstVulkanInstanceOverlay)).instance;
        let vk_physical = gstvk::gst_vulkan_device_get_physical_device(dev_ptr);
        if vk_physical == vk::PhysicalDevice::null() {
            return None;
        }
        // A queue that can run our compute + copy (the encoder owns its own encode queue).
        let queue =
            gstvk::gst_vulkan_device_select_queue(dev_ptr, vk::QueueFlags::GRAPHICS.as_raw());
        let gfx_queue_family = if queue.is_null() {
            0
        } else {
            (*(queue as *const GstVulkanQueueOverlay)).family
        };
        Some(RawVk {
            instance: vk_instance,
            physical: vk_physical,
            device: overlay.device,
            gfx_queue_family,
        })
    }
}

/// Build the H.264 profile caps the encode-src image pool needs (matches what
/// `vulkanh264enc`'s `propose_allocation` feeds its pool). NV12 ⇒ 4:2:0, 8-bit.
pub fn h264_profile_caps(profile: &str) -> gst::Caps {
    gst::Caps::builder("video/x-h264")
        .field("profile", profile)
        .field("chroma-format", "4:2:0")
        .field("bit-depth-luma", 8u32)
        .field("bit-depth-chroma", 8u32)
        .build()
}

/// Create + activate a `GstVulkanImageBufferPool` of encode-src NV12 images on the shared
/// device, configured exactly like the encoder's own input pool (usage
/// `VIDEO_ENCODE_SRC | TRANSFER_DST`, the encode profile chained in), so a buffer acquired
/// from it is a single multiplanar `GstVulkanImageMemory` the encoder views zero-copy.
pub fn encode_src_pool(
    device: &VulkanDevice,
    nv12_caps: &gst::Caps,
    profile_caps: &gst::Caps,
    size: usize,
    min_buffers: u32,
    max_buffers: u32,
) -> Option<gst::BufferPool> {
    unsafe {
        let pool_ptr = gstvk::gst_vulkan_image_buffer_pool_new(device.to_glib_none().0);
        if pool_ptr.is_null() {
            return None;
        }
        // Build the config entirely via the raw GstStructure that set_config consumes, so
        // our Vulkan-typed allocation params (the VIDEO_ENCODE_SRC usage that makes the pool
        // pick the multiplanar NV12 format) actually land -- mixing the safe BufferPoolConfig
        // wrapper with raw-ptr FFI dropped them.
        let pool_bp = pool_ptr as *mut gst::ffi::GstBufferPool;
        let cfg = gst::ffi::gst_buffer_pool_get_config(pool_bp);
        gst::ffi::gst_buffer_pool_config_set_params(
            cfg,
            nv12_caps.to_glib_none().0,
            size as u32,
            min_buffers,
            max_buffers,
        );
        let usage = vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::from_raw(VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_KHR);
        let access =
            (vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE).as_raw() as u64;
        gstvk::gst_vulkan_image_buffer_pool_config_set_allocation_params(
            cfg,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::ImageLayout::from_raw(VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR),
            access,
        );
        gstvk::gst_vulkan_image_buffer_pool_config_set_encode_caps(
            cfg,
            profile_caps.to_glib_none().0,
        );
        let ok = gst::ffi::gst_buffer_pool_set_config(pool_bp, cfg); // transfer-full of cfg
        let pool: gst::BufferPool = from_glib_full(pool_ptr);
        if ok == gst::glib::ffi::GFALSE {
            tracing::warn!(
                "vulkan_share: encode-src pool set_config failed (caps={nv12_caps}, profile={profile_caps})"
            );
            return None;
        }
        if pool.set_active(true).is_err() {
            tracing::warn!("vulkan_share: failed to activate encode-src pool");
            return None;
        }
        Some(pool)
    }
}

/// Allocate ONE encode-src NV12 image as a `GstVulkanImageMemory`, built directly via
/// `gst_vulkan_image_memory_alloc_with_image_info` so we control the `VkImageCreateInfo`
/// (usage `TRANSFER_DST | VIDEO_ENCODE_SRC` + the H.264 `VkVideoProfileListInfoKHR` chained
/// in). This bypasses `GstVulkanImageBufferPool`'s generic-format-feature check, which
/// rejects NV12 on NVIDIA because the encode-input feature is only reported via the
/// profile-specific query. Returns a `GstBuffer` holding the single image memory + VideoMeta.
pub fn alloc_encode_src_buffer(
    device: &VulkanDevice,
    width: u32,
    height: u32,
    profile: &str,
    fmt: PixFmt,
) -> Option<gst::Buffer> {
    let std_h264_idc = match profile {
        "high" | "constrained-high" | "progressive-high" => {
            vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH
        }
        "main" => vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
        _ => vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
    };
    unsafe {
        // Profile chain (matches what the encoder's pool builds from caps): the codec struct
        // chained off the VkVideoProfileInfoKHR; usage info omitted. P010 -> H.265 Main-10
        // 10-bit (for vulkanh265enc); NV12 -> H.264 8-bit (the default Vulkan-encode path).
        let mut h264 = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(std_h264_idc);
        let mut h265 = vk::VideoEncodeH265ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10);
        let bit_depth = match fmt {
            PixFmt::P010 => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
            PixFmt::Nv12 => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        };
        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(bit_depth)
            .chroma_bit_depth(bit_depth);
        profile_info = match fmt {
            PixFmt::P010 => profile_info
                .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
                .push_next(&mut h265),
            PixFmt::Nv12 => profile_info
                .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
                .push_next(&mut h264),
        };
        let profiles = [profile_info];
        let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

        // RX 9070 / GFX12 (RDNA4) workaround: radv mishandles a LINEAR->tiled (different
        // swizzle mode) vkCmdCopyImage on GFX12 (cf. Mesa 26.0.2 "radv: fix copying images
        // with different swizzle modes on SDMA7"), corrupting the encoder's tiled NV12 input
        // (green band / shifted image). Allocating the encode-src LINEAR makes the
        // scratch->encode-src copy LINEAR->LINEAR (no swizzle change), avoiding that path --
        // *if* the GFX12 VCN encoder accepts a linear input image. Opt-in (the tiled default
        // works on RDNA3 and on a fixed radv); falls back to tiled if the LINEAR *allocation*
        // fails (a driver that accepts the LINEAR image but corrupts or fails at encode time
        // is not covered by the fallback).
        let linear_encsrc = std::env::var("WOLF_VULKAN_LINEAR_ENCSRC")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let tiling = if linear_encsrc {
            vk::ImageTiling::LINEAR
        } else {
            vk::ImageTiling::OPTIMAL
        };
        let image_format = match fmt {
            PixFmt::Nv12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
            PixFmt::P010 => vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
        };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(image_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .usage(
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::from_raw(VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_KHR),
            )
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut profile_list);

        let mut mem_ptr = gstvk::gst_vulkan_image_memory_alloc_with_image_info(
            device.to_glib_none().0,
            &image_info as *const vk::ImageCreateInfo as *mut _,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let mut effective_tiling = tiling;
        if mem_ptr.is_null() && linear_encsrc {
            // The tiled-fallback promised above: some encoders (NVIDIA Vulkan-Video)
            // reject a LINEAR encode-src image outright, so a failed LINEAR alloc must
            // not kill the whole vulkan output -- retry with the tiled default.
            tracing::warn!(
                "vulkan_share: LINEAR encode-src alloc failed (WOLF_VULKAN_LINEAR_ENCSRC) -- \
                 falling back to tiled (OPTIMAL)"
            );
            let image_info = image_info.tiling(vk::ImageTiling::OPTIMAL);
            effective_tiling = vk::ImageTiling::OPTIMAL;
            mem_ptr = gstvk::gst_vulkan_image_memory_alloc_with_image_info(
                device.to_glib_none().0,
                &image_info as *const vk::ImageCreateInfo as *mut _,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
        }
        if mem_ptr.is_null() {
            tracing::warn!("vulkan_share: gst_vulkan_image_memory_alloc_with_image_info failed");
            return None;
        }
        if linear_encsrc {
            tracing::info!(
                "vulkan_share: encode-src allocated with tiling={:?} (WOLF_VULKAN_LINEAR_ENCSRC)",
                effective_tiling
            );
        }

        // Our converter always leaves this image in VIDEO_ENCODE_SRC layout (it CPU-waits the
        // copy fence before handing the buffer downstream). Seed gst's *tracked* layout to match.
        // Otherwise `gst_vulkan_operation_add_frame_barrier` reads the stale alloc-time layout
        // (UNDEFINED) and records an UNDEFINED->VIDEO_ENCODE_SRC layout *write* transition for the
        // input image on every encode. With one encoder that's merely wasteful; with the interpipe
        // fan-out (one buffer -> N encoders) two such concurrent layout writes on the shared image,
        // on the video-encode queue with no semaphore between them, deadlock the GPU at frame 2.
        // With the tracked layout already SRC, each encoder's barrier is old==new == a pure read
        // barrier (VIDEO_ENCODE_READ), which is safe to issue concurrently from multiple encoders.
        //
        // `barrier.image_layout` offset in GstVulkanImageMemory (public struct, ABI-stable since
        // 1.18); verified against the running gst headers (sizeof=472, image@120, barrier@288,
        // barrier.image_layout@368).
        // GstVulkanImageMemory.barrier field offsets, verified against the running gst headers
        // (sizeof=472, image@120, barrier@288) and cross-checked against the gst 1.28.4 tag
        // (deploy target): _GstVulkanImageMemory and _GstVulkanBarrierMemoryInfo have identical
        // field order in 1.28.4 and 1.29.1, so these offsets hold for both. The struct is public
        // and ABI-stable since 1.18; re-run the /tmp/off.c offsetof probe if targeting a newer gst.
        const BARRIER_QUEUE_OFFSET: usize = 296; // barrier.parent.queue (GstVulkanQueue*)
        const BARRIER_SEMAPHORE_OFFSET: usize = 320; // barrier.parent.semaphore (VkSemaphore)
        const BARRIER_SEMAPHORE_VALUE_OFFSET: usize = 328; // barrier.parent.semaphore_value
        const BARRIER_IMAGE_LAYOUT_OFFSET: usize = 368; // barrier.image_layout (VkImageLayout)
        let base = mem_ptr as *mut u8;
        *(base.add(BARRIER_IMAGE_LAYOUT_OFFSET) as *mut i32) = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
        // Null the per-memory cross-queue dependency so gst's encoder adds no timeline-semaphore
        // wait/signal nor queue-ownership transfer against our image. gst's normal flow uses this
        // per-memory timeline semaphore to order one consumer after the previous producer op; but
        // (a) our converter already CPU-waits its write fence before handing the buffer downstream,
        // so the write is GPU-complete, and (b) the encoders only READ the image. With the interpipe
        // fan-out, leaving the semaphore set makes two encoders race the shared timeline value
        // (each does add_dependency=read-value, encode=signal value+1, end=increment) -> two
        // submissions signal the same value -> timeline corruption -> GPU hang. Read-only access by
        // N encoders needs no cross-dependency, so clear it.
        *(base.add(BARRIER_QUEUE_OFFSET) as *mut usize) = 0;
        *(base.add(BARRIER_SEMAPHORE_OFFSET) as *mut u64) = 0;
        *(base.add(BARRIER_SEMAPHORE_VALUE_OFFSET) as *mut u64) = 0;

        let mem: gst::Memory = from_glib_full(mem_ptr);
        let mut buffer = gst::Buffer::new();
        {
            let b = buffer.get_mut().unwrap();
            b.append_memory(mem);
            let _ = gst_video::VideoMeta::add(
                b,
                gst_video::VideoFrameFlags::empty(),
                match fmt {
                    PixFmt::Nv12 => gst_video::VideoFormat::Nv12,
                    PixFmt::P010 => gst_video::VideoFormat::P01010le,
                },
                width,
                height,
            );
        }
        Some(buffer)
    }
}

/// Recover the `VkImage` of a buffer's single `GstVulkanImageMemory` (the encode-src image
/// to run our compute/copy into). Returns `None` if the buffer isn't a single vulkan image.
pub fn recover_vk_image(buffer: &gst::Buffer) -> Option<vk::Image> {
    if buffer.n_memory() != 1 {
        return None;
    }
    let mem_ptr = buffer.peek_memory(0).as_ptr() as *mut gst::ffi::GstMemory;
    unsafe {
        if gstvk::gst_is_vulkan_image_memory(mem_ptr) == gst::glib::ffi::GFALSE {
            return None;
        }
        Some((*(mem_ptr as *const GstVulkanImageMemoryOverlay)).image)
    }
}
