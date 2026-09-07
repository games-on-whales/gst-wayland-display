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
//! device, and mints encode-src NV12 images one at a time via
//! `gst_vulkan_image_memory_alloc_with_image_info`, whose `VkImage` we recover.
//!
//! `gstreamer-vulkan` (safe) leaves the Vulkan-typed calls unbound (gir skips vk types), so
//! we call them through `gstreamer-vulkan-sys` (whose `vulkan::*` are ash `vk::*`
//! re-exports), and read the handles we need (`VkInstance`, `VkDevice`, `VkImage`, the
//! queue family) through `utils/vulkan_bridge.c`, a small C shim compiled by `build.rs`
//! against the target's own GStreamer Vulkan headers. Field access is therefore checked by
//! the C compiler and tracks the headers, instead of being guessed at hand-computed offsets.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::utils::vulkan_nv12::PixFmt;
use ash::vk;
use gst::glib::translate::{ToGlibPtr, from_glib_full};
use gst::prelude::*;
use gstreamer_vulkan::prelude::*;
use gstreamer_vulkan::{VulkanDevice, VulkanInstance, VulkanPhysicalDevice, VulkanQueue};
use gstreamer_vulkan_sys as gstvk;
use std::sync::{Arc, Mutex};

// VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR / VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR via raw
// values (stable, and avoids depending on the named ash constants existing).
const VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_KHR: u32 = 0x0000_2000;
const VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR: i32 = 1_000_299_001;

unsafe extern "C" {
    fn wayland_display_vk_instance(device: *mut gstvk::GstVulkanDevice) -> vk::Instance;
    fn wayland_display_vk_instance_handle(instance: *mut gstvk::GstVulkanInstance) -> vk::Instance;
    fn wayland_display_vk_device(device: *mut gstvk::GstVulkanDevice) -> vk::Device;
    fn wayland_display_vk_queue_family(queue: *mut gstvk::GstVulkanQueue) -> u32;
    fn wayland_display_vk_image(memory: *mut gst::ffi::GstMemory) -> vk::Image;
    fn wayland_display_vk_prepare_encode_image(memory: *mut gst::ffi::GstMemory);
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

/// Per-`waylanddisplaysrc`-element owner of the Vulkan objects the encode path shares.
///
/// Historically the
/// owned `GstVulkanInstance` + `GstVulkanDevice` lived in *process-global* `OnceLock` slots,
/// so every session in one process reused the first session's `VkDevice` (a single failure
/// domain — one session's `DEVICE_LOST` corrupts all N; session 2..N's `target_minor`
/// ignored; a deliberate per-process device leak). One `VulkanShare` is now created per
/// element and cloned **once** into that element's compositor thread (mirroring the existing
/// `app_surface_commits` / `renderer_degraded` `Arc`s threaded through
/// `WaylandDisplay::new_with_channel` -> `comp::init`), so each `waylanddisplaysrc` mints,
/// answers `gst.vulkan.{instance,device}` context queries with, and owns its **own** device,
/// released when the element is finalized and this `Arc` drops. N concurrent Vulkan-encode
/// sessions on one host therefore get N isolated devices — a `DEVICE_LOST` on one retires
/// only that element's device, leaving the other N-1 untouched.
///
/// This is a **storage-location** change only: device creation
/// ([`ensure_owned_device`](VulkanShare::ensure_owned_device)) is byte-identical to the
/// previous global path, with no vendor branch, so the RADV path sees no behavioral change.
/// The `wayland_display_vk_*` C bridge is untouched.
pub struct VulkanShare {
    /// The `GstVulkanInstance` we own (keeps it alive + lets us answer `gst.vulkan.instance`
    /// context queries). Replaces the process-global `instance_slot()`.
    instance: Mutex<Option<VulkanInstance>>,
    /// The shared device, filled from `set_context`
    /// ([`handle_set_context`](VulkanShare::handle_set_context)) or minted by
    /// [`ensure_owned_device`](VulkanShare::ensure_owned_device), and read by the converter
    /// when it builds its output ring. Replaces the process-global `device_slot()`.
    device: Mutex<Option<VulkanDevice>>,
}

impl VulkanShare {
    /// A fresh, empty per-element share. Cheap; holds no Vulkan objects until first use. Kept
    /// behind an `Arc` so one clone can be handed to the compositor thread while the element
    /// keeps the other (exactly like `app_surface_commits` / `renderer_degraded`).
    pub fn new() -> Arc<VulkanShare> {
        Arc::new(VulkanShare {
            instance: Mutex::new(None),
            device: Mutex::new(None),
        })
    }

    /// Pull the shared `GstVulkanDevice` out of a received `GstContext` and stash it in **this
    /// element's** slot. Call from `ElementImpl::set_context` for the `gst.vulkan.device`
    /// context type. Returns true if a device is now shared.
    pub fn handle_set_context(&self, context: &gst::Context) -> bool {
        let ctx_ptr = context.to_glib_none().0;
        let mut dev_ptr: *mut gstvk::GstVulkanDevice = std::ptr::null_mut();
        let got = unsafe { gstvk::gst_context_get_vulkan_device(ctx_ptr, &mut dev_ptr) };
        if got == gst::glib::ffi::GFALSE || dev_ptr.is_null() {
            return false;
        }
        let device: VulkanDevice = unsafe { from_glib_full(dev_ptr) };
        let mut slot = self.device.lock().unwrap();
        // Keep the device this element already has. GstBin fans a have-context message to
        // EVERY child, so with two waylanddisplaysrc in one bin the second element's encoder
        // would otherwise overwrite the first element's device while that element's own
        // encoder stays on the original -- producing buffers on one VkDevice and encoding
        // them on another. Under the old process-global slot everyone shared one device and
        // the overwrite was free; per element it is a cross-device hazard, so refuse it here
        // rather than relying on the embedder to check.
        if let Some(existing) = slot.as_ref() {
            if existing.as_ptr() != device.as_ptr() {
                tracing::warn!(
                    "vulkan_share: ignoring a different GstVulkanDevice {dev_ptr:?} offered to an \
                     element already on {:?}",
                    existing.as_ptr()
                );
            }
            return true;
        }
        tracing::debug!("vulkan_share: absorbed shared GstVulkanDevice {dev_ptr:?}");
        *slot = Some(device);
        true
    }

    /// This element's shared device, if one has been created (or absorbed).
    pub fn shared_device(&self) -> Option<VulkanDevice> {
        self.device.lock().unwrap().clone()
    }
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

impl VulkanShare {
    /// Create (once, on **this element**) the `GstVulkanInstance` + `GstVulkanDevice` that *we*
    /// own, on the GPU backing `target_minor`, with the external-memory extensions the
    /// RGBA-dmabuf import needs (`VK_KHR_external_memory_fd` etc.) enabled — which gst-vulkan's
    /// own device does not. This is the device we hand the encoder (see
    /// [`provide_context`](VulkanShare::provide_context)) so producer and encoder share one
    /// device with no zero-copy gap *and* no gstreamer fork. Idempotent **per element** (not
    /// per process): the first call on *this* share mints on *this* element's `target_minor`.
    ///
    /// Device creation below is unchanged from the pre-patch process-global path — no vendor
    /// branch — so RADV behaves byte-identically; only where the result is stored moved from a
    /// `static` slot to `self`.
    /// LOCK ORDER: `device` then `instance`, and this is the only method that holds both.
    /// The `device` guard is deliberately held across the instance open, `Entry::load` and
    /// device enumeration below -- it is the mutual exclusion that stops two threads each
    /// minting a `VkDevice` for this element. Every other accessor takes a single lock as a
    /// temporary (`provide_context` clones instance and device in separate statements, so it
    /// never holds both), which is why the inverse order cannot arise.
    pub fn ensure_owned_device(&self, target_minor: Option<u32>) -> Option<VulkanDevice> {
        let mut slot = self.device.lock().unwrap();
        if let Some(d) = slot.clone() {
            // Whatever this element already has wins over target_minor. Usually that is the
            // device minted below on that very node, so the two agree; the slot records no
            // provenance, so this branch cannot tell the two apart. When they disagree the
            // device was injected by an embedder and the encoder is already bound to it, and
            // minting a second device on the requested node would put producer and encoder on
            // different VkDevices, which is worse than honouring the wrong node.
            return Some(d);
        }
        let instance = VulkanInstance::new();
        if let Err(e) = instance.open() {
            tracing::error!("vulkan_share: GstVulkanInstance open failed: {e}");
            return None;
        }
        let vk_instance = unsafe { wayland_display_vk_instance_handle(instance.as_ptr()) };
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
        *self.instance.lock().unwrap() = Some(instance);
        *slot = Some(device.clone());
        tracing::info!(
            "vulkan_share: created per-element shared GstVulkanDevice (phys idx {index}) with external-memory extensions"
        );
        Some(device)
    }

    /// Answer a `gst.vulkan.{instance,device}` context query on `element` with the device we
    /// own, creating it on `target_minor`'s GPU on first ask. The downstream encoder's
    /// `gst_vulkan_ensure_element_data` then adopts *our* device instead of minting its own.
    pub fn provide_context(
        &self,
        element: &gst::Element,
        query: &mut gst::QueryRef,
        target_minor: Option<u32>,
    ) -> bool {
        if self.ensure_owned_device(target_minor).is_none() {
            return false;
        }
        let inst = self.instance.lock().unwrap().clone();
        let dev = self.device.lock().unwrap().clone();
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

    /// Wait up to `timeout` for **this element's** shared `GstVulkanDevice` to be available.
    ///
    /// The encoder shares its device via a `GstContext` delivered to `set_context` on the
    /// streaming thread, which races the compositor thread that allocates this element's Vulkan
    /// output buffer. Polling here lets the allocation wait for the device to arrive instead of
    /// failing when it merely hasn't been shared *yet*. The race is **per element**, so the
    /// poll is retained, scoped to this share. Returns `None` if it never arrives.
    pub fn wait_for_shared_device(&self, timeout: std::time::Duration) -> Option<VulkanDevice> {
        let start = std::time::Instant::now();
        loop {
            if let Some(dev) = self.shared_device() {
                return Some(dev);
            }
            if start.elapsed() >= timeout {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
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
        let vk_device = wayland_display_vk_device(dev_ptr);
        let vk_instance = wayland_display_vk_instance(dev_ptr);
        if vk_device == vk::Device::null() || vk_instance == vk::Instance::null() {
            return None;
        }
        let vk_physical = gstvk::gst_vulkan_device_get_physical_device(dev_ptr);
        if vk_physical == vk::PhysicalDevice::null() {
            return None;
        }
        // `gst_vulkan_device_select_queue()` is transfer-full: take ownership of the
        // returned GstVulkanQueue so it is not leaked once the family index is read.
        let gfx_ptr =
            gstvk::gst_vulkan_device_select_queue(dev_ptr, vk::QueueFlags::GRAPHICS.as_raw());
        if gfx_ptr.is_null() {
            return None;
        }
        let gfx_queue_family = wayland_display_vk_queue_family(gfx_ptr);
        drop(from_glib_full::<_, VulkanQueue>(gfx_ptr));
        Some(RawVk {
            instance: vk_instance,
            physical: vk_physical,
            device: vk_device,
            gfx_queue_family,
        })
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

        // Seed gst's tracked layout to VIDEO_ENCODE_SRC and clear the per-memory timeline,
        // as PR #37 intends -- but through the target's own headers instead of hand-computed
        // ABI offsets. Both halves matter under the interpipe fan-out (one buffer -> N
        // encoders):
        //   * layout: our converter always leaves the image in VIDEO_ENCODE_SRC, so without
        //     the seed every encoder records an UNDEFINED->VIDEO_ENCODE_SRC layout *write*.
        //     Two concurrent layout writes on the shared image, on the video-encode queue with
        //     no semaphore between them, deadlock the GPU at frame 2. Seeded, each encoder's
        //     barrier is old==new, a pure VIDEO_ENCODE_READ barrier, safe to issue N-way.
        //   * timeline: the encoders only READ, and the converter already CPU-waits its write
        //     fence, so no cross-queue dependency is needed. Left set, two encoders race the
        //     shared timeline value (each reads it, signals value+1) and corrupt it.
        wayland_display_vk_prepare_encode_image(mem_ptr);

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
        let image = wayland_display_vk_image(mem_ptr);
        (image != vk::Image::null()).then_some(image)
    }
}
