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

use crate::utils::video_info::VulkanCodec;
use crate::utils::vulkan_nv12::PixFmt;
use ash::vk;
use gst::glib::translate::{ToGlibPtr, from_glib_full};
use gst::prelude::*;
use gstreamer_vulkan::prelude::*;
use gstreamer_vulkan::{VulkanDevice, VulkanInstance, VulkanPhysicalDevice, VulkanQueue};
use gstreamer_vulkan_sys as gstvk;
use std::sync::{Arc, Condvar, Mutex};

// VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR / VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR via raw
// values (stable, and avoids depending on the named ash constants existing).
const VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_KHR: u32 = 0x0000_2000;

/// Device extensions the converter cannot run without: importing the compositor's RGBA
/// dmabuf. A device lacking any of them is refused instead of reaching an ash stub.
pub const REQUIRED_DEVICE_EXTENSIONS: [&str; 3] = [
    "VK_KHR_external_memory_fd",
    "VK_EXT_external_memory_dma_buf",
    "VK_EXT_image_drm_format_modifier",
];
/// Enabled when available: queue-family-foreign gives the dmabuf acquire/release barriers
/// their proper source family (QUEUE_FAMILY_EXTERNAL is used otherwise).
const OPTIONAL_DEVICE_EXTENSIONS: [&str; 2] = [
    "VK_KHR_external_semaphore_fd",
    "VK_EXT_queue_family_foreign",
];

// VK_KHR_video_encode_av1 postdates ash 0.38; values from vulkan_core.h.
const VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_PROFILE_INFO_KHR: i32 = 1_000_513_005;
const VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_KHR: u32 = 0x0004_0000;
const STD_VIDEO_AV1_PROFILE_MAIN: u32 = 0;

/// `VkVideoEncodeAV1ProfileInfoKHR`.
#[repr(C)]
struct VideoEncodeAv1ProfileInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    std_profile: u32,
}

unsafe extern "C" {
    fn wayland_display_vk_instance(device: *mut gstvk::GstVulkanDevice) -> vk::Instance;
    fn wayland_display_vk_instance_handle(instance: *mut gstvk::GstVulkanInstance) -> vk::Instance;
    fn wayland_display_vk_device(device: *mut gstvk::GstVulkanDevice) -> vk::Device;
    fn wayland_display_vk_queue_family(queue: *mut gstvk::GstVulkanQueue) -> u32;
    fn wayland_display_vk_queue_handle(queue: *mut gstvk::GstVulkanQueue) -> vk::Queue;
    fn wayland_display_vk_image(memory: *mut gst::ffi::GstMemory) -> vk::Image;
    fn wayland_display_vk_linear_layout(
        memory: *mut gst::ffi::GstMemory,
        offsets: *mut usize,
        strides: *mut i32,
    ) -> gst::glib::ffi::gboolean;
    fn wayland_display_vk_prepare_encode_image(memory: *mut gst::ffi::GstMemory);
}

/// Raw Vulkan handles + queue pulled out of the shared `GstVulkanDevice`, ready to drive an
/// `ash`-loaded device.
#[derive(Clone, Debug)]
pub struct RawVk {
    pub instance: vk::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: vk::Device,
    pub gfx_queue_family: u32,
    /// gst's graphics queue. Submits to its `VkQueue` must hold its submit lock, since the
    /// encoders on the same device submit to it too.
    pub gfx_queue: VulkanQueue,
    /// The family gst opens for video encode, if the device has one.
    pub encode_queue_family: Option<u32>,
    /// Whether VK_EXT_queue_family_foreign is enabled on the device.
    pub queue_family_foreign: bool,
}

impl RawVk {
    pub fn gfx_vk_queue(&self) -> vk::Queue {
        unsafe { wayland_display_vk_queue_handle(self.gfx_queue.as_ptr()) }
    }
}

/// Holds a `GstVulkanQueue`'s submit lock for its lifetime.
pub struct QueueSubmitGuard<'a>(&'a VulkanQueue);

impl<'a> QueueSubmitGuard<'a> {
    pub fn new(queue: &'a VulkanQueue) -> Self {
        unsafe { gstvk::gst_vulkan_queue_submit_lock(queue.as_ptr()) };
        QueueSubmitGuard(queue)
    }
}

impl Drop for QueueSubmitGuard<'_> {
    fn drop(&mut self) {
        unsafe { gstvk::gst_vulkan_queue_submit_unlock(self.0.as_ptr()) };
    }
}

/// The required converter extensions `device` does not have enabled.
pub fn missing_extensions(device: &VulkanDevice) -> Vec<&'static str> {
    REQUIRED_DEVICE_EXTENSIONS
        .into_iter()
        .filter(|ext| !device.is_extension_enabled(ext))
        .collect()
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
    options: Mutex<crate::utils::vulkan_nv12::ConversionOptions>,
    /// The shared device, filled from `set_context`
    /// ([`handle_set_context`](VulkanShare::handle_set_context)) or minted by
    /// [`ensure_owned_device`](VulkanShare::ensure_owned_device), and read by the converter
    /// when it builds its output ring. Replaces the process-global `device_slot()`.
    device: Mutex<Option<VulkanDevice>>,
    device_ready: Condvar,
    /// Set once the Vulkan output cannot work for this element (a device offered without
    /// the converter's extensions, or a downstream encoder on another device); the element
    /// then stops offering `memory:VulkanImage` and negotiates dmabuf/RGBA instead.
    refused: std::sync::atomic::AtomicBool,
}

impl VulkanShare {
    /// A fresh, empty per-element share. Cheap; holds no Vulkan objects until first use. Kept
    /// behind an `Arc` so one clone can be handed to the compositor thread while the element
    /// keeps the other (exactly like `app_surface_commits` / `renderer_degraded`).
    pub fn new() -> Arc<VulkanShare> {
        Arc::new(VulkanShare {
            instance: Mutex::new(None),
            options: Mutex::new(Default::default()),
            device: Mutex::new(None),
            device_ready: Condvar::new(),
            refused: Default::default(),
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
                if let (Some(ours), Some(offered)) = (
                    physical_device_uuid(existing),
                    physical_device_uuid(&device),
                ) && ours != offered
                {
                    tracing::warn!(
                        "vulkan_share: rejected device belongs to a different physical GPU; bind the encoder to the source device before starting the pipeline"
                    );
                }
            }
            return true;
        }
        let missing = missing_extensions(&device);
        if !missing.is_empty() {
            tracing::warn!(
                "vulkan_share: refusing shared GstVulkanDevice {dev_ptr:?} without {missing:?}; \
                 memory:VulkanImage output is not offered"
            );
            self.refused
                .store(true, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        tracing::debug!("vulkan_share: absorbed shared GstVulkanDevice {dev_ptr:?}");
        *slot = Some(device);
        self.device_ready.notify_all();
        true
    }

    pub fn conversion_options(&self) -> crate::utils::vulkan_nv12::ConversionOptions {
        *self.options.lock().unwrap()
    }

    pub fn set_conversion_options(&self, options: crate::utils::vulkan_nv12::ConversionOptions) {
        *self.options.lock().unwrap() = options;
    }

    /// This element's shared device, if one has been created (or absorbed).
    pub fn shared_device(&self) -> Option<VulkanDevice> {
        self.device.lock().unwrap().clone()
    }

    /// Whether `memory:VulkanImage` output may still be offered; see `refused`.
    pub fn usable(&self) -> bool {
        !self.refused.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check the device a directly linked downstream element answers a
    /// `gst.vulkan.device` context query with. A device other than ours means producer and
    /// encoder would run on different `VkDevice`s, so the Vulkan output is refused. No
    /// answer (e.g. behind interpipe) proves nothing and is accepted.
    pub fn check_peer_device(&self, src_pad: &gst::Pad) -> bool {
        let Some(ours) = self.shared_device() else {
            return self.usable();
        };
        let mut query = gst::query::Context::new("gst.vulkan.device");
        if !src_pad.peer_query(&mut query) {
            return self.usable();
        }
        let Some(context) = query.context_owned() else {
            return self.usable();
        };
        let mut dev_ptr: *mut gstvk::GstVulkanDevice = std::ptr::null_mut();
        let got =
            unsafe { gstvk::gst_context_get_vulkan_device(context.to_glib_none().0, &mut dev_ptr) };
        if got == gst::glib::ffi::GFALSE || dev_ptr.is_null() {
            return self.usable();
        }
        let theirs: VulkanDevice = unsafe { from_glib_full(dev_ptr) };
        if theirs.as_ptr() != ours.as_ptr() {
            if self.usable() {
                tracing::warn!(
                    "vulkan_share: downstream encoder is on GstVulkanDevice {:?}, not ours \
                     ({:?}); falling back to dmabuf/RGBA output",
                    theirs.as_ptr(),
                    ours.as_ptr()
                );
            }
            self.refused
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.usable()
    }
}

// VK_QUEUE_VIDEO_ENCODE_BIT_KHR — the encoder (gst_vulkan_encoder_create_from_queue) requires a
// queue family with this bit, and gst_vulkan_device_choose_queues only opens an encode queue if the
// chosen physical device has one. Software devices (llvmpipe) do NOT, so a device without this bit
// is useless for the encode path and makes vulkanh264enc fail to link.
const VK_QUEUE_VIDEO_ENCODE_BIT_KHR: vk::QueueFlags = vk::QueueFlags::from_raw(0x0000_0040);

// VkPhysicalDevice handles cannot be compared across Vulkan instances. UUIDs can.
fn physical_device_uuid(device: &VulkanDevice) -> Option<[u8; vk::UUID_SIZE]> {
    unsafe {
        let entry = ash::Entry::load().ok()?;
        let instance = ash::Instance::load(
            entry.static_fn(),
            wayland_display_vk_instance(device.as_ptr()),
        );
        let physical = gstvk::gst_vulkan_device_get_physical_device(device.as_ptr());
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        instance.get_physical_device_properties2(physical, &mut properties);
        Some(id.device_uuid)
    }
}

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
        // enable_extension() fails when the physical device lacks the extension.
        for ext in REQUIRED_DEVICE_EXTENSIONS {
            if !device.enable_extension(ext) {
                tracing::error!("vulkan_share: physical device {index} lacks {ext}");
                return None;
            }
        }
        for ext in OPTIONAL_DEVICE_EXTENSIONS {
            if !device.enable_extension(ext) {
                tracing::info!("vulkan_share: optional {ext} unavailable");
            }
        }
        if let Err(e) = device.open() {
            tracing::error!("vulkan_share: GstVulkanDevice open failed: {e}");
            return None;
        }
        *self.instance.lock().unwrap() = Some(instance);
        *slot = Some(device.clone());
        self.device_ready.notify_all();
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
    /// output buffer. A condition variable lets allocation wait for the device instead of
    /// failing when it merely has not been shared yet. Returns `None` on timeout.
    pub fn wait_for_shared_device(&self, timeout: std::time::Duration) -> Option<VulkanDevice> {
        let slot = self.device.lock().unwrap();
        let (slot, _) = self
            .device_ready
            .wait_timeout_while(slot, timeout, |d| d.is_none())
            .unwrap();
        slot.clone()
    }
}

/// Extract the raw `VkInstance`/`VkPhysicalDevice`/`VkDevice`, gst's graphics queue and the
/// video-encode queue family from the (shared) `GstVulkanDevice`.
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
        // `gst_vulkan_device_select_queue()` is transfer-full.
        let gfx_ptr =
            gstvk::gst_vulkan_device_select_queue(dev_ptr, vk::QueueFlags::GRAPHICS.as_raw());
        if gfx_ptr.is_null() {
            return None;
        }
        let gfx_queue: VulkanQueue = from_glib_full(gfx_ptr);
        let gfx_queue_family = wayland_display_vk_queue_family(gfx_ptr);
        let enc_ptr =
            gstvk::gst_vulkan_device_select_queue(dev_ptr, VK_QUEUE_VIDEO_ENCODE_BIT_KHR.as_raw());
        let encode_queue_family = (!enc_ptr.is_null()).then(|| {
            let family = wayland_display_vk_queue_family(enc_ptr);
            drop(from_glib_full::<_, VulkanQueue>(enc_ptr));
            family
        });
        Some(RawVk {
            instance: vk_instance,
            physical: vk_physical,
            device: vk_device,
            gfx_queue_family,
            gfx_queue,
            encode_queue_family,
            queue_family_foreign: device.is_extension_enabled("VK_EXT_queue_family_foreign"),
        })
    }
}

/// Allocate ONE encode-src NV12/P010 image as a `GstVulkanImageMemory`, built directly via
/// `gst_vulkan_image_memory_alloc_with_image_info` so we control the `VkImageCreateInfo`
/// (usage `TRANSFER_DST | VIDEO_ENCODE_SRC` + the encoder's `VkVideoProfileListInfoKHR`
/// chained in). This bypasses `GstVulkanImageBufferPool`'s generic-format-feature check, which
/// rejects NV12 on NVIDIA because the encode-input feature is only reported via the
/// profile-specific query. The image is shared CONCURRENTly between `queue_families` (our
/// graphics family and the encoder's video-encode family), so no ownership transfer is needed
/// between our copy and the encode read. Returns a `GstBuffer` holding the single image
/// memory + VideoMeta.
#[allow(clippy::too_many_arguments)]
pub fn alloc_encode_src_buffer(
    device: &VulkanDevice,
    width: u32,
    height: u32,
    codec: VulkanCodec,
    profile: &str,
    fmt: PixFmt,
    linear_encsrc: bool,
    queue_families: &[u32],
) -> Option<gst::Buffer> {
    if !codec.accepts(fmt.gst_format(), profile) {
        tracing::error!("Vulkan {codec:?} profile {profile} cannot encode {fmt:?}");
        return None;
    }
    let bit_depth = match fmt {
        PixFmt::P010 => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        PixFmt::Nv12 => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };
    unsafe {
        // Profile chain (matches what the encoder builds from caps): the codec struct chained
        // off the VkVideoProfileInfoKHR; usage info omitted.
        let std_h264_idc = match profile {
            "high" | "constrained-high" | "progressive-high" => {
                vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH
            }
            "main" => vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
            _ => vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
        };
        let mut h264 = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(std_h264_idc);
        let mut h265 = vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(match fmt {
            PixFmt::P010 => vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10,
            PixFmt::Nv12 => vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
        });
        let av1 = VideoEncodeAv1ProfileInfo {
            s_type: vk::StructureType::from_raw(
                VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_PROFILE_INFO_KHR,
            ),
            p_next: std::ptr::null(),
            std_profile: STD_VIDEO_AV1_PROFILE_MAIN,
        };
        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(bit_depth)
            .chroma_bit_depth(bit_depth);
        profile_info = match codec {
            VulkanCodec::H264 => profile_info
                .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
                .push_next(&mut h264),
            VulkanCodec::H265 => profile_info
                .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
                .push_next(&mut h265),
            VulkanCodec::Av1 => {
                profile_info.p_next = &av1 as *const VideoEncodeAv1ProfileInfo as *const _;
                profile_info.video_codec_operation(vk::VideoCodecOperationFlagsKHR::from_raw(
                    VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_KHR,
                ))
            }
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
        // EXCLUSIVE would need a graphics -> video-encode ownership transfer that the encoder
        // (whose barriers use QUEUE_FAMILY_IGNORED) never performs.
        let image_info = if queue_families.len() > 1 {
            image_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(queue_families)
        } else {
            image_info
        };

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
                "vulkan_share: LINEAR encode-src alloc failed (linear-encode-src) -- \
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
                "vulkan_share: encode-src allocated with tiling={:?} (linear-encode-src)",
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
            let format = match fmt {
                PixFmt::Nv12 => gst_video::VideoFormat::Nv12,
                PixFmt::P010 => gst_video::VideoFormat::P01010le,
            };
            let mut offsets = [0usize; 2];
            let mut strides = [0i32; 2];
            if wayland_display_vk_linear_layout(mem_ptr, offsets.as_mut_ptr(), strides.as_mut_ptr())
                != 0
            {
                gst_video::VideoMeta::add_full(
                    b,
                    gst_video::VideoFrameFlags::empty(),
                    format,
                    width,
                    height,
                    &offsets,
                    &strides,
                )
                .ok()?;
            } else {
                if effective_tiling == vk::ImageTiling::LINEAR {
                    tracing::error!("Unable to query LINEAR encode image plane layout");
                    return None;
                }
                // OPTIMAL tiling has no portable byte layout. This meta describes
                // logical video planes only; consumers must use the VkImage API.
                gst_video::VideoMeta::add(
                    b,
                    gst_video::VideoFrameFlags::empty(),
                    format,
                    width,
                    height,
                )
                .ok()?;
            }
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
