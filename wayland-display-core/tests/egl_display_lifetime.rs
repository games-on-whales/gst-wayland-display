//! Lifetime of the EGL display `setup_renderer` shares between the renderers on one node:
//! one per node for the life of the process, which nothing else in the process can take down
//! under a live renderer. (From #82, adapted from per-renderer to per-process lifetime.)
//!
//! Every test here observes the process-wide display cache, so they run one at a time, and
//! this file is its own test binary so nothing else in the process holds a display meanwhile.
use ash::vk;
use smithay::backend::drm::DrmNode;
use smithay::backend::egl::{EGLDevice, ffi};
use smithay::backend::renderer::gles::GlesRenderer;
use std::ptr;
use std::sync::{Mutex, MutexGuard, PoisonError};
use waylanddisplaycore::utils::renderer::setup_renderer;
use waylanddisplaycore::utils::vulkan_nv12::render_node_minor;

type RawDisplay = ffi::egl::types::EGLDisplay;

/// `EGL_TRACK_REFERENCES_KHR` from `EGL_KHR_display_reference`.
const EGL_TRACK_REFERENCES_KHR: ffi::EGLint = 0x3352;

macro_rules! skip {
    ($($a:tt)*) => {{ eprintln!("skip: {}", format!($($a)*)); return; }};
}

fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

fn handle(renderer: &GlesRenderer) -> RawDisplay {
    **renderer.egl_context().display().get_display_handle()
}

/// Whether `display` is initialized: `eglQueryString` fails with `EGL_NOT_INITIALIZED`
/// on a terminated display.
fn initialized(display: RawDisplay) -> bool {
    unsafe { !ffi::egl::QueryString(display, ffi::egl::VENDOR as ffi::EGLint).is_null() }
}

/// Whether `renderer` can still make its context current.
fn usable(renderer: &mut GlesRenderer) -> bool {
    renderer.with_context(|_| ()).is_ok()
}

/// The device-platform display handle for `device` with `attribs` (looked up, not
/// initialized); `None` if the driver rejects them.
fn platform_display(device: &EGLDevice, attribs: &[ffi::EGLint]) -> Option<RawDisplay> {
    let display = unsafe {
        ffi::egl::GetPlatformDisplayEXT(
            ffi::egl::PLATFORM_DEVICE_EXT,
            device.get_device_handle() as *mut _,
            attribs.as_ptr(),
        )
    };
    (display != ffi::egl::NO_DISPLAY).then_some(display)
}

fn tracked_display(device: &EGLDevice) -> Option<RawDisplay> {
    platform_display(
        device,
        &[
            EGL_TRACK_REFERENCES_KHR,
            ffi::egl::TRUE as ffi::EGLint,
            ffi::egl::NONE as ffi::EGLint,
        ],
    )
}

fn plain_display(device: &EGLDevice) -> Option<RawDisplay> {
    platform_display(device, &[ffi::egl::NONE as ffi::EGLint])
}

fn software_device() -> Option<EGLDevice> {
    EGLDevice::enumerate().ok()?.find(|device| {
        device
            .extensions()
            .iter()
            .any(|e| e == "EGL_MESA_device_software")
    })
}

/// Renderers on `node` share one display, which stays initialized through overlapping
/// renderers, after the last one, and for a later one: nothing tears it down between sessions.
fn assert_display_lives_for_the_process(node: Option<DrmNode>) {
    let first = setup_renderer(node);
    let display = handle(&first);
    assert!(initialized(display));
    let mut second = setup_renderer(node);
    assert_eq!(
        handle(&second),
        display,
        "renderers on a node share one display"
    );
    drop(first);
    assert!(initialized(display), "terminated under a live renderer");
    assert!(usable(&mut second));
    drop(second);
    assert!(initialized(display), "terminated with its last renderer");

    let mut later = setup_renderer(node);
    assert_eq!(
        handle(&later),
        display,
        "a later session got another display"
    );
    assert!(usable(&mut later));
}

#[test]
fn software_display_lives_for_the_process() {
    let _serial = serial();
    if software_device().is_none() {
        skip!("no software EGL device");
    }
    assert_display_lives_for_the_process(None);
}

// On drivers without EGL_KHR_display_reference (Mesa today) the cached display is the plain
// one, shared with every other user of the device in the process. If one of them terminates
// it, renderers created afterwards must still start.
#[test]
fn untracked_display_terminated_elsewhere_still_starts_renderers() {
    let _serial = serial();
    let Some(device) = software_device() else {
        skip!("no software EGL device");
    };
    let live = setup_renderer(None);
    let display = handle(&live);
    if tracked_display(&device) == Some(display) {
        skip!("the software driver tracks display references; the plain display is not ours");
    }
    assert_eq!(plain_display(&device), Some(display));

    // Another component in the process terminates the shared plain display.
    unsafe { ffi::egl::Terminate(display) };
    assert!(!initialized(display));

    let mut next = setup_renderer(None);
    assert_eq!(handle(&next), display);
    assert!(
        usable(&mut next),
        "renderer created after the terminate can't start"
    );
    // The terminate freed `live`'s context, and the driver may hand its handle to `next`:
    // dropping `live` would then destroy `next`'s context, which says nothing about the
    // display. Only check that the older renderer's drop leaves the display alone when the
    // two contexts are distinct.
    let reused = live.egl_context().get_context_handle() == next.egl_context().get_context_handle();
    if reused {
        drop(next);
        drop(live);
    } else {
        drop(live);
        assert!(
            usable(&mut next),
            "dropping the older renderer terminated the display"
        );
        drop(next);
    }
    assert!(
        initialized(display),
        "the display was terminated with its last renderer"
    );
}

/// A render node driven by the `nvidia` kernel driver, with NVIDIA's EGL device for it.
fn nvidia_render_node() -> Option<(DrmNode, EGLDevice)> {
    std::fs::read_dir("/dev/dri")
        .ok()?
        .flatten()
        .find_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("renderD") {
                return None;
            }
            let driver = std::fs::read_link(format!("/sys/class/drm/{name}/device/driver")).ok()?;
            if driver.file_name()? != "nvidia" {
                return None;
            }
            let node = DrmNode::from_path(entry.path()).ok()?;
            // The EGL device setup_renderer picks for the node is the first one listed for it.
            // Only run if that is NVIDIA's own: another vendor's library (Mesa) can list one for
            // the same node first, which would not exercise the NVIDIA driver.
            let device = EGLDevice::enumerate()
                .ok()?
                .find(|d| d.try_get_render_node().ok().flatten() == Some(node))?;
            if !device
                .extensions()
                .iter()
                .any(|e| e == "EGL_NV_device_cuda")
            {
                return None;
            }
            Some((node, device))
        })
}

/// Create a VkInstance and a VkDevice on the physical device behind `node` (matched by its
/// DRM render minor via VK_EXT_physical_device_drm), then destroy both. `false` if no
/// NVIDIA Vulkan device maps to `node`.
fn vulkan_teardown_on(entry: &ash::Entry, node: &DrmNode) -> bool {
    let Some(minor) = node.dev_path().and_then(|p| render_node_minor(p.to_str()?)) else {
        return false;
    };
    unsafe {
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .expect("vkCreateInstance");
        let physical = instance
            .enumerate_physical_devices()
            .unwrap_or_default()
            .into_iter()
            .find(|&pd| {
                let has_drm = instance
                    .enumerate_device_extension_properties(pd)
                    .unwrap_or_default()
                    .iter()
                    .any(|e| {
                        e.extension_name_as_c_str() == Ok(ash::ext::physical_device_drm::NAME)
                    });
                if !has_drm {
                    return false;
                }
                let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
                let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
                instance.get_physical_device_properties2(pd, &mut props);
                props.properties.vendor_id == 0x10de
                    && drm.has_render != 0
                    && drm.render_minor as i64 == minor as i64
            });
        if let Some(physical) = physical {
            let priority = [1.0f32];
            let queue = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(0)
                .queue_priorities(&priority)];
            let device = instance
                .create_device(
                    physical,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&queue),
                    None,
                )
                .expect("vkCreateDevice");
            device.destroy_device(None);
        }
        instance.destroy_instance(None);
        physical.is_some()
    }
}

// The NVIDIA Vulkan driver terminates the plain device-platform EGLDisplay of its GPU when it
// is unloaded, and any other EGL user in the process may terminate it too: renderers sharing
// that display lost their context and later ones failed NotInitialized. The cache must hold
// the tracked display, which survives a Vulkan teardown and an outside terminate of the plain
// display.
#[test]
#[ignore = "needs an NVIDIA GPU with EGL and Vulkan; run via ci/harness.sh gpu"]
fn nvidia_vulkan_teardown_keeps_the_shared_display() {
    let _serial = serial();
    let Some((node, device)) = nvidia_render_node() else {
        skip!("no nvidia render node whose first EGL device is NVIDIA's");
    };
    let Ok(entry) = (unsafe { ash::Entry::load() }) else {
        skip!("no Vulkan loader");
    };

    let mut live = setup_renderer(Some(node));
    let display = handle(&live);
    assert_eq!(
        tracked_display(&device),
        Some(display),
        "the cache must hold the reference-tracked display on NVIDIA"
    );
    // Initialize the plain display too, to report whether this driver still terminates it.
    let plain = plain_display(&device).expect("plain device display");
    unsafe { ffi::egl::Initialize(plain, ptr::null_mut(), ptr::null_mut()) };
    let mut overlapping = setup_renderer(Some(node));

    if !vulkan_teardown_on(&entry, &node) {
        unsafe { ffi::egl::Terminate(plain) };
        skip!("no NVIDIA Vulkan device maps to {node:?}");
    }
    eprintln!(
        "plain display after the Vulkan teardown: {}",
        if initialized(plain) {
            "alive (the driver stayed loaded, or no longer terminates it)"
        } else {
            "terminated by the driver"
        }
    );
    // Another component in the process terminates the plain display.
    unsafe { ffi::egl::Terminate(plain) };

    assert!(usable(&mut live), "live renderer lost its EGL display");
    assert!(
        usable(&mut overlapping),
        "live renderer lost its EGL display"
    );
    let mut next = setup_renderer(Some(node));
    assert!(
        usable(&mut next),
        "renderer created after the teardown can't start"
    );
    drop(live);
    drop(overlapping);
    assert!(initialized(display), "terminated under a live renderer");
    drop(next);

    assert_display_lives_for_the_process(Some(node));
}
