use once_cell::sync::Lazy;
use smithay::backend::drm::{DrmNode, NodeType};
use smithay::backend::egl::native::{EGLNativeDisplay, EGLPlatform};
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay, ffi};
use smithay::backend::renderer::gles::GlesRenderer;
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, Once, PoisonError};

/// One EGL display per render node, kept for the life of the process, and whether it is
/// reference-tracked (see [`TrackedEGLDevice`]).
///
/// `eglGetPlatformDisplay` returns the same native display for the same device, and
/// smithay's `EGLDisplay` calls `eglTerminate` when its last clone drops. A display per
/// session would let one session's teardown terminate the display every other session on
/// that GPU is still rendering with (NVIDIA then fails every EGL call with
/// `EGL_NOT_INITIALIZED`). Holding one clone here means it is never terminated under them.
static EGL_DISPLAYS: Lazy<Mutex<HashMap<Option<DrmNode>, (EGLDisplay, bool)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// `EGL_TRACK_REFERENCES_KHR` from `EGL_KHR_display_reference`.
const EGL_TRACK_REFERENCES_KHR: ffi::EGLint = 0x3352;

/// The handle of the reference-tracked display for `device`, or `None` if the driver
/// rejects the attribute. Only looks the handle up; does not initialize it.
fn tracked_display_handle(device: *const std::ffi::c_void) -> Option<ffi::egl::types::EGLDisplay> {
    let attribs = [
        EGL_TRACK_REFERENCES_KHR,
        ffi::egl::TRUE as ffi::EGLint,
        ffi::egl::NONE as ffi::EGLint,
    ];
    let handle = unsafe {
        ffi::egl::GetPlatformDisplayEXT(
            ffi::egl::PLATFORM_DEVICE_EXT,
            device as *mut _,
            attribs.as_ptr(),
        )
    };
    (handle != ffi::egl::NO_DISPLAY).then_some(handle)
}

/// An `EGLDevice` opened as a reference-counted display (`EGL_KHR_display_reference`).
///
/// EGL returns one handle per (device, attributes), and any user of the plain device display
/// in the process can `eglTerminate` it under every compositor on the GPU: NVIDIA's Vulkan
/// driver does whenever it is unloaded (see [`keep_vulkan_drivers_loaded`]), and so can any
/// other EGL user, such as GStreamer's GL elements. A display requested with
/// `EGL_TRACK_REFERENCES_KHR` is a handle of its own that nobody else terminates. glvnd does
/// not advertise the extension, so it is tried blind; a driver that rejects it (Mesa) gets the
/// plain display, as smithay tries the platforms in order.
struct TrackedEGLDevice(EGLDevice);

impl EGLNativeDisplay for TrackedEGLDevice {
    fn supported_platforms(&self) -> Vec<EGLPlatform<'_>> {
        let device = self.0.get_device_handle() as *mut _;
        let platform = |attrib_list| {
            EGLPlatform::new(
                ffi::egl::PLATFORM_DEVICE_EXT,
                "PLATFORM_DEVICE_EXT",
                device,
                attrib_list,
                &["EGL_EXT_platform_device"],
            )
        };
        vec![
            platform(vec![
                EGL_TRACK_REFERENCES_KHR,
                ffi::egl::TRUE as ffi::EGLint,
                ffi::egl::NONE as ffi::EGLint,
            ]),
            platform(vec![ffi::egl::NONE as ffi::EGLint]),
        ]
    }

    fn surface_type(&self) -> ffi::EGLint {
        self.0.surface_type()
    }

    fn identifier(&self) -> Option<String> {
        self.0.identifier()
    }
}

pub fn get_egl_device_for_node(drm_node: &DrmNode) -> EGLDevice {
    let drm_node = drm_node
        .node_with_type(NodeType::Render)
        .and_then(Result::ok)
        .unwrap_or(drm_node.clone());
    EGLDevice::enumerate()
        .expect("Failed to enumerate EGLDevices")
        .find(|d| d.try_get_render_node().unwrap_or_default() == Some(drm_node))
        .expect("Unable to find EGLDevice for drm-node")
}

/// Keep one Vulkan instance alive for the life of the process.
///
/// The Vulkan loader unloads a driver when the last instance in the process goes away (the
/// converter's at the end of every session, GStreamer's), and loads and initializes it again
/// for the next one. NVIDIA's driver (`libGLX_nvidia.so.0`) breaks on both:
///
/// * Unloading it calls `eglTerminate` on the GPU's EGL device display, the display every
///   compositor on that GPU renders with, so their GLES calls start failing with
///   `EGL_NOT_INITIALIZED`.
/// * Loading it while another thread enumerates instance extensions (GStreamer's Vulkan
///   plugin does when it loads) crashes in the loader: a 615.71 driver with loader 1.4.341
///   segfaults on every run of a two-thread repro.
///
/// An instance that is never destroyed keeps the driver loaded and initialized. Call it
/// before anything else in the process can use Vulkan: the first load is the racy one. The
/// source element does when it is constructed, and every Vulkan entry point here does too.
pub fn keep_vulkan_drivers_loaded() {
    static KEPT: Once = Once::new();
    KEPT.call_once(|| {
        // SAFETY: loading the Vulkan loader and creating an instance with no layers or
        // extensions; both are leaked on purpose and never used again.
        let kept = unsafe {
            ash::Entry::load()
                .map_err(|e| e.to_string())
                .and_then(|entry| {
                    let app =
                        ash::vk::ApplicationInfo::default().api_version(ash::vk::API_VERSION_1_1);
                    let instance = entry
                        .create_instance(
                            &ash::vk::InstanceCreateInfo::default().application_info(&app),
                            None,
                        )
                        .map_err(|e| e.to_string())?;
                    Ok((entry, instance))
                })
        };
        match kept {
            Ok(kept) => std::mem::forget(kept),
            Err(e) => tracing::debug!("No Vulkan instance kept ({e})"),
        }
    });
}

/// Lock `mutex`, taking its data even if a thread panicked while holding it.
///
/// For data that a panic can't leave half-updated, where refusing every later caller
/// would turn one failure into a permanent one.
fn lock_recovering<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn setup_renderer(render_node: Option<DrmNode>) -> GlesRenderer {
    // Before the display exists: loading NVIDIA's Vulkan driver initializes it too.
    keep_vulkan_drivers_loaded();
    // EGL init below panics on failure while this guard is held. Recover from poisoning so
    // one failed display doesn't break every later one in the process: an entry is inserted
    // only after its EGLDisplay was created successfully.
    let mut displays = lock_recovering(&EGL_DISPLAYS);
    let (egl, tracked) = match displays.get(&render_node) {
        Some(cached) => cached.clone(),
        None => {
            let device = match render_node.as_ref() {
                Some(render_node) => get_egl_device_for_node(render_node),
                None => EGLDevice::enumerate()
                    .expect("Failed to enumerate EGLDevices")
                    .find(|device| {
                        device
                            .extensions()
                            .iter()
                            .any(|e| e == "EGL_MESA_device_software")
                    })
                    .expect("Failed to find software device"),
            };
            let device_handle = device.get_device_handle();
            let egl = unsafe {
                EGLDisplay::new(TrackedEGLDevice(device)).expect("Failed to create EGLDisplay")
            };
            let tracked =
                tracked_display_handle(device_handle) == Some(egl.get_display_handle().handle);
            tracing::debug!(tracked, "EGL display for {render_node:?}");
            displays.insert(render_node, (egl.clone(), tracked));
            (egl, tracked)
        }
    };
    drop(displays);
    // An untracked display is the plain one every EGL user of the device shares, and any of
    // them may have terminated it since the last session. eglInitialize brings it back and is
    // a no-op on a live display. Not on a tracked display: nobody else terminates it, and an
    // extra initialization would only add a reference.
    if !tracked {
        let initialized = unsafe {
            ffi::egl::Initialize(
                egl.get_display_handle().handle,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if initialized != ffi::egl::TRUE {
            tracing::warn!("Couldn't re-initialize the EGL display");
        }
    }
    // GLES2 is fine for HDR: fp16/10-bit dmabufs import via EGLImage (bypassing the GLES core
    // texture-format limits), so they're in the renderer's importable set already. (A GLES3
    // context fails to configure in this headless/surfaceless EGL setup and isn't needed.)
    let context = EGLContext::new(&egl).expect("Failed to initialize EGL context");
    let renderer = unsafe { GlesRenderer::new(context) }.expect("Failed to initialize renderer");
    renderer
}

#[cfg(test)]
mod tests {
    use super::*;

    // A panic during EGL init (e.g. an unusable render node) used to poison EGL_DISPLAYS,
    // so every later display in the process panicked at `lock().unwrap()`. Tested on a
    // local mutex, so the process-wide map is never poisoned for other tests.
    #[test]
    fn lock_recovering_survives_a_poisoned_mutex() {
        let mutex = Mutex::new(vec![1]);
        let _ = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let mut guard = mutex.lock().unwrap();
                    guard.push(2);
                    panic!("simulated EGL init failure while holding the lock");
                })
                .join()
        });
        assert!(mutex.is_poisoned());

        let mut guard = lock_recovering(&mutex);
        assert_eq!(*guard, vec![1, 2]);
        guard.push(3);
        drop(guard);
        assert_eq!(*lock_recovering(&mutex), vec![1, 2, 3]);
    }

    // Overlapping sessions on one GPU: one session's teardown must never be the last
    // reference to the display (whose drop calls eglTerminate), or it races a session
    // starting on that GPU and terminates the display under it.
    #[test]
    fn a_dropped_renderer_leaves_the_shared_display_usable() {
        use smithay::backend::allocator::Fourcc;
        use smithay::backend::renderer::gles::GlesTexture;
        use smithay::backend::renderer::{Bind, Frame, Offscreen, Renderer};
        use smithay::utils::{Physical, Rectangle, Size, Transform};

        let first = setup_renderer(None);
        let second = setup_renderer(None);
        assert!(
            std::sync::Arc::ptr_eq(
                &first.egl_context().display().get_display_handle(),
                &second.egl_context().display().get_display_handle()
            ),
            "renderers on one device must share one EGL display"
        );
        let handle = first.egl_context().display().get_display_handle();
        drop(first);
        drop(second);
        assert!(
            std::sync::Arc::strong_count(&handle) > 1,
            "dropping every renderer must not drop the last reference to the display"
        );
        drop(handle);

        let mut second = setup_renderer(None);

        let size: Size<i32, Physical> = (16, 16).into();
        let mut texture: GlesTexture = second
            .create_buffer(Fourcc::Abgr8888, (16, 16).into())
            .expect("texture after the first renderer dropped");
        let mut target = second.bind(&mut texture).expect("bind");
        let mut frame = second
            .render(&mut target, size, Transform::Normal)
            .expect("render after the first renderer dropped");
        frame
            .clear([0.0, 0.0, 0.0, 1.0].into(), &[Rectangle::from_size(size)])
            .expect("clear");
        frame
            .finish()
            .expect("finish")
            .wait()
            .expect("the display must still be initialized");
    }
}
