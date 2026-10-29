use once_cell::sync::Lazy;
use smithay::backend::drm::{DrmNode, NodeType};
use smithay::backend::egl::native::{EGLNativeDisplay, EGLPlatform};
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay, ffi};
use smithay::backend::renderer::gles::GlesRenderer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

static EGL_DISPLAYS: Lazy<Mutex<HashMap<Option<DrmNode>, Weak<EGLDisplay>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

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

/// `EGL_TRACK_REFERENCES_KHR` from `EGL_KHR_display_reference`.
const EGL_TRACK_REFERENCES_KHR: ffi::EGLint = 0x3352;

/// An `EGLDevice` opened as a reference-counted display (`EGL_KHR_display_reference`).
///
/// The NVIDIA Vulkan driver terminates the plain (untracked) device-platform `EGLDisplay`
/// of its GPU when a `VkInstance` is destroyed: `vkCreateInstance` + `vkDestroyInstance`
/// alone is enough (driver 595.91.07). EGL returns one handle per (device, attributes), so
/// every renderer on that node -- and any live compositor -- then fails with
/// `NotInitialized`. `VulkanNv12` creates and destroys an instance per NV12 output buffer,
/// and GStreamer's Vulkan elements do the same, so this happens at every NV12 teardown.
///
/// A display requested with `EGL_TRACK_REFERENCES_KHR` is a different handle from the
/// untracked one the driver tears down, and its `eglInitialize`/`eglTerminate` are counted,
/// so dropping one compositor's display can't terminate another's either. glvnd does not
/// advertise the extension, so the attribute is tried blind; a driver that rejects it gets
/// the plain display (smithay tries the platforms in order).
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

/// Lock `mutex`, taking its data even if a thread panicked while holding it.
///
/// For data that a panic can't leave half-updated, where refusing every later caller
/// would turn one failure into a permanent one.
fn lock_recovering<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn setup_renderer(render_node: Option<DrmNode>) -> GlesRenderer {
    // EGL init below panics on failure while this guard is held. Recover from poisoning so
    // one failed display doesn't break every later one in the process: the map only holds
    // Weak refs and an entry is inserted only after its EGLDisplay was created successfully.
    let mut displays = lock_recovering(&EGL_DISPLAYS);
    let maybe_display = displays
        .get(&render_node)
        .and_then(|weak_display| weak_display.upgrade());

    let egl = match maybe_display {
        Some(display) => display,
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
            let egl = unsafe {
                EGLDisplay::new(TrackedEGLDevice(device)).expect("Failed to create EGLDisplay")
            };
            let display = Arc::new(egl);
            displays.insert(render_node, Arc::downgrade(&display));
            display
        }
    };
    // GLES2 is fine for HDR: fp16/10-bit dmabufs import via EGLImage (bypassing the GLES core
    // texture-format limits), so they're in the renderer's importable set already. (A GLES3
    // context fails to configure in this headless/surfaceless EGL setup and isn't needed.)
    let context = EGLContext::new(&egl).expect("Failed to initialize EGL context");
    let renderer = unsafe { GlesRenderer::new(context) }.expect("Failed to initialize renderer");
    drop(displays);
    // The map only holds Weak refs, so the renderer has to own a strong one: otherwise the
    // entry is dead as soon as this returns and the next renderer calls `EGLDisplay::new`
    // again, which re-runs `eglInitialize` on the same handle while only one `eglTerminate`
    // follows -- a leaked reference on a tracked display, which then never terminates.
    renderer
        .egl_context()
        .user_data()
        .insert_if_missing_threadsafe(|| DisplayKeepalive(Some(egl)));
    renderer
}

/// A renderer's strong reference to its cached display, kept in its context's user data.
struct DisplayKeepalive(Option<Arc<EGLDisplay>>);

impl Drop for DisplayKeepalive {
    fn drop(&mut self) {
        // Drop the last reference (and so terminate) under the lock, so `setup_renderer` on
        // another thread either reuses this display or creates one after it is gone -- never
        // re-initializes a handle that is about to be terminated.
        let _displays = lock_recovering(&EGL_DISPLAYS);
        self.0.take();
    }
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

    // The NVIDIA Vulkan driver terminated the shared render-node EGLDisplay whenever a
    // VkInstance was destroyed (every NV12 output teardown): the live renderer lost its
    // context and every later one on the node failed `NotInitialized`.
    #[test]
    #[ignore = "needs a GPU with a Vulkan driver; run via ci/harness.sh gpu"]
    fn vulkan_instance_teardown_keeps_the_shared_display() {
        let Some(node) = std::fs::read_dir("/dev/dri")
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("renderD"))
            .find_map(|e| DrmNode::from_path(e.path()).ok())
        else {
            eprintln!("skip: no render node");
            return;
        };
        let Ok(entry) = (unsafe { ash::Entry::load() }) else {
            eprintln!("skip: no Vulkan loader");
            return;
        };
        let mut live = setup_renderer(Some(node));
        unsafe {
            let instance = entry
                .create_instance(&ash::vk::InstanceCreateInfo::default(), None)
                .expect("vkCreateInstance");
            instance.destroy_instance(None);
        }
        // A renderer for a new compositor on the node reuses the cached display...
        let _next = setup_renderer(Some(node));
        // ...and the one that was already running can still make its context current.
        live.with_context(|_| ())
            .expect("live renderer lost its EGL context");
    }
}
