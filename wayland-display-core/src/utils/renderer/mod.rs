use once_cell::sync::Lazy;
use smithay::backend::drm::{DrmNode, NodeType};
use smithay::backend::egl::native::{EGLNativeDisplay, EGLPlatform};
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay, ffi};
use smithay::backend::renderer::gles::GlesRenderer;
use std::cell::Cell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

/// One display per render node (`None` = the software device), shared by every renderer on it.
///
/// Ownership invariant, against smithay a166cf4 (the rev pinned in Cargo.lock):
/// - A display is terminated when its smithay `EGLDisplayHandle` drops (egl/display.rs:103-113).
///   Its only strong owners are each renderer's `EGLContext::display` field and the
///   [`DisplayKeepalive`] in that context's user data. `EGLContext::drop` destroys the context
///   (egl/context.rs:489-501), then its fields drop in declaration order (context.rs:25-33):
///   `display` before `user_data`, so the keepalive drops the last reference. (The renderer's
///   other possible owner, `GlesRenderer::egl_reader`, exists only after `bind_wl_display`,
///   which we never call; `get_display_handle()` clones elsewhere are temporaries.)
/// - The keepalive drops that last reference -- and so runs `eglTerminate` -- while holding
///   this lock. That is what serializes terminate against create: `EGLDisplay::new` hands back
///   any still-live handle for the same native display from smithay's registry and runs
///   `eglInitialize` on it again (egl/display.rs:220-250). Released outside the lock, a create
///   in the gap between the `Weak` here dying and the terminate would add an initialization
///   that no terminate matches (a leak on a tracked display), so the drop must stay inside it.
/// - That cannot re-enter this lock. `eglTerminate` calls back only into smithay's
///   `EGL_KHR_debug` callback (egl/ffi.rs:42-69), which only logs; the handle's drop takes no
///   lock (smithay's own registry lock is taken only inside `EGLDisplay::new`, so the order is
///   always this lock, then smithay's). [`setup_renderer`] attaches the keepalive only after
///   releasing this lock, so no keepalive is ever dropped while it is held, even when renderer
///   setup panics. [`lock_displays`] asserts that in debug builds.
static EGL_DISPLAYS: Lazy<Mutex<HashMap<Option<DrmNode>, CachedDisplay>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

struct CachedDisplay {
    display: Weak<EGLDisplay>,
    /// Opened with `EGL_TRACK_REFERENCES_KHR` (see [`TrackedEGLDevice`]); false on drivers
    /// that rejected it and gave us the plain, untracked display.
    tracked: bool,
}

thread_local! {
    static HOLDING_DISPLAYS: Cell<bool> = const { Cell::new(false) };
}

/// [`EGL_DISPLAYS`], locked. Re-entering it on one thread would deadlock; debug builds panic
/// instead (see the ownership invariant on [`EGL_DISPLAYS`]).
struct DisplaysGuard(MutexGuard<'static, HashMap<Option<DrmNode>, CachedDisplay>>);

fn lock_displays() -> DisplaysGuard {
    debug_assert!(
        !HOLDING_DISPLAYS.get(),
        "EGL_DISPLAYS re-entered on this thread"
    );
    // EGL init panics on failure while this guard is held. Recover from poisoning so one
    // failed display doesn't break every later one in the process: the map only holds Weak
    // refs and an entry is inserted only after its EGLDisplay was created successfully.
    let guard = lock_recovering(&EGL_DISPLAYS);
    HOLDING_DISPLAYS.set(true);
    DisplaysGuard(guard)
}

impl Drop for DisplaysGuard {
    fn drop(&mut self) {
        HOLDING_DISPLAYS.set(false);
    }
}

impl Deref for DisplaysGuard {
    type Target = HashMap<Option<DrmNode>, CachedDisplay>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DisplaysGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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

/// `EGL_TRACK_REFERENCES_KHR` from `EGL_KHR_display_reference`.
const EGL_TRACK_REFERENCES_KHR: ffi::EGLint = 0x3352;

/// The handle a tracked display for `device` has, or `None` if the driver rejects the
/// attribute. Only looks the handle up; does not initialize it.
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
///
/// This is only about the display being *terminated*. It does not address the separate NVIDIA
/// deadlock seen with many NV12 compositors starting and stopping at once (`vkDestroyDevice` in
/// `VulkanNv12::drop` blocking in the driver while its EGL worker waits on the GLSI lock).
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
    let mut displays = lock_displays();
    let cached = displays
        .get(&render_node)
        .and_then(|cached| Some((cached.display.upgrade()?, cached.tracked)));

    let egl = match cached {
        Some((display, true)) => display,
        Some((display, false)) => {
            // An untracked display is the one handle every user of this device in the process
            // shares, and any of them may `eglTerminate` it. Initializing a live display is a
            // no-op; re-initializing a terminated one lets this renderer start, as it did when
            // every renderer created its own display. (Never on a tracked display: there it
            // would add a reference that nothing releases.)
            unsafe {
                ffi::egl::Initialize(
                    **display.get_display_handle(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            };
            display
        }
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
            let tracked = tracked_display_handle(device_handle) == Some(**egl.get_display_handle());
            let display = Arc::new(egl);
            displays.insert(
                render_node,
                CachedDisplay {
                    display: Arc::downgrade(&display),
                    tracked,
                },
            );
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
    // Attached only now that the lock is released (see the invariant on EGL_DISPLAYS).
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
        // re-initializes a handle that is about to be terminated (see EGL_DISPLAYS).
        let _displays = lock_displays();
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
}
