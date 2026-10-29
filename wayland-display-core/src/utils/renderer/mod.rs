use once_cell::sync::Lazy;
use smithay::backend::drm::{DrmNode, NodeType};
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
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
            let egl = unsafe { EGLDisplay::new(device).expect("Failed to create EGLDisplay") };
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
}
