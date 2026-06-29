use once_cell::sync::Lazy;
use smithay::backend::drm::{DrmNode, NodeType};
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
use smithay::backend::renderer::gles::GlesRenderer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

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

pub fn setup_renderer(render_node: Option<DrmNode>) -> GlesRenderer {
    let mut displays = EGL_DISPLAYS.lock().unwrap();
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
    // EGLContext::new() defaults to a GLES 2 context; GLES2 can't texture RGBA16F / RGB10_A2,
    // so EGL excludes fp16/10-bit from the renderer's importable dmabuf formats and HDR clients
    // can't hand us HDR buffers. Under WOLF_HDR_CM request a GLES 3.0 context so those formats
    // become importable. Gated so the default (SDR) path is byte-for-byte unchanged.
    let context = if std::env::var("WOLF_HDR_CM").is_ok() {
        use smithay::backend::egl::context::{GlAttributes, PixelFormatRequirements};
        let attributes = GlAttributes {
            version: (3, 0),
            profile: None,
            debug: false,
            vsync: false,
        };
        // All-permissive reqs ("don't care") so config selection matches whatever the
        // headless EGL offers (the explicit _8_bit() reqs failed to find a config). We only
        // need GLES 3.0 (so fp16/10-bit dmabufs are textureable), not a specific framebuffer.
        let reqs = PixelFormatRequirements {
            hardware_accelerated: None,
            color_bits: None,
            float_color_buffer: false,
            alpha_bits: None,
            depth_bits: None,
            stencil_bits: None,
            multisampling: None,
        };
        match EGLContext::new_with_config(&egl, attributes, reqs) {
            Ok(ctx) => {
                tracing::info!("WOLF_HDR_CM: created a GLES 3.0 EGL context (fp16/10-bit import)");
                ctx
            }
            Err(e) => {
                tracing::warn!(
                    "WOLF_HDR_CM: GLES 3.0 context failed ({e}); falling back to default"
                );
                EGLContext::new(&egl).expect("Failed to initialize EGL context")
            }
        }
    } else {
        EGLContext::new(&egl).expect("Failed to initialize EGL context")
    };
    let renderer = unsafe { GlesRenderer::new(context) }.expect("Failed to initialize renderer");
    renderer
}
