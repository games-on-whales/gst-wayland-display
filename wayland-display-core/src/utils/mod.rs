pub mod allocator;
pub mod device;
pub mod renderer;
pub mod va_query;
pub mod va_share;
pub mod video_info;
pub mod vulkan_nv12;
pub mod vulkan_share;

mod target;

pub use self::target::*;

pub mod tests {
    #[cfg(test)]
    use smithay::backend::drm::DrmNode;
    use std::sync::Once;
    pub static INIT: Once = Once::new();

    #[cfg(test)]
    pub fn test_init() -> () {
        INIT.call_once(|| {
            tracing_subscriber::fmt::try_init().ok();
            gst::init().expect("Failed to initialize GStreamer");
        });
    }

    /// A render node whose kernel driver is one of `drivers`, for hardware-gated
    /// tests -- so they target the right GPU on a multi-GPU host instead of a
    /// hardcoded `renderD12x` that may be a different vendor. Only nodes this process can
    /// open read-write are returned.
    #[cfg(test)]
    pub fn pick_render_node(drivers: &[&str]) -> Option<DrmNode> {
        for entry in std::fs::read_dir("/dev/dri").ok()?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("renderD") {
                continue;
            }
            let drv = std::fs::read_to_string(format!("/sys/class/drm/{name}/device/uevent"))
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find_map(|l| l.strip_prefix("DRIVER=").map(str::to_owned))
                })
                .unwrap_or_default();
            let path = format!("/dev/dri/{name}");
            // Skip nodes this process can't open (e.g. not in the `render` group): EGL init
            // on them panics and poisons the shared EGL display cache for later tests.
            let usable = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .is_ok();
            if usable && drivers.iter().any(|d| *d == drv) {
                if let Ok(node) = DrmNode::from_path(&path) {
                    return Some(node);
                }
            }
        }
        None
    }
}
