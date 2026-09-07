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
    use std::sync::Once;
    pub static INIT: Once = Once::new();

    #[cfg(test)]
    pub fn test_init() -> () {
        INIT.call_once(|| {
            tracing_subscriber::fmt::try_init().ok();
            gst::init().expect("Failed to initialize GStreamer");
        });
    }
}
