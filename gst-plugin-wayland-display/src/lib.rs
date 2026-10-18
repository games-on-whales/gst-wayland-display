use gst::glib;
#[cfg(feature = "cuda")]
use waylanddisplaycore::utils::allocator::cuda;

pub mod utils;
mod waylandsrc;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    waylandsrc::register(plugin)?;
    tracing_subscriber::fmt::try_init().ok();
    #[cfg(feature = "cuda")]
    match cuda::init_cuda() {
        Ok(_) => {
            tracing::info!("CUDA initialization successful");
        }
        Err(e) => {
            tracing::info!("CUDA initialization failed: {}", e);
        }
    }
    Ok(())
}

gst::plugin_define!(
    waylanddisplaysrc,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "MIT/X11", // https://gitlab.freedesktop.org/gstreamer/gstreamer/-/blob/master/gst/gstplugin.c#L95
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
