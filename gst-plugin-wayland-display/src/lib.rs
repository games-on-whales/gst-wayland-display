use gst::glib;
#[cfg(feature = "cuda")]
use waylanddisplaycore::utils::allocator::cuda;

pub mod utils;
mod waylandsrc;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    // Disable the lavapipe (llvmpipe) Vulkan ICD process-wide before any Vulkan instance is
    // created. On mixed-GPU / software-fallback hosts the loader otherwise dlopen()s lavapipe
    // during vkCreateInstance, and its libLLVM static init collides with the libLLVM already
    // loaded by the mesa GLES renderer -- "CommandLine Option registered more than once" ->
    // llvm::report_fatal_error -> abort (crashes in supported_modifiers()/caps and in gst's
    // gst.vulkan.instance). lavapipe can't do Vulkan video encode anyway, so the producer and
    // the encoder never want it. Honour an explicit user override.
    if std::env::var_os("VK_LOADER_DRIVERS_DISABLE").is_none() {
        // SAFETY: plugin_init runs once at plugin load, before any Vulkan use or worker threads
        // touch the loader, so there is no concurrent env access.
        unsafe { std::env::set_var("VK_LOADER_DRIVERS_DISABLE", "*lvp_icd*") };
    }
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
