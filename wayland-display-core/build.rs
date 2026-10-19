fn main() {
    // Compile a tiny, header-checked bridge for Vulkan handle/barrier fields that bindgen
    // intentionally truncates because Vulkan handles are platform-dependent typedefs.
    // This avoids guessing private struct offsets in Rust and makes any GStreamer ABI drift
    // a compile-time failure against the exact headers used by the plugin.
    let vulkan = pkg_config::Config::new()
        .atleast_version("1.28")
        .probe("gstreamer-vulkan-1.0")
        .expect("gstreamer-vulkan-1.0 >= 1.28 is required");
    let mut bridge = cc::Build::new();
    bridge.file("src/utils/vulkan_bridge.c");
    for include in vulkan.include_paths {
        bridge.include(include);
    }
    bridge.compile("wayland_display_vulkan_bridge");

    // Check if the cuda feature is enabled
    #[cfg(feature = "cuda")]
    {
        // Link GStreamer CUDA library
        if let Err(e) = pkg_config::Config::new()
            .atleast_version("1.24")
            .probe("gstreamer-cuda-1.0")
        {
            eprintln!(
                "Warning: gstreamer-cuda-1.0 not found via pkg-config: {}",
                e
            );
        }
    }

    // Rerun if build script changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/utils/vulkan_bridge.c");
}
