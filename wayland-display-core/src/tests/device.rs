#[test]
fn test_enumerate_gpu_devices() {
    use crate::utils::device::gpu::enumerate_gpu_devices;

    let devices = enumerate_gpu_devices().expect("Failed to enumerate GPU devices");

    // Check that we have at least one device
    assert!(!devices.is_empty(), "No GPU devices found");

    for device in devices {
        // Ensure each device has a valid DRM node
        assert!(
            !device.drm_node().to_string().is_empty(),
            "DRM node path is empty"
        );

        // Ensure the device name is not empty
        assert!(!device.device_name().is_empty(), "Device name is empty");

        tracing::info!("Found GPU: {}", device);
    }
}
