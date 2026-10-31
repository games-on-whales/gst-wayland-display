//! `get_render_device` must answer without a compositor round trip: the
//! streaming thread calls it under the pad stream lock during renegotiation,
//! so a compositor that never answers must not be able to wedge it.

use crate::utils::device::gpu::GPUDevice;
use crate::utils::tests::pick_render_node;
use crate::utils::va_share::VaShare;
use crate::utils::vulkan_share::VulkanShare;
use crate::{Command, MaybeRecv, WaylandDisplay, channel};
use std::sync::{Arc, mpsc};
use std::time::Duration;

#[test]
fn get_render_device_does_not_wait_on_compositor() {
    // A command channel nobody services: the compositor is alive (the
    // receiver exists) but never answers, as when its event loop is wedged.
    let (command_tx, _unserviced_rx) = channel::<Command>();
    let (_devices_tx, devices_rx) = mpsc::channel();
    let (_envs_tx, envs_rx) = mpsc::channel();
    let (_hdr_state_tx, hdr_state_rx) = mpsc::channel();
    let display = WaylandDisplay {
        thread_handle: Some(std::thread::spawn(|| {})),
        command_tx,
        hdr_state_rx,
        reply_timeout: crate::COMPOSITOR_REPLY_TIMEOUT,
        render_device: None,
        tracer: None,
        devices: MaybeRecv::Rx(devices_rx),
        envs: MaybeRecv::Rx(envs_rx),
    };

    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = done_tx.send(display.get_render_device());
    });
    assert!(
        done_rx.recv_timeout(Duration::from_secs(5)).is_ok(),
        "get_render_device blocked on an unanswered compositor"
    );
}

fn constructed_render_devices(render_node: &str) -> [Option<GPUDevice>; 2] {
    let display = WaylandDisplay::new(Some(render_node.into())).unwrap();
    let (command_tx, command_rx) = channel::<Command>();
    let display_with_channel = WaylandDisplay::new_with_channel(
        Some(render_node.into()),
        command_tx,
        command_rx,
        VulkanShare::new(),
        Arc::new(VaShare::default()),
    )
    .unwrap();
    [
        display.get_render_device(),
        display_with_channel.get_render_device(),
    ]
}

#[test]
fn software_render_target_has_no_render_device() {
    assert_eq!(constructed_render_devices("software"), [None, None]);
}

#[test]
#[ignore = "needs a GPU render node; run via ci/harness.sh gpu"]
fn hardware_render_target_resolves_its_render_device() {
    // Not a hard-coded node: on a missing or unopenable one the compositor's EGL init panics and
    // poisons the shared EGL display cache, failing unrelated tests in this process.
    let Some(node) = pick_render_node(&["amdgpu", "radeon", "i915", "xe", "nvidia"]) else {
        eprintln!("skip: no GPU render node");
        return;
    };
    let expected = GPUDevice::try_from(node).unwrap();
    assert_eq!(
        constructed_render_devices(node.dev_path().unwrap().to_str().unwrap()),
        [Some(expected.clone()), Some(expected)]
    );
}
