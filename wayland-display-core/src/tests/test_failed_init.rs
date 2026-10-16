//! A compositor that dies during init (e.g. EGL fails on a bad render node) drops its
//! devices/envs senders without sending. Callers must get an error, not a panic.

use crate::{Command, MaybeRecv, WaylandDisplay, channel};
use std::sync::mpsc;

#[test]
fn dead_compositor_errs_instead_of_panicking() {
    let (command_tx, _command_rx) = channel::<Command>();
    // Senders dropped unsent, as when the compositor thread panics in init.
    let (_, devices_rx) = mpsc::channel();
    let (_, envs_rx) = mpsc::channel();
    let (_hdr_state_tx, hdr_state_rx) = mpsc::channel();
    let mut display = WaylandDisplay {
        thread_handle: Some(std::thread::spawn(|| {})),
        command_tx,
        hdr_state_rx,
        render_device: None,
        tracer: None,
        devices: MaybeRecv::Rx(devices_rx),
        envs: MaybeRecv::Rx(envs_rx),
    };

    assert!(display.env_vars().is_err());
    assert!(display.devices().is_err());
    // Asking again still errs rather than blocking or panicking.
    assert!(display.env_vars().is_err());
}

#[test]
fn published_values_are_returned_and_cached() {
    let (command_tx, _command_rx) = channel::<Command>();
    let (devices_tx, devices_rx) = mpsc::channel();
    let (envs_tx, envs_rx) = mpsc::channel();
    devices_tx
        .send(vec![c"/dev/dri/renderD128".into()])
        .unwrap();
    envs_tx
        .send(vec![c"WAYLAND_DISPLAY=wayland-1".into()])
        .unwrap();
    drop((devices_tx, envs_tx));
    let (_hdr_state_tx, hdr_state_rx) = mpsc::channel();
    let mut display = WaylandDisplay {
        thread_handle: Some(std::thread::spawn(|| {})),
        command_tx,
        hdr_state_rx,
        render_device: None,
        tracer: None,
        devices: MaybeRecv::Rx(devices_rx),
        envs: MaybeRecv::Rx(envs_rx),
    };

    for _ in 0..2 {
        assert_eq!(
            display.env_vars().unwrap().collect::<Vec<_>>(),
            ["WAYLAND_DISPLAY=wayland-1"]
        );
        assert_eq!(
            display.devices().unwrap().collect::<Vec<_>>(),
            ["/dev/dri/renderD128"]
        );
    }
}
