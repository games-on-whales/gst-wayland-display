//! Calls that wait on a compositor reply must give up when the compositor never
//! answers (e.g. its event loop is wedged): the streaming thread makes them under
//! the pad stream lock, so an unbounded wait wedges the whole element.

use crate::{COMPOSITOR_REPLY_TIMEOUT, Command, MaybeRecv, WaylandDisplay, channel};
use std::sync::mpsc;
use std::time::Duration;

fn unanswered_display() -> WaylandDisplay {
    // A command channel nobody services: the compositor is alive (the receiver
    // exists) but never answers. Leak the receiver so it outlives the test thread.
    let (command_tx, unserviced_rx) = channel::<Command>();
    std::mem::forget(unserviced_rx);
    let (_devices_tx, devices_rx) = mpsc::channel();
    let (_envs_tx, envs_rx) = mpsc::channel();
    let (_hdr_state_tx, hdr_state_rx) = mpsc::channel();
    WaylandDisplay {
        thread_handle: Some(std::thread::spawn(|| {})),
        command_tx,
        hdr_state_rx,
        reply_timeout: Duration::from_millis(200),
        render_device: None,
        tracer: None,
        devices: MaybeRecv::Rx(devices_rx),
        envs: MaybeRecv::Rx(envs_rx),
    }
}

fn returns_within<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = done_tx.send(f());
    });
    done_rx.recv_timeout(Duration::from_secs(5)).ok()
}

#[test]
fn get_supported_dma_formats_errors_on_unanswered_compositor() {
    let display = unanswered_display();
    let result = returns_within(move || display.get_supported_dma_formats().is_err())
        .expect("get_supported_dma_formats blocked on an unanswered compositor");
    assert!(
        result,
        "an unanswered query must be an error, not empty formats"
    );
}

#[test]
fn frame_errors_on_unanswered_compositor() {
    let display = unanswered_display();
    let result =
        returns_within(move || display.frame()).expect("frame blocked on an unanswered compositor");
    assert_eq!(result.unwrap_err(), gst::FlowError::Error);
}

#[test]
fn get_supported_dma_formats_errors_when_compositor_is_gone() {
    // The compositor thread died: its receiver is dropped. Must not panic.
    let mut display = unanswered_display();
    let (command_tx, rx) = channel::<Command>();
    drop(rx);
    display.command_tx = command_tx;
    let result = returns_within(move || display.get_supported_dma_formats().is_err())
        .expect("get_supported_dma_formats panicked or blocked on a dead compositor");
    assert!(result);
}

#[test]
fn drop_does_not_wait_forever_on_unanswered_compositor() {
    // A compositor thread that never acts on Quit: teardown (stop() drops the display)
    // must not hang on joining it.
    let mut display = unanswered_display();
    let (_never_tx, never_rx) = mpsc::channel::<()>();
    std::mem::forget(_never_tx);
    display.thread_handle = Some(std::thread::spawn(move || {
        let _ = never_rx.recv();
    }));
    returns_within(move || drop(display))
        .expect("dropping the display blocked on an unanswered compositor");
}

#[test]
fn reply_timeout_leaves_room_for_slow_first_frames() {
    // The first reply can queue behind compositor start-up and CUDA init; a tight
    // bound would kill healthy streams.
    assert!(COMPOSITOR_REPLY_TIMEOUT >= Duration::from_secs(10));
}
