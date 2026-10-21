//! Calls that wait on a compositor reply must give up when the compositor never
//! answers (e.g. its event loop is wedged): the streaming thread makes them under
//! the pad stream lock, so an unbounded wait wedges the whole element.

use crate::{COMPOSITOR_REPLY_TIMEOUT, Command, MaybeRecv, PendingFrame, WaylandDisplay, channel};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

fn unanswered_display() -> WaylandDisplay {
    // A command channel nobody services: the compositor is alive (the receiver
    // exists) but never answers. Leak the receiver so it outlives the test thread.
    let (command_tx, unserviced_rx) = channel::<Command>();
    std::mem::forget(unserviced_rx);
    // Senders kept alive but never used: a compositor that never publishes its devices/envs.
    let (devices_tx, devices_rx) = mpsc::channel::<Vec<std::ffi::CString>>();
    let (envs_tx, envs_rx) = mpsc::channel::<Vec<std::ffi::CString>>();
    std::mem::forget((devices_tx, envs_tx));
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
fn env_vars_and_devices_error_on_unanswered_compositor() {
    // start() reads env_vars() under the state lock; a compositor that never publishes them
    // must not wedge it.
    let mut display = unanswered_display();
    let result = returns_within(move || {
        let envs_err = display.env_vars().is_err();
        (envs_err, display.devices().is_err())
    })
    .expect("env_vars/devices blocked on an unanswered compositor");
    assert_eq!(result, (true, true));
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

// The compositor must be able to hand over a frame nobody is receiving yet (a create() cancelled
// by unlock() picks it up later); a rendezvous channel blocks its event loop until then.
#[test]
fn frame_reply_does_not_block_the_compositor() {
    gst::init().unwrap();
    let display = unanswered_display();
    let (command_tx, unserviced) = channel::<Command>();
    let mut display = display;
    display.command_tx = command_tx;
    let pending = display.request_frame().unwrap();
    let Ok(Command::Buffer(reply, _)) = unserviced.try_recv() else {
        panic!("no frame request sent");
    };
    assert!(
        reply.try_send(Ok(gst::Buffer::new())).is_ok(),
        "delivering the frame would block the compositor"
    );
    assert!(pending.wait().is_ok());
    std::mem::forget(unserviced);
}

// after start() waited out env_vars(), dropping the display waited a second full
// timeout for Quit, and C callers asking for the env length and then the entries waited twice.
#[test]
fn init_timeout_is_not_waited_out_again() {
    let mut display = unanswered_display();
    let (_never_tx, never_rx) = mpsc::channel::<()>();
    std::mem::forget(_never_tx);
    display.thread_handle = Some(std::thread::spawn(move || {
        let _ = never_rx.recv();
    }));
    let elapsed = returns_within(move || {
        assert!(display.env_vars().is_err());
        // Any further wait now would be the full timeout, far beyond the bound below.
        display.reply_timeout = Duration::from_secs(60);
        let started = Instant::now();
        assert!(display.env_vars().is_err());
        drop(display);
        started.elapsed()
    })
    .expect("waited out the reply timeout again");
    assert!(elapsed < Duration::from_secs(1), "took {elapsed:?}");
}

#[test]
fn value_published_after_a_timeout_is_still_taken() {
    let (envs_tx, envs_rx) = mpsc::channel();
    let mut envs = MaybeRecv::Rx(envs_rx);
    assert_eq!(
        envs.get(Duration::from_millis(10)).unwrap_err(),
        mpsc::RecvTimeoutError::Timeout
    );
    envs_tx.send(vec![1]).unwrap();
    assert_eq!(envs.get(Duration::ZERO).unwrap(), &vec![1]);
}

// GstBaseSrc unlock(): a frame wait on a wedged compositor must end once cancelled, not after the
// reply timeout, or stopping the pipeline waits behind it.
#[test]
fn cancelled_frame_wait_returns_flushing() {
    let (buffer_tx, buffer_rx) = mpsc::sync_channel(0);
    let pending = PendingFrame {
        buffer_rx,
        timeout: Duration::from_secs(60),
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let cancelled = cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        cancelled.store(true, Ordering::Release);
    });
    let result = returns_within(move || pending.wait_unless(&cancel))
        .expect("a cancelled frame wait kept waiting");
    assert_eq!(result.unwrap_err(), gst::FlowError::Flushing);
    drop(buffer_tx);
}
