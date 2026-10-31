//! A teardown that times out detaches the compositor thread instead of joining it. A new display
//! must start and work while the detached one is still wedged, and the detached one must release
//! what it holds once it wakes up.

use crate::utils::tests::test_init;
use crate::{Command, GstVideoInfo, WaylandDisplay};
use gst_video::{VideoFormat, VideoInfo};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The wayland socket `display` listens on, from the environment it publishes.
fn socket_path(display: &mut WaylandDisplay) -> PathBuf {
    let name = display
        .env_vars()
        .expect("the compositor did not start")
        .find_map(|var| var.strip_prefix("WAYLAND_DISPLAY="))
        .expect("no WAYLAND_DISPLAY")
        .to_owned();
    PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap()).join(name)
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting until {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn new_display_works_while_a_detached_compositor_is_wedged() {
    test_init();

    let mut wedged = WaylandDisplay::new(Some("software".into())).unwrap();
    let wedged_socket = socket_path(&mut wedged);
    let wedged_lock = wedged_socket.with_extension("lock");
    assert!(wedged_socket.exists() && wedged_lock.exists());

    // Wedge the real event loop: it blocks handing over a reply nobody receives.
    let (blocking_tx, blocking_rx) = mpsc::sync_channel(0);
    wedged
        .command_tx
        .send(Command::GetSupportedDmaFormats(blocking_tx))
        .unwrap();
    // Commands are handled in order, so this one is never answered while the loop is wedged.
    let (probe_tx, probe_rx) = mpsc::sync_channel(1);
    wedged
        .command_tx
        .send(Command::GetSupportedDmaFormats(probe_tx))
        .unwrap();
    assert!(
        probe_rx.recv_timeout(Duration::from_millis(300)).is_err(),
        "the compositor is not wedged"
    );

    // Teardown gives up on the join and detaches the thread.
    wedged.reply_timeout = Duration::from_millis(200);
    let dropping = Instant::now();
    drop(wedged);
    assert!(dropping.elapsed() < Duration::from_secs(5));
    assert!(
        wedged_socket.exists() && wedged_lock.exists(),
        "the detached compositor should still hold its socket while wedged"
    );

    // A new display gets its own socket and renders while the detached one is alive.
    let mut fresh = WaylandDisplay::new(Some("software".into())).unwrap();
    let fresh_socket = socket_path(&mut fresh);
    assert_ne!(fresh_socket, wedged_socket, "socket name collision");
    std::os::unix::net::UnixStream::connect(&fresh_socket).expect("connect to the new display");
    fresh.set_video_info(GstVideoInfo::RAW(
        VideoInfo::builder(VideoFormat::Rgbx, 320, 240)
            .fps(gst::Fraction::new(30, 1))
            .build()
            .unwrap(),
    ));
    for _ in 0..3 {
        assert!(fresh.frame().is_ok(), "the new display did not render");
    }

    // Waking the detached compositor: it acts on the Quit that teardown queued, exits, and
    // removes its socket and lock file.
    drop(blocking_rx);
    wait_until("the detached compositor released its socket", || {
        !wedged_socket.exists() && !wedged_lock.exists()
    });
    assert!(
        probe_rx.try_recv().is_ok(),
        "the woken compositor did not drain its queue"
    );

    // Its exit left the new display working.
    assert!(
        fresh.frame().is_ok(),
        "the new display broke when the old one exited"
    );
    assert!(fresh_socket.exists());
    let joining = Instant::now();
    drop(fresh);
    assert!(joining.elapsed() < Duration::from_secs(5));
    // Drop joins within its bound; a compositor slower than that to exit (a loaded machine)
    // is detached instead and removes its socket as it finishes.
    wait_until("the dropped compositor released its socket", || {
        !fresh_socket.exists()
    });
}
