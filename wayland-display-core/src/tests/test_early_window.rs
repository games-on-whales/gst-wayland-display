//! A client that creates its toplevel before the output exists (i.e. before caps
//! negotiation runs `apply_video_info`) must still get configured and mapped once the
//! output appears, instead of having its window silently dropped.

use crate::comp::apply_video_info;
use crate::tests::client::Configure;
use crate::tests::fixture::Fixture;
use crate::utils::RenderTarget;
use crate::utils::video_info::GstVideoInfo;
use gst_video::{VideoFormat, VideoInfo};
use smithay::utils::IsAlive;
use test_log::test;
use wayland_protocols::xdg::shell::client::xdg_toplevel;

fn apply(f: &mut Fixture, width: u32, height: u32) {
    apply_video_info(
        &mut f.server,
        GstVideoInfo::RAW(
            VideoInfo::builder(VideoFormat::Rgba, width, height)
                .build()
                .unwrap(),
        ),
        &RenderTarget::Software,
        None,
    );
    f.round_trip();
    f.round_trip();
}

fn assert_fullscreen_configure(f: &Fixture, size: (i32, i32)) {
    assert_fullscreen(
        f.client.last_configure().expect("no configure received"),
        size,
    );
}

fn assert_fullscreen(configure: &Configure, size: (i32, i32)) {
    assert_eq!(configure.size, size);
    assert!(configure.states.contains(&xdg_toplevel::State::Fullscreen));
    assert!(configure.states.contains(&xdg_toplevel::State::Activated));
}

/// Window `idx` has received exactly `count` configures, the last a fullscreen one at `size`.
fn assert_configures(f: &Fixture, idx: usize, count: usize, size: (i32, i32)) {
    let configures = f.client.configures_of(idx);
    assert_eq!(configures.len(), count, "window {idx}: configure count");
    assert_fullscreen(&configures.last().unwrap().1, size);
}

/// Every toplevel the compositor still holds back from the space is alive.
fn assert_held_windows_alive(f: &Fixture) {
    for window in f
        .server
        .pending_windows
        .iter()
        .chain(&f.server.windows_awaiting_output)
    {
        assert!(window.alive(), "a destroyed toplevel is still held");
    }
}

#[test]
fn window_committed_before_output_is_mapped() {
    let mut f = Fixture::without_output();
    assert!(f.server.output.is_none());

    // Initial (bufferless) commit with no output: the compositor can't size the
    // configure yet, so it must hold the window rather than answer or drop it.
    f.client.create_window();
    f.round_trip();
    assert_eq!(f.client.configure_count(), 0);

    apply(&mut f, 320, 240);
    assert_eq!(
        f.client.configure_count(),
        1,
        "the held window must receive its initial configure once the output exists"
    );
    assert_fullscreen_configure(&f, (320, 240));

    f.client.setup_window(320, 240);
    f.round_trip();
    f.round_trip();

    assert_eq!(f.server.space.elements().count(), 1);
    assert!(f.server.pending_windows.is_empty());
    assert!(
        f.server
            .seat
            .get_keyboard()
            .unwrap()
            .current_focus()
            .is_some(),
        "the mapped window must get keyboard focus"
    );
}

#[test]
fn resize_before_map_reconfigures_unmapped_window() {
    let mut f = Fixture::without_output();

    f.client.create_window();
    f.round_trip();
    apply(&mut f, 320, 240);
    assert_eq!(f.client.configure_count(), 1);

    // Renegotiation before the client acks and maps: exactly one more configure, at
    // the new size -- not a stale size, and not a duplicate initial configure.
    apply(&mut f, 640, 480);
    assert_eq!(f.client.configure_count(), 2);
    assert_fullscreen_configure(&f, (640, 480));

    f.client.setup_window(640, 480);
    f.round_trip();
    f.round_trip();
    assert_eq!(f.server.space.elements().count(), 1);
}

#[test]
fn uncommitted_window_is_not_configured_early() {
    let mut f = Fixture::without_output();

    // xdg-shell: the initial configure answers the initial commit, so a toplevel that
    // hasn't committed when the output appears must not be configured yet.
    f.client.create_window_uncommitted();
    f.round_trip();
    apply(&mut f, 320, 240);
    apply(&mut f, 640, 480);
    assert_eq!(f.client.configure_count(), 0);

    f.client.commit_window(0);
    f.round_trip();
    assert_eq!(f.client.configure_count(), 1);
    assert_fullscreen_configure(&f, (640, 480));

    f.client.setup_window(640, 480);
    f.round_trip();
    f.round_trip();
    assert_eq!(f.server.space.elements().count(), 1);
}

#[test]
fn window_destroyed_before_output_is_forgotten() {
    let mut f = Fixture::without_output();

    f.client.create_window();
    f.round_trip();
    f.client.destroy_window();
    f.round_trip();

    apply(&mut f, 320, 240);

    assert!(f.server.pending_windows.is_empty());
    assert!(f.server.windows_awaiting_output.is_empty());
    assert_eq!(f.server.space.elements().count(), 0);
}

/// Committed and uncommitted toplevels waiting for the first output at the same time:
/// only the committed ones (in `windows_awaiting_output`) get the initial configure when
/// the output appears; the uncommitted ones (in `pending_windows`) get theirs when they
/// commit.
#[test]
fn several_windows_across_both_queues() {
    let mut f = Fixture::without_output();

    f.client.create_window(); // 0: committed
    f.client.create_window_uncommitted(); // 1: uncommitted
    f.client.create_window(); // 2: committed
    f.client.create_window_uncommitted(); // 3: uncommitted
    f.round_trip();

    assert_eq!(f.server.windows_awaiting_output.len(), 2);
    assert_eq!(f.server.pending_windows.len(), 2);
    for idx in 0..4 {
        assert!(f.client.configures_of(idx).is_empty());
    }

    apply(&mut f, 320, 240);
    assert_configures(&f, 0, 1, (320, 240));
    assert!(f.client.configures_of(1).is_empty());
    assert_configures(&f, 2, 1, (320, 240));
    assert!(f.client.configures_of(3).is_empty());
    assert!(f.server.windows_awaiting_output.is_empty());
    assert_eq!(f.server.pending_windows.len(), 4);

    f.client.commit_window(1);
    f.round_trip();
    assert_configures(&f, 1, 1, (320, 240));
    assert!(f.client.configures_of(3).is_empty());

    // Renegotiation resizes every configured window once and still leaves the
    // uncommitted one alone.
    apply(&mut f, 640, 480);
    assert_configures(&f, 0, 2, (640, 480));
    assert_configures(&f, 1, 2, (640, 480));
    assert_configures(&f, 2, 2, (640, 480));
    assert!(f.client.configures_of(3).is_empty());

    f.client.commit_window(3);
    f.round_trip();
    assert_configures(&f, 3, 1, (640, 480));

    for idx in 0..4 {
        f.client.setup_window_at(idx, 640, 480);
    }
    f.round_trip();
    f.round_trip();
    assert_eq!(f.server.space.elements().count(), 4);
    assert!(f.server.pending_windows.is_empty());
    for idx in 0..4 {
        assert_eq!(
            f.client.configures_of(idx).len(),
            if idx == 3 { 1 } else { 2 }
        );
    }
}

/// Destroying one toplevel from each holding list before the output appears drops it
/// from that list, and the survivors are configured as usual.
#[test]
fn windows_destroyed_from_both_queues_before_output() {
    let mut f = Fixture::without_output();

    f.client.create_window(); // committed: windows_awaiting_output
    f.client.create_window_uncommitted(); // uncommitted: pending_windows
    f.client.create_window(); // committed: windows_awaiting_output
    f.client.create_window_uncommitted(); // uncommitted: pending_windows
    f.round_trip();
    assert_eq!(f.server.windows_awaiting_output.len(), 2);
    assert_eq!(f.server.pending_windows.len(), 2);

    f.client.destroy_window_at(2);
    f.client.destroy_window_at(1);
    f.round_trip();
    // The client now holds the committed window at 0 and the uncommitted one at 1.
    assert_eq!(f.server.windows_awaiting_output.len(), 1);
    assert_eq!(f.server.pending_windows.len(), 1);
    assert_held_windows_alive(&f);

    apply(&mut f, 320, 240);
    assert_configures(&f, 0, 1, (320, 240));
    assert!(f.client.configures_of(1).is_empty());
    assert!(f.server.windows_awaiting_output.is_empty());
    assert_eq!(f.server.pending_windows.len(), 2);
    assert_held_windows_alive(&f);

    f.client.commit_window(1);
    f.round_trip();
    assert_configures(&f, 1, 1, (320, 240));
}

/// Configured-but-unmapped windows each get exactly one configure per resolution change,
/// at the new size.
#[test]
fn renegotiation_reconfigures_each_unmapped_window_once() {
    let mut f = Fixture::without_output();
    apply(&mut f, 320, 240);

    for _ in 0..3 {
        f.client.create_window();
    }
    f.round_trip();
    for idx in 0..3 {
        assert_configures(&f, idx, 1, (320, 240));
    }

    apply(&mut f, 640, 480);
    for idx in 0..3 {
        assert_configures(&f, idx, 2, (640, 480));
    }

    apply(&mut f, 1280, 720);
    for idx in 0..3 {
        assert_configures(&f, idx, 3, (1280, 720));
    }
    assert_eq!(f.server.pending_windows.len(), 3);
    assert_eq!(f.server.space.elements().count(), 0);
}

/// A renegotiation that keeps the resolution must not re-configure unmapped windows,
/// whether their last configure is still unacked or was acked but not committed.
#[test]
fn same_size_renegotiation_sends_no_configure() {
    let mut f = Fixture::without_output();
    apply(&mut f, 320, 240);

    f.client.create_window();
    f.client.create_window();
    f.round_trip();
    // Window 0 acks its configure without committing; window 1 leaves it unacked.
    f.client.ack_window(0);
    f.round_trip();
    for idx in 0..2 {
        assert_configures(&f, idx, 1, (320, 240));
    }

    apply(&mut f, 320, 240);
    for idx in 0..2 {
        assert_configures(&f, idx, 1, (320, 240));
    }

    // A real change still gets through, once, and repeating it is again a no-op.
    apply(&mut f, 640, 480);
    apply(&mut f, 640, 480);
    for idx in 0..2 {
        assert_configures(&f, idx, 2, (640, 480));
    }
}

/// A window destroyed after its configure but before mapping is forgotten: the next
/// renegotiation configures only the survivor and doesn't touch the dead toplevel.
#[test]
fn window_destroyed_after_configure_is_not_reconfigured() {
    let mut f = Fixture::without_output();
    apply(&mut f, 320, 240);

    f.client.create_window();
    f.client.create_window();
    f.round_trip();
    assert_eq!(f.server.pending_windows.len(), 2);

    f.client.destroy_window_at(0);
    f.round_trip();
    assert_eq!(f.server.pending_windows.len(), 1);
    assert_held_windows_alive(&f);

    apply(&mut f, 640, 480);
    // The survivor is now at index 0.
    assert_configures(&f, 0, 2, (640, 480));
    assert_eq!(f.server.pending_windows.len(), 1);
    assert_held_windows_alive(&f);

    f.client.setup_window_at(0, 640, 480);
    f.round_trip();
    f.round_trip();
    assert_eq!(f.server.space.elements().count(), 1);
    assert!(f.server.pending_windows.is_empty());
}
