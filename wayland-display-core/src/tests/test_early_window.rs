//! A client that creates its toplevel before the output exists (i.e. before caps
//! negotiation runs `apply_video_info`) must still get configured and mapped once the
//! output appears, instead of having its window silently dropped.

use crate::comp::apply_video_info;
use crate::tests::fixture::Fixture;
use crate::utils::RenderTarget;
use crate::utils::video_info::GstVideoInfo;
use gst_video::{VideoFormat, VideoInfo};
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
    let configure = f.client.last_configure().expect("no configure received");
    assert_eq!(configure.size, size);
    assert!(configure.states.contains(&xdg_toplevel::State::Fullscreen));
    assert!(configure.states.contains(&xdg_toplevel::State::Activated));
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
