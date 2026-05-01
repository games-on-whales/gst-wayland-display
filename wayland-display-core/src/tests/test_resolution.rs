//! Integration tests for the `Command::VideoInfo` resolution-switching path.
//!
//! Drives the same `apply_video_info` function the live handler uses (see
//! `comp::init`), then round-trips through a real Wayland client to assert that
//! `wl_output.mode` and `xdg_toplevel.configure` events land on the client with
//! the expected dimensions.
//!
//! Complements wolf's Catch2 [WAYLAND][resolution] suite: those drive resolution
//! changes from Wolf's side of the FFI boundary; these drive them at the source,
//! and failing here localises a regression to the compositor rather than to the
//! wolf <-> gst-wayland-display glue.
//!
//! Note on `wl_output.Mode` assertions vs `xdg_toplevel.Configure` size
//! assertions: the compositor intersects each toplevel's declared `max_size`
//! with the new output rect before calling `send_configure`. A client that has
//! not explicitly set a `max_size` leaves it at the xdg-shell default `(0, 0)`
//! (meaning "no limit"), which `Rectangle::from_size` treats as an empty rect,
//! so the intersection is empty and the configure goes out with `size = None`
//! (wire-encoded as `width=0, height=0`, xdg-shell's "client picks" value).
//! These tests therefore assert on `wl_output.mode` dimensions and on the fact
//! that a fresh configure was broadcast -- not on the configure's width/height,
//! which is intentionally unconstrained for Wolf's client model.

use crate::comp::apply_video_info;
use crate::tests::fixture::Fixture;
use crate::utils::RenderTarget;
use crate::utils::video_info::GstVideoInfo;
use gst::Fraction;
use gst_video::{VideoFormat, VideoInfo};
use test_log::test;
use wayland_client::protocol::wl_output;

fn make_video_info(width: u32, height: u32, fps: i32) -> GstVideoInfo {
    GstVideoInfo::RAW(
        VideoInfo::builder(VideoFormat::Rgba, width, height)
            .fps(Fraction::new(fps, 1))
            .build()
            .expect("failed to build VideoInfo"),
    )
}

/// Drive `apply_video_info` with software rendering (matches the Fixture's
/// `RenderTarget::Software` backing), then round-trip twice so the compositor
/// can dispatch events on the event-loop tick and the client can read them off
/// the socket.
fn apply(f: &mut Fixture, width: u32, height: u32, fps: i32) {
    apply_video_info(
        &mut f.server,
        make_video_info(width, height, fps),
        &RenderTarget::Software,
        None,
    );
    f.round_trip();
    f.round_trip();
}

fn latest_mode_dimensions(client_events: &[wl_output::Event]) -> Option<(i32, i32, i32)> {
    client_events.iter().rev().find_map(|e| match e {
        wl_output::Event::Mode {
            width,
            height,
            refresh,
            ..
        } => Some((*width, *height, *refresh)),
        _ => None,
    })
}

#[test]
fn apply_video_info_broadcasts_new_mode_to_client() {
    // Fixture::new() already creates a 320x240 output via create_server_output,
    // so this exercises the "output already running" branch of apply_video_info
    // -- the exact path that used to be short-circuited by the early return.
    let mut f = Fixture::new();
    f.create_window(320, 240);

    // Drain the initial mode/geometry events so we only look at post-apply ones.
    f.client.get_output_events().clear();
    let configures_before = f.client.configure_count();

    apply(&mut f, 1920, 1080, 60);

    let mode = latest_mode_dimensions(f.client.get_output_events())
        .expect("expected a wl_output.mode event after apply");
    assert_eq!(
        (mode.0, mode.1),
        (1920, 1080),
        "client should observe the newly negotiated 1920x1080 mode",
    );
    assert!(
        mode.2 > 0,
        "refresh rate should be non-zero, got {}",
        mode.2
    );

    assert!(
        f.client.configure_count() > configures_before,
        "apply_video_info should trigger a fresh xdg_toplevel.Configure (was {}, now {})",
        configures_before,
        f.client.configure_count(),
    );
}

#[test]
fn repeated_video_info_propagates_each_change_to_client() {
    // Core regression guard: before the early-return was removed, the second
    // VideoInfo was silently dropped and the client stayed on the first mode.
    let mut f = Fixture::new();
    f.create_window(320, 240);

    for &(w, h, fps) in &[(1280, 720, 60), (2560, 1440, 60), (800, 600, 30)] {
        f.client.get_output_events().clear();
        let before = f.client.configure_count();

        apply(&mut f, w, h, fps);

        let mode = latest_mode_dimensions(f.client.get_output_events())
            .unwrap_or_else(|| panic!("no mode event after apply {}x{}@{}", w, h, fps));
        assert_eq!(
            (mode.0, mode.1),
            (w as i32, h as i32),
            "client should converge on {}x{} after apply",
            w,
            h
        );
        assert!(
            f.client.configure_count() > before,
            "apply {}x{}@{} should broadcast a new configure (was {}, now {})",
            w,
            h,
            fps,
            before,
            f.client.configure_count(),
        );
    }
}

#[test]
fn framerate_only_change_updates_mode_refresh() {
    // Same dimensions, different fps: mode.refresh must move; width/height stay.
    let mut f = Fixture::new();
    f.create_window(320, 240);
    apply(&mut f, 1920, 1080, 60);

    let baseline =
        latest_mode_dimensions(f.client.get_output_events()).expect("baseline mode missing");
    assert_eq!((baseline.0, baseline.1), (1920, 1080));
    let baseline_refresh = baseline.2;

    f.client.get_output_events().clear();
    apply(&mut f, 1920, 1080, 30);

    let after =
        latest_mode_dimensions(f.client.get_output_events()).expect("post-fps-change mode missing");
    assert_eq!((after.0, after.1), (1920, 1080));
    assert_ne!(
        after.2, baseline_refresh,
        "refresh rate should change when fps changes",
    );
}

#[test]
fn downscale_updates_mode_and_triggers_configure() {
    // Regression guard: after a large-to-small resize the client must observe a
    // mode event reflecting the new smaller output and a corresponding fresh
    // configure -- not get stuck on the previous larger mode.
    let mut f = Fixture::new();
    f.create_window(320, 240);
    apply(&mut f, 2560, 1440, 60);
    f.client.get_output_events().clear();
    let before = f.client.configure_count();

    apply(&mut f, 640, 480, 60);

    let mode = latest_mode_dimensions(f.client.get_output_events())
        .expect("mode event missing after downscale");
    assert_eq!((mode.0, mode.1), (640, 480));
    assert!(
        f.client.configure_count() > before,
        "downscale should trigger a configure",
    );
}

#[test]
fn idempotent_same_video_info_is_safe_and_stays_on_mode() {
    // Repeated same-params apply must not crash, re-map, or drift the client
    // off the target mode.
    let mut f = Fixture::new();
    f.create_window(320, 240);

    // Establish the target mode and clear everything before entering the loop,
    // so the assertion below only sees events produced by the idempotent calls.
    apply(&mut f, 1920, 1080, 60);
    f.client.get_output_events().clear();

    for _ in 0..4 {
        apply(&mut f, 1920, 1080, 60);
    }

    // Every mode event observed across the idempotent applies must describe
    // 1920x1080. Catches a regression where repeated applies would drift.
    let events = f.client.get_output_events();
    let mut saw_mode = false;
    for ev in events.iter() {
        if let wl_output::Event::Mode { width, height, .. } = ev {
            saw_mode = true;
            assert_eq!(
                (*width, *height),
                (1920, 1080),
                "stray mode event with unexpected dimensions",
            );
        }
    }
    assert!(
        saw_mode,
        "expected at least one wl_output.mode event across the idempotent applies",
    );
}

#[test]
fn apply_after_window_mapped_triggers_configure() {
    // A client that maps a window BEFORE apply_video_info runs should still
    // receive a configure when apply runs -- this is the specific path that
    // used to be broken by the early return.
    let mut f = Fixture::new();
    f.create_window(320, 240);
    let before = f.client.configure_count();

    apply(&mut f, 1280, 720, 60);

    assert!(
        f.client.configure_count() > before,
        "apply_video_info on a mapped toplevel should trigger at least one additional configure (was {}, now {})",
        before,
        f.client.configure_count(),
    );

    // And the client must see a wl_output.mode with the new dimensions.
    let mode = latest_mode_dimensions(f.client.get_output_events())
        .expect("expected wl_output.mode event");
    assert_eq!((mode.0, mode.1), (1280, 720));
}
