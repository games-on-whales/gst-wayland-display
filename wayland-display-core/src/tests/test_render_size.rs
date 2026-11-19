//! Render size (the app-facing `wl_output` mode) is decoupled from the encode size
//! carried by the negotiated caps, and is sticky across caps re-negotiation.

use crate::comp::{apply_render_size, apply_ui_scale, apply_video_info};
use crate::tests::fixture::Fixture;
use crate::tests::test_resolution::{latest_mode_dimensions, make_video_info};
use crate::utils::RenderTarget;
use smithay::input::pointer::CursorImageStatus;
use smithay::utils::Point;
use test_log::test;
use wayland_client::protocol::wl_output;

const WHITE: u32 = 0x00FF_FFFF;

fn apply_encode(f: &mut Fixture, width: u32, height: u32, fps: i32) {
    apply_video_info(
        &mut f.server,
        make_video_info(width, height, fps),
        &RenderTarget::Software,
        None,
    );
    f.round_trip();
    f.round_trip();
}

fn apply_render(f: &mut Fixture, width: i32, height: i32) {
    apply_render_size(&mut f.server, (width, height).into());
    f.round_trip();
    f.round_trip();
}

fn apply_scale(f: &mut Fixture, scale: f64) {
    apply_ui_scale(&mut f.server, scale);
    f.round_trip();
    f.round_trip();
}

fn mode_dimensions(f: &mut Fixture) -> Option<(i32, i32)> {
    latest_mode_dimensions(f.client.get_output_events()).map(|(w, h, _)| (w, h))
}

/// The most recent `wl_output::Event::Scale`. Integer, per the protocol: smithay sends
/// `ceil(fractional)` here and the exact value via `wp_fractional_scale_v1`.
fn output_scale_event(f: &mut Fixture) -> Option<i32> {
    f.client
        .get_output_events()
        .iter()
        .rev()
        .find_map(|e| match e {
            wl_output::Event::Scale { factor } => Some(*factor),
            _ => None,
        })
}

#[test]
fn render_size_changes_wl_output_mode_but_not_encode_size() {
    let mut f = Fixture::new();
    f.create_window(320, 240);

    apply_encode(&mut f, 1920, 1080, 60);
    assert_eq!(mode_dimensions(&mut f), Some((1920, 1080)));

    f.client.get_output_events().clear();
    apply_render(&mut f, 1280, 720);
    assert_eq!(
        mode_dimensions(&mut f),
        Some((1280, 720)),
        "clients should observe the requested render size as the output mode",
    );

    // encode side untouched
    let vi = f.server.video_info.as_ref().unwrap();
    assert_eq!(
        (vi.width(), vi.height()),
        (1920, 1080),
        "render size must not disturb the negotiated encode size",
    );
}

#[test]
fn render_size_survives_video_info_reapply() {
    let mut f = Fixture::new();
    f.create_window(320, 240);

    apply_encode(&mut f, 1920, 1080, 60);
    apply_render(&mut f, 1280, 720);

    // re-negotiation of the same encode caps must NOT reset render size
    f.client.get_output_events().clear();
    apply_encode(&mut f, 1920, 1080, 60);

    assert_eq!(mode_dimensions(&mut f), Some((1280, 720)));
    let vi = f.server.video_info.as_ref().unwrap();
    assert_eq!((vi.width(), vi.height()), (1920, 1080));
}

#[test]
fn render_size_larger_than_encode_is_clamped() {
    let mut f = Fixture::new();
    f.create_window(320, 240);

    apply_encode(&mut f, 1280, 720, 60);
    f.client.get_output_events().clear();
    apply_render(&mut f, 1920, 1080);

    assert_eq!(
        mode_dimensions(&mut f),
        Some((1280, 720)),
        "a render size larger than the encode size is clamped to the encode size",
    );
}

#[test]
fn zero_render_size_returns_to_following_the_encode_size() {
    let mut f = Fixture::new();
    f.create_window(320, 240);

    apply_encode(&mut f, 1920, 1080, 60);
    apply_render(&mut f, 1280, 720);
    assert_eq!(mode_dimensions(&mut f), Some((1280, 720)));

    f.client.get_output_events().clear();
    apply_render(&mut f, 0, 0);
    assert_eq!(
        mode_dimensions(&mut f),
        Some((1920, 1080)),
        "a (0,0) render size means 'follow the encode size'",
    );
    assert!(f.server.render_size.is_none());
}

#[test]
fn render_size_before_video_info_is_applied_on_negotiation() {
    let mut f = Fixture::new();
    f.create_window(320, 240);

    apply_render(&mut f, 1280, 720);
    f.client.get_output_events().clear();
    apply_encode(&mut f, 1920, 1080, 60);

    assert_eq!(
        mode_dimensions(&mut f),
        Some((1280, 720)),
        "a render size requested before caps negotiation must survive it",
    );
}

#[test]
fn ui_scale_is_sent_as_preferred_scale_to_mapped_toplevel() {
    let mut f = Fixture::new();
    let surface = f.create_window_with_fractional_scale(320, 240);
    apply_encode(&mut f, 1920, 1080, 60);

    // The scale is announced at wp_fractional_scale_v1 creation time, at the default 1.0.
    assert_eq!(f.client.last_preferred_scale(&surface), Some(120));
    assert_eq!(mode_dimensions(&mut f), Some((1920, 1080)));

    let before = f.client.configure_count();
    f.client.get_output_events().clear();
    apply_ui_scale(&mut f.server, 2.0);
    f.round_trip();
    f.round_trip();

    assert_eq!(
        f.client.last_preferred_scale(&surface),
        Some(240),
        "the UI scale must reach the client as preferred_scale in 1/120ths",
    );
    assert!(
        f.client.configure_count() > before,
        "a UI-scale change must be followed by a (non-empty) configure (was {}, now {})",
        before,
        f.client.configure_count(),
    );
    assert!(
        f.client.configure_count() >= 2,
        "the toplevel should have been configured at least at map and at the scale change",
    );

    // The PHYSICAL mode does not move -- the scale is what shrinks the logical size, so the
    // composite density (and therefore the encode framebuffer) is untouched.
    assert_eq!(
        mode_dimensions(&mut f),
        Some((1920, 1080)),
        "a UI-scale change must not move the physical wl_output mode",
    );
    assert_eq!(
        output_scale_event(&mut f),
        Some(2),
        "legacy clients must see the UI scale as the (integer) wl_output scale",
    );
    assert_eq!(
        f.client.last_configure_size(),
        Some((960, 540)),
        "the logical size the toplevel is configured at is mode / ui_scale",
    );

    // ... and the encode size is untouched.
    let vi = f.server.video_info.as_ref().unwrap();
    assert_eq!((vi.width(), vi.height()), (1920, 1080));
}

#[test]
fn ui_scale_shrinks_logical_mode_and_keeps_scene_full_frame() {
    let mut f = Fixture::new();
    hide_cursor(&mut f);

    // Encode 1920x1080, render 1280x720, UI scale 2.
    apply_encode(&mut f, 1920, 1080, 60);
    apply_render(&mut f, 1280, 720);
    f.client.get_output_events().clear();
    apply_scale(&mut f, 2.0);

    // The PHYSICAL mode stays the render size; the scale is what makes the desktop logically
    // half as big, so the UI grows without the framebuffer or the encode size changing.
    assert_eq!(mode_dimensions(&mut f), Some((1280, 720)));
    assert_eq!(output_scale_event(&mut f), Some(2));

    // A HiDPI-aware client: 640x360 logical, backed by a 1280x720 buffer.
    f.create_solid_window_hidpi(1280, 720, 640, 360, WHITE);
    assert_eq!(
        f.client.last_configure_size(),
        Some((640, 360)),
        "the toplevel must be configured at the LOGICAL size (render / ui_scale)",
    );

    // That 1280x720 buffer is composited at the render density (1:1, not downsampled to the
    // 640x360 logical size) and then upscaled 1.5x into the encode framebuffer, which it
    // fills edge to edge.
    let (px, stride) = frame_pixels(&mut f);
    assert_lit(&px, stride, 10, 10);
    assert_lit(&px, stride, 1900, 1000);
}

/// Production cold start: `State::new` leaves no Output, no mode and no video_info, so the
/// first `apply_video_info` creates the Output and sets its very first mode. That call has
/// **no previous logical extent** to remap the pointer from, and `clamp_coords` runs against
/// an Output whose mode was set moments earlier. The seeded [`Fixture::new`] hides both cases
/// behind its 320x240 seed, so nothing covered this ordering until now.
#[test]
fn cold_start_first_mode_set_is_safe_without_a_previous_mode() {
    let mut f = Fixture::new_cold();
    assert!(f.server.output.is_none(), "precondition: no Output yet");
    assert!(f.server.video_info.is_none());

    // The one call that creates the Output and sets the first mode.
    apply_encode(&mut f, 1920, 1080, 60);

    let p = f.server.pointer_location;
    assert!(
        p.x.is_finite() && p.y.is_finite(),
        "an empty/absent previous extent must not divide by zero into a NaN pointer, got {p:?}",
    );
    assert_eq!(
        (p.x, p.y),
        (960.0, 540.0),
        "with no previous extent the pointer centres, as it always did",
    );
    assert_eq!(mode_dimensions(&mut f), Some((1920, 1080)));

    // Input and a real frame (cursor included -- it is built with the output scale) must both
    // survive the freshly-created Output.
    f.server
        .pointer_motion_absolute(0, Point::from((10.0, 10.0)));
    let (px, _stride) = frame_pixels(&mut f);
    assert!(!px.is_empty());

    // A second mode set now *does* have a previous extent: 1920x1080 -> 1280x720.
    apply_render(&mut f, 1280, 720);
    let p = f.server.pointer_location;
    assert!(p.x.is_finite() && p.y.is_finite(), "got {p:?}");
}

/// Same cold ordering, but the UI scale is requested *before* any caps arrive (the session
/// start Task 4 forwards): `apply_ui_scale` must not touch the mode path with no Output, and
/// the scale must still be in force once the first `apply_video_info` creates one.
#[test]
fn cold_start_ui_scale_before_any_video_info() {
    let mut f = Fixture::new_cold();

    apply_ui_scale(&mut f.server, 2.0);
    assert!(f.server.output.is_none(), "still no Output to configure");

    apply_encode(&mut f, 1920, 1080, 60);

    assert_eq!(output_scale_event(&mut f), Some(2));
    let p = f.server.pointer_location;
    assert!(p.x.is_finite() && p.y.is_finite(), "got {p:?}");
    assert_eq!(
        (p.x, p.y),
        (480.0, 270.0),
        "centre of the 960x540 LOGICAL extent",
    );
}

/// `forward_display_geometry` re-sends the render size after every `Command::VideoInfo`, on
/// both set_caps arms — so an unchanged render size arrives on every caps renegotiation,
/// i.e. on every ABR resolution step. Each redundant apply would otherwise re-run the whole
/// mode path: a second `change_current_state`, a second damage-tracker rebuild (a full-damage
/// frame) and a second configure to every toplevel.
#[test]
fn a_redundant_render_size_apply_is_a_no_op() {
    let mut f = Fixture::new();
    f.create_window(320, 240);
    apply_encode(&mut f, 1920, 1080, 60);
    apply_render(&mut f, 1280, 720);

    let before = f.client.configure_count();
    f.client.get_output_events().clear();
    apply_render(&mut f, 1280, 720);

    assert_eq!(
        f.client.configure_count(),
        before,
        "re-applying the SAME render size must not reconfigure the toplevel",
    );
    assert!(
        f.client.get_output_events().is_empty(),
        "... nor re-send any wl_output state: {:?}",
        f.client.get_output_events(),
    );

    // ... but a real change still does.
    apply_render(&mut f, 960, 540);
    assert!(f.client.configure_count() > before);
    assert_eq!(mode_dimensions(&mut f), Some((960, 540)));
}

/// The element re-sends the UI scale after every render-size change, so a redundant apply is
/// the common case, not an edge case. Under design 2 it would otherwise re-run the whole mode
/// path (new damage tracker + a configure to every toplevel) for no reason.
#[test]
fn a_redundant_ui_scale_apply_is_a_no_op() {
    let mut f = Fixture::new();
    f.create_window(320, 240);
    apply_encode(&mut f, 1920, 1080, 60);
    apply_scale(&mut f, 2.0);

    let before = f.client.configure_count();
    apply_scale(&mut f, 2.0);
    assert_eq!(
        f.client.configure_count(),
        before,
        "re-applying the SAME scale must not reconfigure the toplevel",
    );

    // ... but a real change still does.
    apply_scale(&mut f, 1.5);
    assert!(f.client.configure_count() > before);
}

#[test]
fn mode_change_remaps_the_pointer_instead_of_recentring_it() {
    let mut f = Fixture::new();
    apply_encode(&mut f, 1920, 1080, 60);

    // Put the cursor somewhere distinctly off-centre: 1/4 across, 3/4 down.
    f.server.set_pointer_location(Point::from((480.0, 810.0)));

    // Halve the logical extent via the UI scale: 1920x1080 -> 960x540.
    apply_scale(&mut f, 2.0);

    let p = f.server.pointer_location;
    assert!(
        (p.x - 240.0).abs() < 1.0 && (p.y - 405.0).abs() < 1.0,
        "the pointer must keep its RELATIVE position across a scale change \
         (expected ~(240, 405), got {p:?})",
    );
    assert_ne!(
        (p.x, p.y),
        (480.0, 270.0),
        "and must not be teleported to the centre of the new extent",
    );
}

#[test]
fn ui_scale_reaches_a_toplevel_still_pending_its_initial_configure() {
    let mut f = Fixture::new();

    // Created + committed, but the initial configure is not acked yet: the toplevel lives
    // in `pending_windows`, not in the space. This is the window a session-start UiScale
    // lands in.
    let surface = f.client.map_toplevel_with_fractional_scale();
    f.round_trip();
    assert_eq!(f.client.last_preferred_scale(&surface), Some(120));
    assert!(
        f.server.space.elements().next().is_none(),
        "precondition: the toplevel must not be mapped yet",
    );

    apply_ui_scale(&mut f.server, 2.0);
    f.round_trip();

    // Completing the map must not lose it either.
    f.finish_window(320, 240);

    assert_eq!(
        f.client.last_preferred_scale(&surface),
        Some(240),
        "a UiScale arriving before the initial configure must still reach the client",
    );
}

#[test]
fn ui_scale_out_of_range_is_clamped() {
    let mut f = Fixture::new();

    apply_ui_scale(&mut f.server, 7.5);
    assert_eq!(f.server.ui_scale, 3.0);

    apply_ui_scale(&mut f.server, 0.2);
    assert_eq!(f.server.ui_scale, 1.0);
}

#[test]
fn render_size_change_triggers_a_configure() {
    let mut f = Fixture::new();
    f.create_window(320, 240);
    apply_encode(&mut f, 1920, 1080, 60);

    let before = f.client.configure_count();
    apply_render(&mut f, 1280, 720);

    assert!(
        f.client.configure_count() > before,
        "a render-size change should re-configure mapped toplevels (was {}, now {})",
        before,
        f.client.configure_count(),
    );
}

// ---------------------------------------------------------------------------------------
// Compositing: the scene is built at the render size and upscaled into the encode-sized
// framebuffer (aspect-preserving, centred). These render a real frame through the
// `RenderTarget::Software` path and read the RAW gst::Buffer back.
// ---------------------------------------------------------------------------------------

/// Render one frame and return `(rgba_bytes, stride)` of the encode-sized framebuffer.
fn frame_pixels(f: &mut Fixture) -> (Vec<u8>, usize) {
    let stride = f.server.video_info.as_ref().unwrap().stride()[0] as usize;
    let (buffer, _result) = f.server.create_frame().expect("create_frame failed");
    let map = buffer.map_readable().expect("failed to map frame buffer");
    (map.as_slice().to_vec(), stride)
}

fn pixel(px: &[u8], stride: usize, x: usize, y: usize) -> [u8; 3] {
    let i = y * stride + x * 4;
    [px[i], px[i + 1], px[i + 2]]
}

#[track_caller]
fn assert_lit(px: &[u8], stride: usize, x: usize, y: usize) {
    let p = pixel(px, stride, x, y);
    assert!(
        p.iter().all(|&c| c > 200),
        "expected the scaled scene (white) at ({x},{y}), got {p:?}",
    );
}

#[track_caller]
fn assert_dark(px: &[u8], stride: usize, x: usize, y: usize) {
    let p = pixel(px, stride, x, y);
    assert!(
        p.iter().all(|&c| c < 50),
        "expected letterbox/clear (black) at ({x},{y}), got {p:?}",
    );
}

/// The cursor is composited *inside* the scaled scene, so it would land on the sample
/// points below. These tests are about the scene transform, so take it out of the frame.
fn hide_cursor(f: &mut Fixture) {
    f.server.cursor_state = CursorImageStatus::Hidden;
}

#[test]
fn render_size_scene_is_upscaled_into_the_encode_framebuffer() {
    let mut f = Fixture::new();
    hide_cursor(&mut f);

    // Exact 2x: 960x540 render into a 1920x1080 encode framebuffer, no bars.
    apply_encode(&mut f, 1920, 1080, 60);
    apply_render(&mut f, 960, 540);
    f.create_solid_window(960, 540, WHITE);

    let (px, stride) = frame_pixels(&mut f);
    // Before the upscale existed the scene was drawn 1:1 into the top-left, so (1900,1000)
    // was the black clear colour.
    assert_lit(&px, stride, 1900, 1000);
    assert_lit(&px, stride, 10, 10);
}

#[test]
fn render_size_with_a_mismatched_aspect_is_pillarboxed() {
    let mut f = Fixture::new();
    hide_cursor(&mut f);

    // 1280x1080 into 1920x1080: min(1.5, 1.0) = 1.0, so 320px black bars left and right.
    apply_encode(&mut f, 1920, 1080, 60);
    apply_render(&mut f, 1280, 1080);
    f.create_solid_window(1280, 1080, WHITE);

    let (px, stride) = frame_pixels(&mut f);
    assert_dark(&px, stride, 10, 540);
    assert_lit(&px, stride, 960, 540);
    assert_dark(&px, stride, 1910, 540);
}

#[test]
fn no_render_size_composites_one_to_one() {
    let mut f = Fixture::new();
    hide_cursor(&mut f);

    // No render size => scale 1.0, origin (0,0). The rescale/relocate wrappers must be an
    // exact identity: a 960x540 window still occupies only the top-left of a 1920x1080
    // framebuffer, exactly as before this change.
    apply_encode(&mut f, 1920, 1080, 60);
    f.create_solid_window(960, 540, WHITE);

    let (px, stride) = frame_pixels(&mut f);
    assert_lit(&px, stride, 10, 10);
    assert_lit(&px, stride, 950, 530);
    assert_dark(&px, stride, 970, 530);
    assert_dark(&px, stride, 950, 550);
    assert_dark(&px, stride, 1900, 1000);
}

#[test]
fn pointer_motion_absolute_clamps_to_the_render_extent() {
    let mut f = Fixture::new();
    f.create_window(320, 240);

    apply_encode(&mut f, 1920, 1080, 60);
    apply_render(&mut f, 960, 540);

    // `Command::PointerMotionAbsolute` (the FFI entry point) carries output-space coordinates
    // and `clamp_coords` clamps them to the *output mode* -- which is the render size, not the
    // encode size. This is the clamp half of the contract.
    //
    // The other producer of absolute motion, libinput's `InputEvent::PointerMotionAbsolute`
    // (`comp/input.rs:507-521`), is correct by construction and not re-asserted here: it
    // renormalises the device's 0..1 position against `output.current_mode()` via
    // `x_transformed`/`y_transformed` before calling into the same function, so it lands in
    // render space for the same reason the clamp does -- both read the current output mode.
    f.server
        .pointer_motion_absolute(0, Point::from((1920.0, 1080.0)));
    f.round_trip();

    assert_eq!(
        f.server.pointer_location,
        Point::from((958.0, 538.0)),
        "absolute pointer input must clamp into the render extent, not the encode extent",
    );
}
