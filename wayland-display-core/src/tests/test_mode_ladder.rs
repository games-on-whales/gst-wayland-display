//! The `wl_output` mode ladder: extra advertised modes so an in-app display/resolution
//! menu has something to list, without moving the mode the compositor actually composites
//! at (that stays the render size).

use crate::comp::{apply_mode_ladder, apply_render_size, apply_video_info};
use crate::tests::fixture::Fixture;
use crate::tests::test_resolution::make_video_info;
use crate::utils::RenderTarget;
use test_log::test;
use wayland_client::protocol::wl_output;

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

/// Every `wl_output::mode` the client has seen, as `(width, height, is_current)`.
fn modes(f: &mut Fixture) -> Vec<(i32, i32, bool)> {
    f.client
        .get_output_events()
        .iter()
        .filter_map(|e| match e {
            wl_output::Event::Mode {
                flags,
                width,
                height,
                ..
            } => Some((
                *width,
                *height,
                flags
                    .into_result()
                    .map(|f| f.contains(wl_output::Mode::Current))
                    .unwrap_or(false),
            )),
            _ => None,
        })
        .collect()
}

/// The ladder is advertised at **bind** time, which is the only moment `wl_output` sends a
/// mode list — so this uses a cold fixture (no Output until the first `apply_video_info`),
/// which is also the production ordering: the element forwards the ladder before/with the
/// caps, the app container connects afterwards.
#[test]
fn mode_ladder_is_advertised_on_wl_output() {
    let mut f = Fixture::new_cold();

    // 2560x1440 is above the encode size and must be dropped; the other three are offered.
    apply_mode_ladder(
        &mut f.server,
        &[(2560, 1440), (1920, 1080), (1280, 720), (960, 540)],
    );
    apply_encode(&mut f, 1920, 1080, 60);

    let modes = modes(&mut f);
    let dims: Vec<(i32, i32)> = modes.iter().map(|&(w, h, _)| (w, h)).collect();
    assert_eq!(
        dims,
        vec![(1920, 1080), (1280, 720), (960, 540)],
        "the client should see exactly the ladder rungs that fit inside the encode size",
    );
    assert_eq!(
        modes
            .iter()
            .filter(|&&(_, _, current)| current)
            .map(|&(w, h, _)| (w, h))
            .collect::<Vec<_>>(),
        vec![(1920, 1080)],
        "the CURRENT flag must be on the internal (render) size, and on nothing else",
    );
}

/// The ladder is advisory: it must not disturb the mode the compositor composites at, nor
/// the encode size.
#[test]
fn mode_ladder_does_not_move_the_current_mode() {
    let mut f = Fixture::new_cold();

    apply_mode_ladder(&mut f.server, &[(1280, 720), (960, 540)]);
    apply_encode(&mut f, 1920, 1080, 60);

    let current = f.server.output.as_ref().unwrap().current_mode().unwrap();
    assert_eq!((current.size.w, current.size.h), (1920, 1080));
    let vi = f.server.video_info.as_ref().unwrap();
    assert_eq!((vi.width(), vi.height()), (1920, 1080));
}

/// A ladder set *after* the encode size is known is applied straight away, and re-applying
/// the same ladder is a no-op (the element re-forwards it on every caps re-negotiation).
#[test]
fn mode_ladder_after_caps_is_applied_and_is_idempotent() {
    let mut f = Fixture::new_cold();
    apply_encode(&mut f, 1920, 1080, 60);

    apply_mode_ladder(&mut f.server, &[(1280, 720), (960, 540)]);
    let after_first: Vec<_> = f
        .server
        .output
        .as_ref()
        .unwrap()
        .modes()
        .iter()
        .map(|m| (m.size.w, m.size.h))
        .collect();
    assert!(after_first.contains(&(1280, 720)) && after_first.contains(&(960, 540)));

    apply_mode_ladder(&mut f.server, &[(1280, 720), (960, 540)]);
    let after_second: Vec<_> = f
        .server
        .output
        .as_ref()
        .unwrap()
        .modes()
        .iter()
        .map(|m| (m.size.w, m.size.h))
        .collect();
    assert_eq!(after_first, after_second, "a re-apply must not duplicate");
}

/// Clearing the ladder retires the rungs from the Output's list, except the current /
/// preferred mode — deleting that would leave the output modeless.
#[test]
fn clearing_the_ladder_retires_its_rungs() {
    let mut f = Fixture::new_cold();
    apply_mode_ladder(&mut f.server, &[(1920, 1080), (1280, 720), (960, 540)]);
    apply_encode(&mut f, 1920, 1080, 60);

    apply_mode_ladder(&mut f.server, &[]);

    let output = f.server.output.as_ref().unwrap();
    let dims: Vec<_> = output
        .modes()
        .iter()
        .map(|m| (m.size.w, m.size.h))
        .collect();
    assert_eq!(
        dims,
        vec![(1920, 1080)],
        "only the current mode should be left",
    );
    assert_eq!(
        output.current_mode().map(|m| (m.size.w, m.size.h)),
        Some((1920, 1080)),
        "the current mode must survive the retirement pass",
    );
}

/// `Output::change_current_state` APPENDS every mode it is handed and never removes the one
/// it replaced, so without an explicit sweep every render size the session has ever used
/// stays advertised forever. Invisible while nothing listed the modes; with a ladder, an
/// in-app display menu shows the lot.
#[test]
fn superseded_render_sizes_do_not_accumulate_as_modes() {
    let mut f = Fixture::new_cold();
    apply_encode(&mut f, 1920, 1080, 60);

    apply_render_size(&mut f.server, (1280, 1080).into());
    apply_render_size(&mut f.server, (1920, 1080).into());

    let dims: Vec<_> = f
        .server
        .output
        .as_ref()
        .unwrap()
        .modes()
        .iter()
        .map(|m| (m.size.w, m.size.h))
        .collect();
    assert_eq!(
        dims,
        vec![(1920, 1080)],
        "a superseded render size must not linger as an advertised mode",
    );
}

/// ... but a superseded render size that IS a ladder rung stays, because the ladder wants it.
#[test]
fn a_superseded_render_size_that_is_a_ladder_rung_is_kept() {
    let mut f = Fixture::new_cold();
    apply_mode_ladder(&mut f.server, &[(1280, 720)]);
    apply_encode(&mut f, 1920, 1080, 60);

    apply_render_size(&mut f.server, (1280, 720).into());
    apply_render_size(&mut f.server, (1920, 1080).into());

    let dims: Vec<_> = f
        .server
        .output
        .as_ref()
        .unwrap()
        .modes()
        .iter()
        .map(|m| (m.size.w, m.size.h))
        .collect();
    assert!(dims.contains(&(1280, 720)), "got {dims:?}");
    assert!(dims.contains(&(1920, 1080)), "got {dims:?}");
    assert_eq!(dims.len(), 2, "and nothing else: {dims:?}");
}

/// A rung that fit the previous encode size but not the new one is retired on the next caps
/// negotiation (and comes back if the encode size grows again).
#[test]
fn ladder_rungs_follow_the_encode_size_across_renegotiation() {
    let mut f = Fixture::new_cold();
    apply_mode_ladder(&mut f.server, &[(1920, 1080), (1280, 720)]);
    apply_encode(&mut f, 1920, 1080, 60);

    apply_encode(&mut f, 1280, 720, 60);
    let dims: Vec<_> = f
        .server
        .output
        .as_ref()
        .unwrap()
        .modes()
        .iter()
        .map(|m| (m.size.w, m.size.h))
        .collect();
    assert!(
        !dims.contains(&(1920, 1080)),
        "a rung above the new encode size must be retired, got {dims:?}",
    );
    assert!(dims.contains(&(1280, 720)));
}
