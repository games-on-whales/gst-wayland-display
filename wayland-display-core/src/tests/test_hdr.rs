//! Dynamic-HDR detection: what the compositor derives from clients' colour-management
//! state, driven through a real `wp_color_management_v1` client.
//!
//! Covers the properties the converter and the element rely on: HDR comes from a
//! surface's *declared* transfer function, latched on commit; it is read from the topmost
//! window (not whichever was mapped first); the committed output state only flips after the
//! debounce and is what picks the frame's shader; and a refusal pins the output to SDR.

use crate::Command;
use crate::comp::apply_video_info;
use crate::tests::fixture::Fixture;
use crate::utils::RenderTarget;
use crate::utils::color::{ContentTransfer, FrameColor};
use crate::utils::video_info::GstVideoInfo;
use smithay::wayland::seat::WaylandFocus;
use std::sync::mpsc;
use std::time::Duration;
use test_log::test;
use wayland_protocols::wp::color_management::v1::client::wp_color_manager_v1::{
    Primaries, TransferFunction,
};

/// Two mapped toplevels, like Steam (bottom) plus a game (top).
fn two_windows() -> Fixture {
    let mut f = Fixture::with_hdr(true);
    f.create_window(320, 240);
    f.create_window(320, 240);
    assert_eq!(f.server.space.elements().count(), 2);
    f
}

fn set_pq(f: &mut Fixture, idx: usize, mastering_primaries: Option<[i32; 8]>) {
    f.client.set_image_description(
        idx,
        TransferFunction::St2084Pq,
        Primaries::Bt2020,
        mastering_primaries,
    );
    f.round_trip();
}

fn hdr_states(rx: &mpsc::Receiver<Command>) -> Vec<(bool, Option<String>)> {
    rx.try_iter()
        .filter_map(|cmd| match cmd {
            Command::HdrState { hdr, mastering, .. } => Some((hdr, mastering)),
            _ => None,
        })
        .collect()
}

#[test]
fn declared_pq_takes_effect_on_commit() {
    let mut f = two_windows();
    set_pq(&mut f, 1, None);
    // Staged but not committed: the surface's current state is still undeclared.
    assert!(!f.server.output_hdr_state());

    f.client.commit_window(1);
    f.round_trip();
    assert!(f.server.output_hdr_state());
}

#[test]
fn hdr_state_comes_from_the_topmost_window() {
    let mut f = two_windows();
    // The bottom window declaring PQ must not make the output HDR.
    set_pq(&mut f, 0, None);
    f.client.commit_window(0);
    f.round_trip();
    assert!(!f.server.output_hdr_state());

    set_pq(&mut f, 1, None);
    f.client.commit_window(1);
    f.round_trip();
    assert!(f.server.output_hdr_state());
}

#[test]
fn debounced_flip_drives_frame_color_and_refusal_pins_sdr() {
    let mut f = two_windows();
    let (tx, rx) = mpsc::channel();
    f.server.hdr_state_tx = Some(tx);

    set_pq(&mut f, 1, None);
    f.client.commit_window(1);
    f.round_trip();

    // Before the debounce elapses the frame stays tagged SDR, so PQ content must be
    // converted for an SDR output (tone-mapped by the compositor), never passed through.
    f.server.update_hdr_state();
    assert!(hdr_states(&rx).is_empty());
    f.server.create_frame().expect("frame");
    assert_eq!(
        f.server.last_frame_color,
        FrameColor {
            output_hdr: false,
            content: ContentTransfer::Sdr
        }
    );

    std::thread::sleep(Duration::from_millis(700));
    f.server.update_hdr_state();
    assert_eq!(hdr_states(&rx), vec![(true, None)]);
    assert!(f.server.output_hdr());
    f.server.create_frame().expect("frame");
    assert_eq!(
        f.server.last_frame_color,
        FrameColor {
            output_hdr: true,
            content: ContentTransfer::Pq
        }
    );

    // Downstream refused HDR: the very next frame is SDR, signalled once, and it stays so.
    f.server.refuse_hdr();
    assert_eq!(hdr_states(&rx), vec![(false, None)]);
    assert!(!f.server.output_hdr_state());
    assert!(!f.server.output_hdr());
    std::thread::sleep(Duration::from_millis(700));
    f.server.update_hdr_state();
    f.server.update_hdr_state();
    assert!(hdr_states(&rx).is_empty());
}

#[test]
fn unset_mastering_luminance_defaults_to_the_pq_primary_volume() {
    let mut f = two_windows();
    let (tx, rx) = mpsc::channel();
    f.server.hdr_state_tx = Some(tx);

    // Target primaries only: the target luminance must default to the PQ primary volume
    // (0.005 .. 10000 cd/m²), not to a zero maximum.
    let bt2020 = [
        708000, 292000, 170000, 797000, 131000, 46000, 312700, 329000,
    ];
    set_pq(&mut f, 1, Some(bt2020));
    f.client.commit_window(1);
    f.round_trip();

    f.server.update_hdr_state();
    std::thread::sleep(Duration::from_millis(700));
    f.server.update_hdr_state();
    let states = hdr_states(&rx);
    let mastering = states[0].1.as_deref().expect("mastering metadata");
    // gst units: chromaticity * 50000, then max:min luminance * 10000 cd/m².
    assert_eq!(
        mastering,
        "35400:14600:8500:39850:6550:2300:15635:16450:100000000:50"
    );
}

#[test]
fn colorimetry_only_renegotiation_leaves_clients_and_pointer_alone() {
    let mut f = two_windows();
    // Establish a negotiated contract (the fixture's initial output is built manually).
    let initial = f.server.video_info.as_ref().unwrap().clone();
    apply_video_info(
        &mut f.server,
        GstVideoInfo::RAW(initial),
        &RenderTarget::Software,
        None,
    );
    f.round_trip();
    f.server.set_pointer_location((10.0, 20.0).into());
    let configures = f.client.configure_count();

    // What the element sends on an HDR flip: same format/size/fps, new colorimetry.
    let pq: gst_video::VideoColorimetry = "bt2100-pq".parse().unwrap();
    let info = gst_video::VideoInfo::builder(gst_video::VideoFormat::Rgba, 320, 240)
        .colorimetry(&pq)
        .build()
        .unwrap();
    apply_video_info(
        &mut f.server,
        GstVideoInfo::RAW(info),
        &RenderTarget::Software,
        None,
    );
    f.round_trip();
    f.round_trip();

    assert_eq!(f.server.pointer_location, (10.0, 20.0).into());
    assert_eq!(f.client.configure_count(), configures);
    assert_eq!(
        f.server
            .video_info
            .as_ref()
            .unwrap()
            .colorimetry()
            .transfer(),
        gst_video::VideoTransferFunction::Smpte2084
    );
}

#[test]
fn hdr_metadata_removal_restores_defaults_without_a_mode_flip() {
    use crate::wayland::handlers::color_management::set_surface_hdr_color;
    let mut f = two_windows();
    let (tx, rx) = mpsc::channel();
    f.server.hdr_state_tx = Some(tx);
    set_pq(
        &mut f,
        1,
        Some([
            708000, 292000, 170000, 797000, 131000, 46000, 312700, 329000,
        ]),
    );
    f.client.commit_window(1);
    f.round_trip();
    f.server.update_hdr_state();
    std::thread::sleep(Duration::from_millis(700));
    f.server.update_hdr_state();
    assert!(hdr_states(&rx)[0].1.is_some());

    let surface = f
        .server
        .space
        .elements()
        .last()
        .unwrap()
        .wl_surface()
        .unwrap()
        .into_owned();
    set_surface_hdr_color(
        &surface,
        Some(
            crate::wayland::handlers::color_management::SurfaceHdrColor {
                transfer: ContentTransfer::Pq,
                is_bt2020: true,
                ..Default::default()
            },
        ),
    );
    f.client.commit_window(1);
    f.round_trip();
    f.server.update_hdr_state();
    assert_eq!(hdr_states(&rx), vec![(true, None)]);
    assert!(f.server.output_hdr());
}

#[test]
fn color_state_waits_for_the_buffer_acquire_blocker() {
    use smithay::reexports::wayland_server::Resource;
    use smithay::wayland::compositor::CompositorHandler;
    use smithay::wayland::compositor::{Barrier, add_blocker, add_pre_commit_hook};
    let mut f = two_windows();
    let surface = f
        .server
        .space
        .elements()
        .last()
        .unwrap()
        .wl_surface()
        .unwrap()
        .into_owned();
    let barrier = Barrier::new(false);
    let pending = barrier.clone();
    add_pre_commit_hook::<crate::comp::State, _>(&surface, move |_, _, surface| {
        add_blocker(surface, pending.clone());
    });
    set_pq(&mut f, 1, None);
    f.client.commit_window(1);
    f.round_trip();
    assert!(
        !f.server.output_hdr_state(),
        "blocked buffer must keep its old color state"
    );
    barrier.signal();
    let client = surface.client().unwrap();
    let dh = f.server.dh.clone();
    f.server
        .client_compositor_state(&client)
        .blocker_cleared(&mut f.server, &dh);
    assert!(
        f.server.output_hdr_state(),
        "releasing the buffer must release its color state"
    );
}

#[test]
fn renegotiating_the_output_lifts_a_refusal() {
    let mut f = two_windows();
    set_pq(&mut f, 1, None);
    f.client.commit_window(1);
    f.round_trip();
    f.server.refuse_hdr();
    assert!(!f.server.output_hdr_state());

    // A refusal belongs to the output it was made for (e.g. NV12 caps, which cannot carry
    // HDR); renegotiated caps may accept HDR.
    let info = gst_video::VideoInfo::builder(gst_video::VideoFormat::Rgba, 640, 480)
        .fps(gst::Fraction::new(60, 1))
        .build()
        .unwrap();
    apply_video_info(
        &mut f.server,
        GstVideoInfo::RAW(info),
        &RenderTarget::Software,
        None,
    );
    assert!(f.server.output_hdr_state());
}

#[test]
fn hdr_texture_shaders_compile() {
    let mut f = Fixture::with_hdr(true);
    assert!(
        f.server.hdr_programs().is_some(),
        "the GLES driver must accept the HDR texture shaders"
    );
}
