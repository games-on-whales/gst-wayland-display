//! HDR state across a window's surface tree: a browser keeps its root UI sRGB while it
//! presents PQ video on a subsurface. The output must follow the PQ child (with its
//! metadata) through repeated sRGB root commits, return to SDR when the child's
//! declaration is cleared, and the frame must be composited in PQ while any surface is PQ.

use crate::tests::fixture::Fixture;
use crate::utils::color::ContentTransfer;
use crate::wayland::handlers::color_management::{
    MasteringDisplayInfo, SurfaceHdrColor, set_surface_hdr_color,
};
use smithay::wayland::seat::WaylandFocus;
use test_log::test;

#[test]
fn pq_child_survives_sdr_root_commits_and_clearing_it_restores_sdr() {
    let mut f = Fixture::with_hdr(true);
    f.create_window(320, 240);
    let child_client = f.client.create_subsurface(0, 0, 0, 160, 120);
    f.round_trip();

    let window = f.server.space.elements().last().unwrap();
    let root = window.wl_surface().unwrap().into_owned();
    let mut children = Vec::new();
    window.with_surfaces(|surface, _| {
        if *surface != root {
            children.push(surface.clone());
        }
    });
    assert_eq!(children.len(), 1, "must exercise a real Wayland subsurface");
    let child = children.pop().unwrap();

    set_surface_hdr_color(
        &child,
        Some(SurfaceHdrColor {
            transfer: ContentTransfer::Pq,
            is_bt2020: true,
            mastering: Some(MasteringDisplayInfo {
                primaries: [34000, 16000, 13250, 34500, 7500, 3000, 15635, 16450],
                max_luminance: 10_000_000,
                min_luminance: 1,
            }),
            max_cll: Some(1000),
            max_fall: Some(400),
        }),
    );
    set_surface_hdr_color(&root, Some(SurfaceHdrColor::default()));
    // A desync subsurface's state applies with its own commit.
    child_client.commit();
    f.client.commit_window(0);
    f.round_trip();
    assert!(
        f.server.output_hdr_state(),
        "an sRGB root UI must not hide its PQ video"
    );
    assert_eq!(
        f.server.active_surface_mastering_caps(),
        Some((
            Some("34000:16000:13250:34500:7500:3000:15635:16450:10000000:1".into()),
            Some("1000:400".into())
        ))
    );

    for _ in 0..3 {
        set_surface_hdr_color(&root, Some(SurfaceHdrColor::default()));
        f.client.commit_window(0);
        f.round_trip();
        assert!(
            f.server.output_hdr_state(),
            "a root commit dropped the PQ child"
        );
    }

    set_surface_hdr_color(&child, Some(SurfaceHdrColor::default()));
    child_client.commit();
    f.client.commit_window(0);
    f.round_trip();
    assert!(
        !f.server.output_hdr_state(),
        "declaring the child SDR must restore SDR"
    );
}

#[test]
fn a_pq_window_under_an_sdr_window_leaves_the_frame_and_output_sdr() {
    let mut f = Fixture::with_hdr(true);
    f.create_window(320, 240);
    f.create_window(320, 240);
    let bottom = f
        .server
        .space
        .elements()
        .next()
        .unwrap()
        .wl_surface()
        .unwrap()
        .into_owned();
    set_surface_hdr_color(
        &bottom,
        Some(SurfaceHdrColor {
            transfer: ContentTransfer::Pq,
            is_bt2020: true,
            ..Default::default()
        }),
    );
    f.client.commit_window(0);
    f.round_trip();
    // The output follows the topmost window, so the frame stays SDR: the PQ window below is
    // tone-mapped on its own, and the SDR window on top is drawn untouched (not dimmed by a
    // whole-frame tone map).
    assert!(!f.server.output_hdr_state());
    f.server.create_frame().expect("frame");
    assert_eq!(f.server.last_frame_color.content, ContentTransfer::Sdr);
}

// #79's own regression, with the same assertions: the declarations are cleared to "none"
// (the client stops declaring) rather than declared sRGB. A declaration is double-buffered
// state, so each change is committed as a client does.
#[test]
fn pq_child_survives_sdr_root_and_returns_to_sdr_when_cleared() {
    let mut f = Fixture::new();
    f.create_window(320, 240);
    let child_client = f.client.create_subsurface(0, 0, 0, 160, 120);
    f.round_trip();
    f.round_trip();

    let window = f.server.space.elements().next().unwrap();
    let root = window.wl_surface().unwrap().into_owned();
    let mut children = Vec::new();
    window.with_surfaces(|surface, _states| {
        if surface != &root {
            children.push(surface.clone());
        }
    });
    assert_eq!(children.len(), 1, "must exercise a real Wayland subsurface");
    let child = &children[0];
    set_surface_hdr_color(
        child,
        Some(SurfaceHdrColor {
            transfer: ContentTransfer::Pq,
            is_bt2020: true,
            mastering: Some(MasteringDisplayInfo {
                primaries: [34000, 16000, 13250, 34500, 7500, 3000, 15635, 16450],
                max_luminance: 1000,
                min_luminance: 1,
            }),
            max_cll: Some(1000),
            max_fall: Some(400),
        }),
    );
    set_surface_hdr_color(&root, None);
    child_client.commit();
    f.client.commit_window(0);
    f.round_trip();
    assert!(
        f.server.output_hdr_state(),
        "the sRGB UI root must not hide PQ video"
    );

    let (mastering, cll) = f.server.active_surface_mastering_caps().unwrap();
    assert_eq!(
        mastering.as_deref(),
        Some("34000:16000:13250:34500:7500:3000:15635:16450:1000:1")
    );
    assert_eq!(cll.as_deref(), Some("1000:400"));

    for _ in 0..3 {
        set_surface_hdr_color(&root, None);
        f.client.commit_window(0);
        f.round_trip();
        assert!(f.server.output_hdr_state());
    }
    set_surface_hdr_color(child, None);
    child_client.commit();
    f.round_trip();
    assert!(
        !f.server.output_hdr_state(),
        "clearing the child must restore SDR"
    );
}
