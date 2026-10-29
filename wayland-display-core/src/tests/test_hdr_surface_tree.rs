use super::fixture::Fixture;
use crate::wayland::handlers::color_management::{
    MasteringDisplayInfo, SurfaceHdrColor, set_surface_hdr_color,
};

#[test]
fn pq_child_survives_sdr_root_and_returns_to_sdr_when_cleared() {
    let mut f = Fixture::new();
    f.create_window(320, 240);
    let (_client_child, _role) = f.client.create_subsurface();
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
            is_pq: true,
            is_bt2020: true,
            mastering: Some(MasteringDisplayInfo {
                primaries: [34000, 16000, 13250, 34500, 7500, 3000, 15635, 16450],
                max_luminance: 1000,
                min_luminance: 1,
            }),
            max_cll: Some(1000),
            max_fall: Some(400),
            ..Default::default()
        }),
    );
    set_surface_hdr_color(&root, None);
    assert!(
        f.server.output_hdr_state(),
        "the sRGB UI root must not hide PQ video"
    );

    let (mastering, cll) = f.server.active_surface_mastering_caps().unwrap();
    assert_eq!(
        mastering,
        "34000:16000:13250:34500:7500:3000:15635:16450:1000:1"
    );
    assert_eq!(cll, "1000:400");

    for _ in 0..3 {
        set_surface_hdr_color(&root, None);
        f.round_trip();
        assert!(f.server.output_hdr_state());
    }
    set_surface_hdr_color(child, None);
    assert!(
        !f.server.output_hdr_state(),
        "clearing the child must restore SDR"
    );
}
