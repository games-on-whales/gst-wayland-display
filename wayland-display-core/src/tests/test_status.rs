//! The status text (`WaylandDisplay::set_status`) is drawn while no application window is
//! mapped, so a client watching a session whose app is still downloading or starting sees
//! what is happening instead of a black screen; the first mapped window clears it.

use crate::tests::fixture::Fixture;
use test_log::test;

/// Fully white pixels in a frame: status text is white, the empty desktop is black.
fn white_pixels(f: &mut Fixture) -> usize {
    let (buffer, _) = f.server.create_frame().expect("frame");
    let map = buffer.map_readable().unwrap();
    map.as_slice()
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[..3] == [255, 255, 255])
        .count()
}

fn set_status(f: &mut Fixture, status: Option<&str>) {
    f.server.set_status(status.map(str::to_owned));
}

#[test]
fn status_is_drawn_until_a_window_is_mapped() {
    let mut f = Fixture::new();
    let baseline = white_pixels(&mut f);

    set_status(&mut f, Some("Downloading Steam..."));
    assert!(
        white_pixels(&mut f) > baseline + 100,
        "the status text was not drawn on the empty desktop"
    );

    f.create_window(320, 240);
    white_pixels(&mut f);
    assert_eq!(
        f.server.status, None,
        "a mapped window did not clear the status"
    );
}

#[test]
fn clearing_the_status_removes_it() {
    let mut f = Fixture::new();
    let baseline = white_pixels(&mut f);
    set_status(&mut f, Some("Starting Steam..."));
    assert!(white_pixels(&mut f) > baseline + 100);

    set_status(&mut f, None);
    assert_eq!(white_pixels(&mut f), baseline);
}
