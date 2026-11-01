//! Pixels of frames composited with mixed SDR and PQ content, rendered by the real GLES
//! renderer (software fixture, raw RGBA output) and read back: the working space follows
//! the output state, the surfaces of the other transfer are converted per surface, and the
//! buffer's source-colour metadata says what was composited.

use crate::Command;
use crate::comp::rendering::convert_solid;
use crate::tests::fixture::Fixture;
use crate::utils::color::ContentTransfer;
use crate::wayland::handlers::color_management::{SurfaceHdrColor, set_surface_hdr_color};
use gst::meta::CustomMeta;
use smithay::wayland::seat::WaylandFocus;
use std::sync::mpsc;
use std::time::Duration;
use test_log::test;

const W: u32 = 320;
const H: u32 = 240;
/// The default SDR reference white (203 nits) as a fraction of 10,000.
const REF_WHITE_SCALE: f32 = 203.0 / 10_000.0;

/// The test client's buffer at (x, y) as straight [0, 1] RGB (see `client::draw`).
fn client_pixel(x: u32, y: u32) -> [f32; 3] {
    let r = (((W - x) * 0xFF) / W).min(((H - y) * 0xFF) / H);
    let g = ((x * 0xFF) / W).min(((H - y) * 0xFF) / H);
    let b = (((W - x) * 0xFF) / W).min((y * 0xFF) / H);
    [r, g, b].map(|c| c as f32 / 255.0)
}

fn pq() -> Option<SurfaceHdrColor> {
    Some(SurfaceHdrColor {
        transfer: ContentTransfer::Pq,
        is_bt2020: true,
        ..Default::default()
    })
}

struct Frame {
    rgba: Vec<u8>,
    pq_tagged: bool,
}

impl Frame {
    fn at(&self, x: u32, y: u32) -> [f32; 3] {
        let i = ((y * W + x) * 4) as usize;
        [0, 1, 2].map(|c| self.rgba[i + c] as f32 / 255.0)
    }
}

fn render(f: &mut Fixture) -> Frame {
    let (buffer, _) = f.server.create_frame().expect("frame");
    let pq_tagged = CustomMeta::from_buffer(&buffer, crate::utils::source_color::SOURCE_COLOR_META)
        .expect("raw output carries the source-colour meta")
        .structure()
        .get::<bool>("pq-passthrough")
        .unwrap();
    let rgba = buffer.map_readable().unwrap().as_slice().to_vec();
    assert_eq!(rgba.len(), (W * H * 4) as usize);
    Frame { rgba, pq_tagged }
}

fn assert_close(actual: [f32; 3], expected: [f32; 3], what: &str) {
    for c in 0..3 {
        assert!(
            (actual[c] - expected[c]).abs() <= 3.0 / 255.0,
            "{what}: got {actual:?}, expected {expected:?}"
        );
    }
}

fn convert(rgb: [f32; 3], to_pq: bool) -> [f32; 3] {
    let c = convert_solid([rgb[0], rgb[1], rgb[2], 1.0], to_pq, REF_WHITE_SCALE);
    [c[0], c[1], c[2]]
}

/// Sample points away from the gradient's 1-pixel steps.
const SAMPLES: [(u32, u32); 4] = [(40, 30), (200, 60), (100, 180), (280, 200)];

#[test]
fn sdr_frame_is_drawn_as_is() {
    let mut f = Fixture::with_hdr(true);
    f.create_window(W as u16, H as u16);
    f.round_trip();
    let frame = render(&mut f);
    assert!(!frame.pq_tagged);
    for (x, y) in SAMPLES {
        assert_close(frame.at(x, y), client_pixel(x, y), "SDR surface");
    }
}

#[test]
fn pq_surface_on_an_sdr_output_is_tone_mapped_and_the_frame_stays_sdr() {
    let mut f = Fixture::with_hdr(true);
    f.create_window(W as u16, H as u16);
    let root = f
        .server
        .space
        .elements()
        .last()
        .unwrap()
        .wl_surface()
        .unwrap()
        .into_owned();
    set_surface_hdr_color(&root, pq());
    f.client.commit_window(0);
    f.round_trip();

    let frame = render(&mut f);
    assert!(!frame.pq_tagged, "an SDR output composites an SDR frame");
    for (x, y) in SAMPLES {
        let expected = convert(client_pixel(x, y), false);
        assert_close(frame.at(x, y), expected, "tone-mapped PQ surface");
        assert!(
            (0..3).any(|c| (expected[c] - client_pixel(x, y)[c]).abs() > 0.05),
            "sample {x},{y} does not distinguish tone-mapped from raw"
        );
    }
}

#[test]
fn sdr_subsurface_on_an_hdr_output_is_pq_encoded_and_the_frame_is_pq() {
    let mut f = Fixture::with_hdr(true);
    f.create_window(W as u16, H as u16);
    let (tx, rx) = mpsc::channel::<Command>();
    f.server.hdr_state_tx = Some(tx);
    let root = f
        .server
        .space
        .elements()
        .last()
        .unwrap()
        .wl_surface()
        .unwrap()
        .into_owned();
    set_surface_hdr_color(&root, pq());
    f.client.commit_window(0);
    // An undeclared (sRGB) subsurface over the right half.
    let _child = f.client.create_subsurface(0, 160, 0, 160, 240);
    f.round_trip();
    f.server.update_hdr_state();
    std::thread::sleep(Duration::from_millis(700));
    f.server.update_hdr_state();
    assert!(f.server.output_hdr(), "output must be HDR");
    drop(rx);

    let frame = render(&mut f);
    assert!(
        frame.pq_tagged,
        "an HDR output with PQ content composites a PQ frame"
    );
    // Left half: the PQ root, drawn as-is.
    for (x, y) in [(40, 30), (100, 180)] {
        assert_close(frame.at(x, y), client_pixel(x, y), "PQ root");
    }
    // Right half: the subsurface shows the client buffer scaled into 160x240 from (160, 0);
    // sample its own buffer coordinates.
    for (x, y) in [(200, 60), (280, 200)] {
        let (bx, by) = ((x - 160) * 2, y);
        let expected = convert(client_pixel(bx, by), true);
        assert_close(frame.at(x, y), expected, "PQ-encoded SDR subsurface");
    }
}

#[test]
fn convert_solid_maps_sdr_white_to_the_reference_white_and_back() {
    // sRGB white at 203 nits is PQ ~0.580; tone-mapping that back lands at the roll-off of
    // the reference white, not above it.
    let pq_white = convert([1.0, 1.0, 1.0], true);
    for c in pq_white {
        assert!((c - 0.5806).abs() < 0.002, "PQ of 203 nits, got {c}");
    }
    let back = convert(pq_white, false);
    for c in back {
        assert!(
            c > 0.9 && c <= 1.0,
            "reference white tone-maps near white, got {c}"
        );
    }
    // Black stays black; premultiplied alpha is kept.
    assert_eq!(
        convert([0.0; 3], true),
        [0.0, 0.0, 0.0].map(|_| convert([0.0; 3], true)[0])
    );
    let half = convert_solid([0.5, 0.5, 0.5, 0.5], true, REF_WHITE_SCALE);
    assert!((half[3] - 0.5).abs() < 1e-6);
    assert!((half[0] - pq_white[0] * 0.5).abs() < 1e-3);
}

#[test]
fn single_pixel_sdr_surface_on_an_hdr_output_is_pq_encoded() {
    // Smithay draws single-pixel buffers as solid colours, which bypass texture shaders.
    let mut f = Fixture::with_hdr(true);
    f.create_window(W as u16, H as u16);
    let (tx, _rx) = mpsc::channel::<Command>();
    f.server.hdr_state_tx = Some(tx);
    let root = f
        .server
        .space
        .elements()
        .last()
        .unwrap()
        .wl_surface()
        .unwrap()
        .into_owned();
    set_surface_hdr_color(&root, pq());
    f.client.commit_window(0);
    let child = f.client.create_subsurface(0, 160, 0, 160, 240);
    // Opaque sRGB white.
    f.client.attach_single_pixel(&child, [u32::MAX; 4]);
    f.round_trip();
    f.server.update_hdr_state();
    std::thread::sleep(Duration::from_millis(700));
    f.server.update_hdr_state();
    assert!(f.server.output_hdr());

    let frame = render(&mut f);
    assert!(frame.pq_tagged);
    let pq_white = convert([1.0; 3], true);
    assert_close(
        frame.at(240, 120),
        pq_white,
        "PQ-encoded single-pixel white",
    );
    assert_close(frame.at(40, 30), client_pixel(40, 30), "PQ root");
}

#[test]
fn without_hdr_shaders_a_pq_frame_falls_back_to_whole_frame_pq() {
    let mut f = Fixture::with_hdr(true);
    f.create_window(W as u16, H as u16);
    // As if the GLES driver rejected the HDR texture shaders.
    f.server.hdr_programs = Some(None);
    let root = f
        .server
        .space
        .elements()
        .last()
        .unwrap()
        .wl_surface()
        .unwrap()
        .into_owned();
    set_surface_hdr_color(&root, pq());
    f.client.commit_window(0);
    f.round_trip();

    let frame = render(&mut f);
    // Nothing converted, and the frame is handed over as PQ, which is right for the PQ
    // surface rather than showing it as gamma or blowing it out as SDR.
    assert!(frame.pq_tagged);
    assert_eq!(f.server.last_frame_color.content, ContentTransfer::Pq);
    for (x, y) in SAMPLES {
        assert_close(frame.at(x, y), client_pixel(x, y), "unconverted PQ surface");
    }
}

#[test]
fn a_refusal_overrides_a_stable_transport_hdr_output() {
    let mut f = Fixture::with_hdr(true);
    f.server.stable_output_hdr = Some(true);
    assert!(f.server.output_hdr());
    // An NV12 stream cannot carry HDR, even on a stable PQ transport.
    f.server.refuse_hdr();
    assert!(!f.server.output_hdr());
}

#[test]
fn the_legacy_10_bit_guess_only_stands_for_the_active_root() {
    use crate::wayland::handlers::color_management::stage_legacy_pq_for_tests;
    let mut f = Fixture::with_hdr(true);
    f.create_window(W as u16, H as u16);
    let child = f.client.create_subsurface(0, 160, 0, 160, 240);
    f.round_trip();
    let window = f.server.space.elements().last().unwrap();
    let root = window.wl_surface().unwrap().into_owned();
    let mut server_child = None;
    window.with_surfaces(|surface, _| {
        if *surface != root {
            server_child = Some(surface.clone());
        }
    });
    // An undeclared subsurface whose buffer the legacy rule would call PQ.
    stage_legacy_pq_for_tests(&server_child.unwrap());
    child.commit();
    f.round_trip();
    let frame = render(&mut f);
    assert!(!frame.pq_tagged);
    assert!(!f.server.output_hdr_state());
    for (x, y) in [(200, 60), (280, 200)] {
        let (bx, by) = ((x - 160) * 2, y);
        assert_close(
            frame.at(x, y),
            client_pixel(bx, by),
            "undeclared subsurface",
        );
    }

    // The same guess on the active root does count.
    stage_legacy_pq_for_tests(&root);
    f.client.commit_window(0);
    f.round_trip();
    assert!(f.server.output_hdr_state());
}
