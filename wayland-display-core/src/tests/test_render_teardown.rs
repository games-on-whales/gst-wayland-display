//! A stopped compositor frees its GPU objects instead of leaving them to an EGL display that
//! is never terminated.

use super::fixture::Fixture;

/// Renderbuffer names in use on the renderer's context.
fn live_renderbuffers(f: &mut Fixture) -> usize {
    f.server
        .renderer
        .with_context(|gl| {
            (1..1024u32)
                .filter(|&id| unsafe { gl.IsRenderbuffer(id) } != 0)
                .count()
        })
        .expect("renderer context")
}

#[test]
fn shutdown_frees_the_render_target() {
    let mut f = Fixture::new();
    f.server.create_frame().expect("create_frame failed");
    assert!(
        live_renderbuffers(&mut f) > 0,
        "the fixture renders into a renderbuffer"
    );

    f.server.release_render_resources();
    assert_eq!(live_renderbuffers(&mut f), 0);
}
