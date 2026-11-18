//! Property-surface tests for `waylanddisplaysrc`.
//!
//! These only construct the element (no `start()`, so no compositor thread, no GPU and
//! no wayland socket) and exercise the GObject property plumbing: defaults, read-back,
//! and the boundary/clamping behaviour of the ParamSpecs. Runtime forwarding to a live
//! compositor is covered by the `wayland-display-core` unit tests.

use gst::prelude::*;
use std::sync::Once;

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        gst::init().expect("gst init");
        gstwaylanddisplaysrc::plugin_register_static().expect("register plugin");
    });
}

fn make_src() -> gst::Element {
    init();
    gst::ElementFactory::make("waylanddisplaysrc")
        .build()
        .expect("create waylanddisplaysrc")
}

#[test]
fn keyboard_layout_defaults_to_empty_and_reads_back() {
    let src = make_src();
    assert_eq!(src.property::<String>("keyboard-layout"), "");
    assert_eq!(src.property::<String>("keyboard-variant"), "");
    src.set_property("keyboard-layout", "de");
    src.set_property("keyboard-variant", "nodeadkeys");
    assert_eq!(src.property::<String>("keyboard-layout"), "de");
    assert_eq!(src.property::<String>("keyboard-variant"), "nodeadkeys");
}
