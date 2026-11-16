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

#[test]
fn app_surface_commit_counter_is_read_only_and_starts_zero() {
    let src = make_src();
    let pspec = src
        .find_property("app-surface-commits")
        .expect("counter property");
    assert!(pspec.flags().contains(gst::glib::ParamFlags::READABLE));
    assert!(!pspec.flags().contains(gst::glib::ParamFlags::WRITABLE));
    assert_eq!(src.property::<u64>("app-surface-commits"), 0);
}

#[test]
fn render_size_and_ui_scale_defaults() {
    let src = make_src();
    assert_eq!(src.property::<i32>("render-width"), 0);
    assert_eq!(src.property::<i32>("render-height"), 0);
    assert_eq!(src.property::<String>("render-size"), "0x0");
    assert_eq!(src.property::<f64>("ui-scale"), 1.0);
}

#[test]
fn render_size_string_sets_both_dimensions() {
    let src = make_src();
    src.set_property("render-size", "1280x720");
    assert_eq!(src.property::<i32>("render-width"), 1280);
    assert_eq!(src.property::<i32>("render-height"), 720);
    assert_eq!(src.property::<String>("render-size"), "1280x720");
}

#[test]
fn render_size_string_resets_to_follow_encode() {
    let src = make_src();
    src.set_property("render-size", "1280x720");
    src.set_property("render-size", "0x0");
    assert_eq!(src.property::<i32>("render-width"), 0);
    assert_eq!(src.property::<i32>("render-height"), 0);
    assert_eq!(src.property::<String>("render-size"), "0x0");
}

#[test]
fn render_size_string_rejects_malformed_values() {
    let src = make_src();
    src.set_property("render-size", "1280x720");
    for bad in [
        "",
        "bogus",
        "1280",
        "1280x",
        "x720",
        "1280x720x60",
        "-1x720",
        "1280 x 720",
        "1280X720",
        "1280x720 ",
        "20000x720", // above the 16384 ParamSpec maximum
        "1920x0",    // half-zero has no meaning; "0x0" is the reset
    ] {
        src.set_property("render-size", bad);
        assert_eq!(
            src.property::<String>("render-size"),
            "1280x720",
            "malformed render-size {bad:?} must be ignored, not applied"
        );
    }
}

/// The live defect this property exists for: with a pair already stored, setting
/// `render-width` alone must NOT hand the compositor a half-updated `1920x720`.
///
/// The element test cannot observe `Command` sends (the compositor thread only exists
/// after `start()`, which needs a GPU + wayland socket), so this pins the *stored* state
/// and the reachable half of the contract; the forwarding rule itself is asserted by the
/// comment-documented guard in `set_property` and by the core's sticky-size unit tests.
#[test]
fn int_props_store_a_resize_without_committing_it() {
    let src = make_src();
    src.set_property("render-size", "1280x720");
    src.set_property("render-width", 1920i32);
    // Stored (so a later `render-size` or caps re-negotiation is consistent) ...
    assert_eq!(src.property::<i32>("render-width"), 1920);
    assert_eq!(src.property::<i32>("render-height"), 720);
    // ... and the atomic property is the way to actually apply the new pair.
    src.set_property("render-size", "1920x1080");
    assert_eq!(src.property::<String>("render-size"), "1920x1080");
}

/// First-time completion through the int props stays supported (the `gst-launch
/// render-width=1280 render-height=720` path).
#[test]
fn int_props_complete_an_initial_pair() {
    let src = make_src();
    src.set_property("render-width", 1280i32);
    src.set_property("render-height", 720i32);
    assert_eq!(src.property::<String>("render-size"), "1280x720");
}

#[test]
fn render_size_round_trips() {
    let src = make_src();
    src.set_property("render-width", 1280i32);
    src.set_property("render-height", 720i32);
    assert_eq!(src.property::<i32>("render-width"), 1280);
    assert_eq!(src.property::<i32>("render-height"), 720);
}

#[test]
fn render_size_resets_to_follow_encode() {
    let src = make_src();
    src.set_property("render-width", 1280i32);
    src.set_property("render-height", 720i32);
    src.set_property("render-width", 0i32);
    src.set_property("render-height", 0i32);
    assert_eq!(src.property::<i32>("render-width"), 0);
    assert_eq!(src.property::<i32>("render-height"), 0);
}

#[test]
fn ui_scale_round_trips() {
    let src = make_src();
    src.set_property("ui-scale", 1.5f64);
    assert_eq!(src.property::<f64>("ui-scale"), 1.5);
}

#[test]
fn mode_ladder_defaults_to_empty() {
    let src = make_src();
    assert_eq!(src.property::<String>("mode-ladder"), "");
}

#[test]
fn mode_ladder_round_trips() {
    let src = make_src();
    src.set_property("mode-ladder", "1920x1080,1600x900,1280x720");
    assert_eq!(
        src.property::<String>("mode-ladder"),
        "1920x1080,1600x900,1280x720"
    );

    // A single rung is a ladder too, and "" clears it.
    src.set_property("mode-ladder", "1280x720");
    assert_eq!(src.property::<String>("mode-ladder"), "1280x720");
    src.set_property("mode-ladder", "");
    assert_eq!(src.property::<String>("mode-ladder"), "");
}

#[test]
fn mode_ladder_rejects_malformed_values() {
    let src = make_src();
    src.set_property("mode-ladder", "1920x1080,1280x720");
    for bad in [
        "bogus",
        "1920",
        "1920x",
        "x1080",
        "1920x1080,",
        ",1920x1080",
        "1920x1080,,1280x720",
        "1920x1080, 1280x720", // whitespace is not accepted
        "1920x1080,-1x720",
        "1920x1080,20000x720", // above the 16384 dimension maximum
        "0x0",                 // a zero-sized mode is meaningless; "" is the reset
        "1920x1080,1280x0",
    ] {
        src.set_property("mode-ladder", bad);
        assert_eq!(
            src.property::<String>("mode-ladder"),
            "1920x1080,1280x720",
            "malformed mode-ladder {bad:?} must be ignored, not applied"
        );
    }
}

#[test]
fn properties_are_readwrite_and_not_construct_only() {
    let src = make_src();
    for name in [
        "render-size",
        "render-width",
        "render-height",
        "ui-scale",
        "mode-ladder",
    ] {
        let pspec = src.find_property(name).expect("property exists");
        let flags = pspec.flags();
        assert!(
            flags.contains(gst::glib::ParamFlags::READABLE)
                && flags.contains(gst::glib::ParamFlags::WRITABLE),
            "{name} must be READWRITE"
        );
        assert!(
            !flags.contains(gst::glib::ParamFlags::CONSTRUCT_ONLY),
            "{name} must be live-writable, not construct-only"
        );
    }
}

#[test]
fn render_size_bounds() {
    let src = make_src();
    let w = src.find_property("render-width").unwrap();
    let spec = w
        .downcast::<gst::glib::ParamSpecInt>()
        .expect("render-width is an int");
    assert_eq!(spec.minimum(), 0);
    assert_eq!(spec.maximum(), 16384);

    let s = src.find_property("ui-scale").unwrap();
    let spec = s
        .downcast::<gst::glib::ParamSpecDouble>()
        .expect("ui-scale is a double");
    assert_eq!(spec.minimum(), 1.0);
    assert_eq!(spec.maximum(), 3.0);
}

#[test]
fn test_picture_is_off_by_default_and_reads_back() {
    let src = make_src();
    assert!(!src.property::<bool>("test-picture"));
    src.set_property("test-picture", true);
    assert!(src.property::<bool>("test-picture"));
}
