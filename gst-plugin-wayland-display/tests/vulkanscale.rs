//! `vulkanscale` — negotiation maths with no GPU, and one GPU-gated proof that a live
//! mid-stream size change reaches the Vulkan encoder's bitstream.
//!
//! The GPU test is the answer to the open question from the 2026-08-16 spike: whether a
//! Vulkan encoder survives `set_format` at a new size (DPB re-init) while running. It
//! asserts on the **decoded** frame size: the chain ends `enc ! parse ! dec ! fakesink`
//! and the size comes off the Vulkan decoder's src caps, which it derives from the SPS /
//! sequence header it just parsed. That oracle is deliberate: `h264parse` lets the
//! *upstream* caps override the SPS ("sps should give this but upstream overrides",
//! gsth264parse.c), so on an image whose encoder does not refresh its src caps the parser
//! reports the stale launch size for a bitstream that really did change. The decoder
//! cannot be fooled that way.
//!
//! The source is `waylanddisplaysrc vulkan=true`, not
//! `videotestsrc ! vulkanupload ! vulkancolorconvert`: a generic
//! `GstVulkanImageBufferPool` hands out one single-plane image per plane, and
//! `gst_vulkan_video_image_create_view` returns NULL for any buffer with more than one
//! memory, so that chain aborts inside `gst_vulkan_encoder_encode` (`pic->in_buffer &&
//! pic->img_view`) with or without this element. Only the producer's single multiplanar
//! `VIDEO_ENCODE_SRC` image can feed these encoders at all.
//!
//! **These tests need `vulkan-enc-output-state-on-resize.patch` in the image's GStreamer.**
//! Stock `vulkanh264enc`/`vulkanh265enc` early-return from `new_sequence` when the profile
//! is unchanged, so a pure size change keeps the launch-size Vulkan video session and DPB
//! pool and never refreshes the output caps. A step back UP then encodes into undersized
//! DPB images: NVENC MMU-faults (Xid 31, `FAULT_PTE ACCESS_TYPE_VIRT_WRITE`) and the next
//! `vkGetQueryPoolResults` returns `VK_ERROR_DEVICE_LOST`. It is nothing to do with this
//! element -- the same failure reproduces with the compositor doing the resize and no
//! `vulkanscale` in the graph -- but it is what `multi_session_live_resize` guards.

use gst::prelude::*;
use std::sync::{Arc, Mutex, Once};
use std::time::Instant;

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        gst::init().expect("gst init");
        gstwaylanddisplaysrc::plugin_register_static().expect("register plugin");
        ensure_runtime_dir();
    });
}

fn ensure_runtime_dir() {
    use std::os::unix::fs::PermissionsExt;
    let ok = std::env::var_os("XDG_RUNTIME_DIR")
        .map(|d| std::path::Path::new(&d).is_dir())
        .unwrap_or(false);
    if !ok {
        let dir = std::env::temp_dir().join(format!("wlrun-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create XDG_RUNTIME_DIR");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).ok();
        unsafe { std::env::set_var("XDG_RUNTIME_DIR", &dir) };
    }
}

fn render_node_for(drivers: &[&str]) -> Option<String> {
    for e in std::fs::read_dir("/dev/dri").ok()?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.starts_with("renderD") {
            continue;
        }
        let drv = std::fs::read_to_string(format!("/sys/class/drm/{name}/device/uevent"))
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("DRIVER=").map(str::to_owned))
            })
            .unwrap_or_default();
        if drivers.contains(&drv.as_str()) {
            return Some(format!("/dev/dri/{name}"));
        }
    }
    None
}

fn have(element: &str) -> bool {
    gst::ElementFactory::find(element).is_some()
}

macro_rules! skip {
    ($($a:tt)*) => {{ eprintln!("skip: {}", format!($($a)*)); return; }};
}

// ── GPU-free: the element and its negotiation ────────────────────────────────

#[test]
fn element_is_registered_with_the_vulkan_image_caps() {
    init();
    let el = gst::ElementFactory::make("vulkanscale")
        .build()
        .expect("vulkanscale must be registered by plugin_init");
    let tmpl = el
        .pad_template("sink")
        .expect("sink pad template")
        .caps()
        .to_string();
    assert!(tmpl.contains("memory:VulkanImage"), "{tmpl}");
    assert!(tmpl.contains("NV12"), "{tmpl}");
    assert!(tmpl.contains("P010_10LE"), "{tmpl}");
}

#[test]
fn defaults_are_bilinear_and_no_negotiated_size() {
    init();
    let el = gst::ElementFactory::make("vulkanscale").build().unwrap();
    assert_eq!(el.property::<i32>("current-width"), 0);
    assert_eq!(el.property::<i32>("current-height"), 0);
    // `method` round-trips through its nick, which is how a pipeline description sets it.
    el.set_property_from_str("method", "nearest");
    el.set_property_from_str("method", "bilinear");
}

/// The src pad must offer any size for a fixed sink size — that freedom is what lets a
/// downstream capsfilter pick a rung.
#[test]
fn src_caps_query_frees_the_size() {
    init();
    let el = gst::ElementFactory::make("vulkanscale").build().unwrap();
    let src = el.static_pad("src").unwrap();
    let caps = src.query_caps(None);
    let s = caps.structure(0).expect("a structure");
    assert!(
        s.get::<gst::IntRange<i32>>("width").is_ok(),
        "width should be a range, got {caps}"
    );
    assert!(
        s.get::<gst::IntRange<i32>>("height").is_ok(),
        "height should be a range, got {caps}"
    );
}

/// …and a downstream filter must be able to pin it, which is the rung actually taking
/// effect.
#[test]
fn src_caps_query_honours_a_downstream_filter() {
    init();
    let el = gst::ElementFactory::make("vulkanscale").build().unwrap();
    let src = el.static_pad("src").unwrap();
    let filter = gst::Caps::builder("video/x-raw")
        .features(["memory:VulkanImage"])
        .field("format", "NV12")
        .field("width", 1280i32)
        .field("height", 720i32)
        .build();
    let caps = src.query_caps(Some(&filter));
    let s = caps.structure(0).expect("a structure");
    assert_eq!(s.get::<i32>("width").unwrap(), 1280, "{caps}");
    assert_eq!(s.get::<i32>("height").unwrap(), 720, "{caps}");
}

// ── GPU-gated: a live rung step through a real Vulkan encoder ────────────────

#[derive(Default)]
struct Seen {
    /// Every distinct (width, height) the parser reported, in order.
    sizes: Vec<(i32, i32)>,
    buffers: u64,
    /// Buffers counted since the flip was requested.
    after_flip: Option<u64>,
}

/// The decoded sizes, the frame rate and the frame count of one [`flip`] run.
type FlipResult = (Vec<(i32, i32)>, f64, u64);

/// Drive `waylanddisplaysrc(1080p) ! vulkanscale ! capsfilter(720p) ! enc ! parse` and
/// flip the capsfilter to 1080p mid-stream. Returns the sizes the parser reported and the
/// achieved frame rate.
///
/// `enc_caps` is an optional capsfilter placed on the ENCODER OUTPUT, written with a
/// trailing `! ` when non-empty (e.g. `"! video/x-h265,profile=main "`). It exists
/// because a Vulkan encoder whose src template advertises a profile its own
/// `H265ProfileMap`/`H264ProfileMap` cannot map will happily negotiate that profile when
/// downstream leaves it free, and then fail to open the video session. Pinning the
/// profile keeps the test on the negotiation path a real encode pipeline uses.
fn flip(
    enc: &str,
    enc_caps: &str,
    parse: &str,
    dec: &str,
    fmt: &str,
    node: &str,
) -> Result<FlipResult, String> {
    let desc = format!(
        "waylanddisplaysrc render-node={node} vulkan=true \
         ! video/x-raw(memory:VulkanImage),format={fmt},width=1920,height=1080,framerate=60/1 \
         ! vulkanscale name=scale \
         ! capsfilter name=cf caps=video/x-raw\\(memory:VulkanImage\\),format={fmt},width=1280,height=720 \
         ! {enc} {enc_caps}! {parse} ! {dec} ! fakesink name=sink sync=false"
    );
    let pipeline = gst::parse::launch(&desc)
        .map_err(|e| format!("parse: {e}"))?
        .downcast::<gst::Pipeline>()
        .map_err(|_| "not a pipeline".to_string())?;
    let cf = pipeline.by_name("cf").unwrap();
    let scale = pipeline.by_name("scale").unwrap();
    let sink = pipeline.by_name("sink").unwrap();

    let seen = Arc::new(Mutex::new(Seen::default()));
    {
        let seen = Arc::clone(&seen);
        sink.static_pad("sink").unwrap().add_probe(
            gst::PadProbeType::BUFFER | gst::PadProbeType::EVENT_DOWNSTREAM,
            move |pad, info| {
                let mut s = seen.lock().unwrap();
                match &info.data {
                    Some(gst::PadProbeData::Buffer(_)) => {
                        s.buffers += 1;
                        if let Some(n) = s.after_flip.as_mut() {
                            *n += 1;
                        }
                    }
                    Some(gst::PadProbeData::Event(e)) => {
                        if let gst::EventView::Caps(c) = e.view() {
                            let st = c.caps().structure(0).unwrap();
                            let wh = (
                                st.get::<i32>("width").unwrap_or(0),
                                st.get::<i32>("height").unwrap_or(0),
                            );
                            if s.sizes.last() != Some(&wh) {
                                eprintln!("  decoded frame size -> {}x{}", wh.0, wh.1);
                                s.sizes.push(wh);
                            }
                        }
                        let _ = pad;
                    }
                    _ => {}
                }
                gst::PadProbeReturn::Ok
            },
        );
    }

    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| format!("set Playing: {e:?}"))?;

    let bus = pipeline.bus().unwrap();
    let start = Instant::now();
    let mut flipped = false;
    let mut err: Option<String> = None;
    let deadline = std::time::Duration::from_secs(30);
    let mut first_frame_at: Option<Instant> = None;
    while start.elapsed() < deadline {
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100))
            && let gst::MessageView::Error(e) = msg.view()
        {
            err = Some(format!(
                "{}: {} ({:?})",
                e.src().map(|s| s.path_string()).unwrap_or_default(),
                e.error(),
                e.debug()
            ));
            break;
        }
        let (n, after) = {
            let s = seen.lock().unwrap();
            (s.buffers, s.after_flip.unwrap_or(0))
        };
        if n > 0 && first_frame_at.is_none() {
            first_frame_at = Some(Instant::now());
        }
        if !flipped && n >= 60 {
            eprintln!("  flipping capsfilter -> 1920x1080 after {n} frames");
            cf.set_property(
                "caps",
                gst::Caps::builder("video/x-raw")
                    .features(["memory:VulkanImage"])
                    .field("format", fmt)
                    .field("width", 1920i32)
                    .field("height", 1080i32)
                    .build(),
            );
            seen.lock().unwrap().after_flip = Some(0);
            flipped = true;
        }
        if flipped && after >= 60 {
            break;
        }
    }
    // Steady-state rate over the frames that flowed after the first one.
    let elapsed = first_frame_at
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let (sizes, buffers, cw) = {
        let s = seen.lock().unwrap();
        (
            s.sizes.clone(),
            s.buffers,
            scale.property::<i32>("current-width"),
        )
    };
    eprintln!("  vulkanscale current-width property = {cw}");
    let _ = pipeline.set_state(gst::State::Null);
    if let Some(e) = err {
        return Err(format!("{e} (sizes so far: {sizes:?})"));
    }
    let fps = if elapsed > 0.0 {
        (buffers.saturating_sub(1)) as f64 / elapsed
    } else {
        0.0
    };
    Ok((sizes, fps, buffers))
}

fn run_flip_for(enc: &str, enc_caps: &str, parse: &str, dec: &str, fmt: &str) {
    init();
    let Some(node) = render_node_for(&["nvidia"]) else {
        skip!("no nvidia render node")
    };
    if !have(enc) {
        skip!("no {enc} in this build/driver");
    }
    if !have(parse) {
        skip!("no {parse}");
    }
    if !have(dec) {
        skip!("no {dec} (the bitstream oracle)");
    }
    let (sizes, fps, buffers) = match flip(enc, enc_caps, parse, dec, fmt, &node) {
        Ok(v) => v,
        // The encoder could not even open a video session on this driver — nothing to do
        // with the scaler (the same pipeline without `vulkanscale` fails identically), so
        // report it as unrunnable rather than as a resize regression.
        Err(e) if e.contains("Unable to start vulkan encoder") => {
            skip!("{enc}: no usable Vulkan video session on this driver: {e}")
        }
        Err(e) => panic!("{enc}: {e}"),
    };
    eprintln!("{enc}: sizes={sizes:?} frames={buffers} fps={fps:.1}");
    assert_eq!(
        sizes.first().copied(),
        Some((1280, 720)),
        "{enc}: the first decoded size must be 1280x720, got {sizes:?}"
    );
    assert!(
        sizes.contains(&(1920, 1080)),
        "{enc}: the mid-stream flip must reach the bitstream, got {sizes:?}"
    );
    assert!(
        fps >= 55.0,
        "{enc}: expected >= 55 fps sustained at 1080p60 input, got {fps:.1}"
    );
}

#[test]
#[ignore = "needs an nvidia GPU with vulkanh264enc; run via ci/harness.sh gpu"]
fn h264_live_resize_reaches_the_bitstream() {
    run_flip_for("vulkanh264enc", "", "h264parse", "vulkanh264dec", "NV12");
}

#[test]
#[ignore = "needs an nvidia GPU with vulkanh265enc; run via ci/harness.sh gpu"]
fn h265_live_resize_reaches_the_bitstream() {
    // NOT a Main-10-only encoder, and NOT a driver limitation — that was this test's
    // own misdiagnosis. `vulkanh265enc`'s src template advertises
    // `{ main, main-10, main-444 }`, but `H265ProfileMap[]` in vkh265enc.c maps only
    // main / main-10 / main-still-picture. With downstream free, negotiation lands on
    // `main-444`, `gst_vulkan_h265_profile_type()` returns
    // STD_VIDEO_H265_PROFILE_IDC_INVALID, and the driver correctly rejects it with
    // `Video profile format not supported (-1000023003)` from
    // vkGetPhysicalDeviceVideoCapabilitiesKHR. Pinning `profile=main` makes 8-bit NV12
    // HEVC work.
    run_flip_for(
        "vulkanh265enc",
        "! video/x-h265,profile=main ",
        "h265parse",
        "vulkanh265dec",
        "NV12",
    );
}

#[test]
#[ignore = "needs an nvidia GPU with the vendored vulkanav1enc; run via ci/harness.sh gpu"]
fn av1_live_resize_reaches_the_bitstream() {
    run_flip_for("vulkanav1enc", "", "av1parse", "vulkanav1dec", "NV12");
}

/// Two sequential encode sessions in ONE process, each stepping down and back up. The
/// streaming server may run many sessions per process, so this is the shape production
/// has — and the shape that caught the encoder's stale video session / DPB pool: the
/// SECOND session's step back up used to MMU-fault the GPU (Xid 31) and surface as
/// `VK_ERROR_DEVICE_LOST` from `vkGetQueryPoolResults`, while the first session was fine.
#[test]
#[ignore = "needs an nvidia GPU with vulkanh264enc; run via ci/harness.sh gpu"]
fn multi_session_live_resize() {
    init();
    let Some(node) = render_node_for(&["nvidia"]) else {
        skip!("no nvidia render node")
    };
    for el in ["vulkanh264enc", "h264parse", "vulkanh264dec"] {
        if !have(el) {
            skip!("no {el}");
        }
    }
    for session in 1..=2 {
        let (sizes, _, _) = flip(
            "vulkanh264enc",
            "",
            "h264parse",
            "vulkanh264dec",
            "NV12",
            &node,
        )
        .unwrap_or_else(|e| {
            panic!(
                "session {session}: {e}\n\
                     A device loss here means the image lacks \
                     deploy/patches/vulkan/vulkan-enc-output-state-on-resize.patch: the \
                     encoder kept the launch-size video session and DPB pool across the \
                     size change."
            )
        });
        assert_eq!(
            sizes,
            vec![(1280, 720), (1920, 1080)],
            "session {session}: both sizes must reach the bitstream"
        );
    }
}

/// At the launch size the element must cost nothing: `GstBaseTransform` passthrough means
/// the producer's buffer reaches the encoder unchanged, which is the whole reason the
/// scaler can sit in the graph permanently.
#[test]
#[ignore = "needs an nvidia GPU with vulkanh264enc; run via ci/harness.sh gpu"]
fn equal_size_is_passthrough() {
    init();
    let Some(node) = render_node_for(&["nvidia"]) else {
        skip!("no nvidia render node")
    };
    if !have("vulkanh264enc") {
        skip!("no vulkanh264enc");
    }
    let desc = format!(
        "waylanddisplaysrc render-node={node} vulkan=true num-buffers=30 \
         ! video/x-raw(memory:VulkanImage),format=NV12,width=1280,height=720,framerate=60/1 \
         ! vulkanscale name=scale \
         ! capsfilter caps=video/x-raw\\(memory:VulkanImage\\),format=NV12,width=1280,height=720 \
         ! vulkanh264enc ! h264parse ! fakesink sync=false"
    );
    let pipeline = gst::parse::launch(&desc)
        .expect("parse")
        .downcast::<gst::Pipeline>()
        .unwrap();
    let scale = pipeline.by_name("scale").unwrap();
    pipeline.set_state(gst::State::Playing).expect("playing");
    let bus = pipeline.bus().unwrap();
    let mut eos = false;
    for msg in bus.iter_timed(gst::ClockTime::from_seconds(30)) {
        match msg.view() {
            gst::MessageView::Eos(..) => {
                eos = true;
                break;
            }
            gst::MessageView::Error(e) => {
                let _ = pipeline.set_state(gst::State::Null);
                panic!("{}: {:?}", e.error(), e.debug());
            }
            _ => {}
        }
    }
    let passthrough = scale
        .downcast_ref::<gst_base::BaseTransform>()
        .map(|t| {
            use gst_base::prelude::BaseTransformExt;
            t.is_passthrough()
        })
        .unwrap_or(false);
    let _ = pipeline.set_state(gst::State::Null);
    assert!(eos, "pipeline did not reach EOS");
    assert!(passthrough, "equal in/out size must engage passthrough");
}
