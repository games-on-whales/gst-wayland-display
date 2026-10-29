//! Real-GPU encode integration tests, generalised across vendors.
//!
//! Every test is `#[ignore]` and auto-detects its hardware: it skips cleanly
//! (printing why) when the matching GPU or encoder element is absent, so a plain
//! `cargo test` stays green on a machine with no GPU. The harness
//! (`ci/harness.sh`, phase `gpu`) runs them with `--ignored` on the boxes that
//! do have GPUs. `GST_WAYLAND_DISPLAY_REQUIRE_GPU=1` makes the harness require
//! a render node. Tests for other vendors or unsupported codecs still skip.
//!
//! What they cover that the unit tests don't:
//!   * the source negotiates an NV12 modifier the encoder accepts and drives a
//!     real VA / CUDA encoder to EOS (AMD, Intel, Nvidia);
//!   * teardown is clean -- any GLib `CRITICAL` (e.g. the CUDA-context
//!     double-unref class) is made fatal, so it fails the test;
//!   * `supported_nv12_modifiers()` returns a usable set for a present GPU.

use gst::prelude::*;
use std::sync::Once;

#[path = "common/hdr_client.rs"]
mod hdr_client;

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        gst::init().expect("gst init");
        gstwaylanddisplaysrc::plugin_register_static().expect("register plugin");
        // Turn a teardown CRITICAL (bad g_object_unref etc.) into a hard failure.
        gst::glib::log_set_always_fatal(gst::glib::LogLevels::LEVEL_CRITICAL);
        ensure_runtime_dir();
    });
}

/// The compositor needs a private `XDG_RUNTIME_DIR` for its wayland socket.
fn ensure_runtime_dir() {
    use std::os::unix::fs::PermissionsExt;
    let ok = std::env::var_os("XDG_RUNTIME_DIR")
        .map(|d| std::path::Path::new(&d).is_dir())
        .unwrap_or(false);
    if !ok {
        let dir = std::env::temp_dir().join(format!("wlrun-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create XDG_RUNTIME_DIR");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).ok();
        // SAFETY: tests are single-threaded at this point (inside Once).
        unsafe { std::env::set_var("XDG_RUNTIME_DIR", &dir) };
    }
}

/// First render node whose kernel driver matches any of `drivers`.
fn render_node_for(drivers: &[&str]) -> Option<String> {
    // The harness already supplies VULKAN_ENC_NODE/NV12_TEST_NODE. Honor that
    // selection for the full suite, so a multi-GPU host can validate each device.
    let selected =
        std::env::var_os("VULKAN_ENC_NODE").or_else(|| std::env::var_os("NV12_TEST_NODE"));
    let entries = std::fs::read_dir("/dev/dri").ok()?;
    for e in entries.flatten() {
        if selected
            .as_ref()
            .is_some_and(|node| e.path() != std::path::Path::new(node))
        {
            continue;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.starts_with("renderD") {
            continue;
        }
        let uevent = format!("/sys/class/drm/{name}/device/uevent");
        let drv = std::fs::read_to_string(&uevent)
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("DRIVER=").map(str::to_owned))
            })
            .unwrap_or_default();
        if drivers.iter().any(|d| *d == drv) {
            return Some(format!("/dev/dri/{name}"));
        }
    }
    None
}

fn any_render_node() -> Option<String> {
    render_node_for(&["amdgpu", "radeon", "i915", "xe", "nvidia"])
}

fn have(element: &str) -> bool {
    gst::ElementFactory::find(element).is_some()
}

/// Run a gst-launch description to EOS. Returns Err on a bus ERROR, on timeout
/// before EOS, or (via the fatal-criticals handler) aborts on a GLib CRITICAL.
fn run_to_eos(desc: &str) -> Result<(), String> {
    let pipeline = gst::parse::launch(desc)
        .map_err(|e| format!("parse: {e}"))?
        .downcast::<gst::Pipeline>()
        .map_err(|_| "not a pipeline".to_string())?;
    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| format!("set Playing: {e:?}"))?;
    let bus = pipeline.bus().expect("bus");
    let mut saw_eos = false;
    for msg in bus.iter_timed(gst::ClockTime::from_seconds(30)) {
        match msg.view() {
            gst::MessageView::Eos(..) => {
                saw_eos = true;
                break;
            }
            gst::MessageView::Error(err) => {
                let _ = pipeline.set_state(gst::State::Null);
                return Err(format!(
                    "{}: {} ({:?})",
                    err.src().map(|s| s.path_string()).unwrap_or_default(),
                    err.error(),
                    err.debug()
                ));
            }
            _ => {}
        }
    }
    // Null transition exercises teardown -- where the CUDA-context double-unref
    // CRITICAL used to fire (now fatal via log_set_always_fatal).
    let _ = pipeline.set_state(gst::State::Null);
    if saw_eos {
        Ok(())
    } else {
        Err("timed out before EOS (no frames encoded)".into())
    }
}

macro_rules! skip {
    ($($a:tt)*) => {{
        let why = format!($($a)*);
        eprintln!("skip: {why}");
        return;
    }};
}

/// Run `desc` (whose source is named `src`) to EOS and return the caps the source
/// negotiated.
fn negotiated_caps(desc: &str) -> gst::Caps {
    let pipeline = gst::parse::launch(desc)
        .expect("parse")
        .downcast::<gst::Pipeline>()
        .expect("pipeline");
    let src = pipeline.by_name("src").expect("element named src");
    pipeline.set_state(gst::State::Playing).expect("Playing");
    let bus = pipeline.bus().expect("bus");
    let mut saw_eos = false;
    for msg in bus.iter_timed(gst::ClockTime::from_seconds(30)) {
        match msg.view() {
            gst::MessageView::Eos(..) => {
                saw_eos = true;
                break;
            }
            gst::MessageView::Error(err) => {
                let _ = pipeline.set_state(gst::State::Null);
                panic!("{}: {:?}", err.error(), err.debug());
            }
            _ => {}
        }
    }
    let caps = src
        .static_pad("src")
        .and_then(|p| p.current_caps())
        .expect("negotiated caps");
    let _ = pipeline.set_state(gst::State::Null);
    assert!(
        saw_eos,
        "caps were negotiated but the source never reached EOS"
    );
    caps
}

fn caps_field(caps: &gst::Caps, field: &str) -> Option<String> {
    caps.structure(0)?.get::<String>(field).ok()
}

#[test]
#[ignore = "needs an AMD GPU with vah265enc; run via ci/harness.sh gpu"]
fn amd_va_encode_to_eos() {
    init();
    let Some(node) = render_node_for(&["amdgpu", "radeon"]) else {
        skip!("no AMD render node")
    };
    if !have("vah265enc") {
        skip!("no vah265enc");
    }
    run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} num-buffers=20 ! vah265enc ! fakesink"
    ))
    .expect("AMD VA encode");
}

#[test]
#[ignore = "needs an Intel GPU with vah265lpenc; run via ci/harness.sh gpu"]
fn intel_va_encode_to_eos() {
    init();
    let Some(node) = render_node_for(&["i915", "xe"]) else {
        skip!("no Intel render node")
    };
    if !have("vah265lpenc") {
        skip!("no vah265lpenc (Intel is low-power-only)");
    }
    // SAFETY: single-threaded test setup.
    unsafe { std::env::set_var("LIBVA_DRIVER_NAME", "iHD") };
    run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} num-buffers=20 ! vah265lpenc ! fakesink"
    ))
    .expect("Intel VA encode");
}

/// The converter advertises a usable NV12 modifier set for a present GPU --
/// the precondition for any encoder negotiating with the source.
#[test]
#[ignore = "needs a GPU with a Vulkan driver; run via ci/harness.sh gpu"]
fn supported_nv12_modifiers_nonempty() {
    init();
    let Some(node) = any_render_node() else {
        skip!("no render node")
    };
    let minor = waylanddisplaycore::utils::vulkan_nv12::render_node_minor(&node);
    let mods = waylanddisplaycore::utils::vulkan_nv12::supported_nv12_modifiers(minor);
    assert!(
        !mods.is_empty(),
        "expected a non-empty NV12 export modifier set for {node} (minor {minor:?})"
    );
}

/// The Vulkan-encode happy path: the source hands `vulkanh264enc` a shared-device
/// NV12 `memory:VulkanImage` and drives it to EOS on AMD or NVIDIA. Explicit
/// dimensions also exercise NVENC above its minimum supported frame size.
#[test]
#[ignore = "needs AMD/NVIDIA with vulkanh264enc; run via ci/harness.sh gpu"]
fn vulkan_encode_to_eos() {
    init();
    let Some(node) = render_node_for(&["amdgpu", "nvidia"]) else {
        skip!("no AMD/NVIDIA render node")
    };
    if !have("vulkanh264enc") {
        skip!("no vulkanh264enc (needs Vulkan video-encode)");
    }
    run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} vulkan=true num-buffers=60 ! \
         video/x-raw(memory:VulkanImage),width=1280,height=720,framerate=60/1 \
         ! vulkanh264enc ! fakesink"
    ))
    .expect("Vulkan encode");
}

/// Regression for the device-sharing race: forcing `memory:VulkanImage` with **no**
/// encoder downstream used to abort the process with a Rust panic in the compositor thread
/// (`GsVulkanBuf .expect("...no shared GstVulkanDevice?")`). The source now answers the
/// device context itself and mints its own device, so it must run to EOS on that device.
#[test]
#[ignore = "needs a GPU render node; run via ci/harness.sh gpu"]
fn vulkan_without_encoder_uses_its_own_device() {
    init();
    let Some(node) = any_render_node() else {
        skip!("no render node")
    };
    run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} vulkan=true num-buffers=5 ! \
         video/x-raw(memory:VulkanImage),format=NV12,width=320,height=240,framerate=30/1 ! fakesink"
    ))
    .expect("Vulkan output without an encoder");
}

/// The NV12 dmabuf converter's caps carry the BT.601 matrix its shader applies, so an HD
/// stream is not decoded with the BT.709 default.
#[test]
#[ignore = "needs a GPU with a Vulkan driver; run via ci/harness.sh gpu"]
fn nv12_caps_carry_the_bt601_matrix() {
    init();
    let Some(node) = any_render_node() else {
        skip!("no render node")
    };
    let caps = negotiated_caps(&format!(
        "waylanddisplaysrc name=src render-node={node} num-buffers=5 ! \
         video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12,width=1280,height=720,framerate=30/1 ! \
         fakesink"
    ));
    let colorimetry: gst_video::VideoColorimetry = caps_field(&caps, "colorimetry")
        .expect("colorimetry")
        .parse()
        .unwrap();
    assert_eq!(colorimetry.matrix(), gst_video::VideoColorMatrix::Bt601);
}

/// Static `hdr`: the P010 converter output is tagged BT.2100 PQ with mastering and
/// content-light-level metadata derived from `hdr-peak-luminance`.
#[test]
#[ignore = "needs a GPU with a Vulkan driver; run via ci/harness.sh gpu"]
fn p010_hdr_caps_carry_pq_and_metadata() {
    init();
    let Some(node) = any_render_node() else {
        skip!("no render node")
    };
    let caps = negotiated_caps(&format!(
        "waylanddisplaysrc name=src render-node={node} hdr=true hdr-peak-luminance=600 \
         num-buffers=5 ! \
         video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=P010,width=1280,height=720,framerate=30/1 ! \
         fakesink"
    ));
    assert_eq!(
        caps_field(&caps, "colorimetry").as_deref(),
        Some("bt2100-pq")
    );
    assert_eq!(
        caps_field(&caps, "mastering-display-info").as_deref(),
        Some("35400:14600:8500:39850:6550:2300:15635:16450:6000000:1")
    );
    assert_eq!(
        caps_field(&caps, "content-light-level").as_deref(),
        Some("600:240")
    );
}

/// dynamic-hdr with no HDR client: the P010 stream starts, and stays, BT.709 without static
/// metadata, through the dynamic-HDR pipeline set.
#[test]
#[ignore = "needs a GPU with a Vulkan driver; run via ci/harness.sh gpu"]
fn dynamic_hdr_p010_starts_sdr() {
    init();
    let Some(node) = any_render_node() else {
        skip!("no render node")
    };
    let caps = negotiated_caps(&format!(
        "waylanddisplaysrc name=src render-node={node} dynamic-hdr=true num-buffers=30 ! \
         video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=P010,width=1280,height=720,framerate=30/1 ! \
         fakesink"
    ));
    assert_eq!(caps_field(&caps, "colorimetry").as_deref(), Some("bt709"));
    assert_eq!(caps_field(&caps, "mastering-display-info"), None);
}

/// P010 `memory:VulkanImage` into `vulkanh265enc` Main-10 (the patched GStreamer), with the
/// encode-src image created for the H.265 profile.
#[test]
#[ignore = "needs vulkanh265enc (patched GStreamer); run via ci/harness.sh gpu"]
fn vulkan_h265_p010_encode_to_eos() {
    init();
    let Some(node) = render_node_for(&["amdgpu", "nvidia"]) else {
        skip!("no AMD/NVIDIA render node")
    };
    if !have("vulkanh265enc") {
        skip!("no vulkanh265enc (needs the patched GStreamer)");
    }
    run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} vulkan=true hdr=true num-buffers=60 ! \
         video/x-raw(memory:VulkanImage),format=P010_10LE,width=1280,height=720,framerate=60/1 \
         ! vulkanh265enc ! fakesink"
    ))
    .expect("Vulkan H.265 Main-10 encode");
}

/// NV12 `memory:VulkanImage` into `vulkanh265enc` Main (8-bit): the encode-src image must
/// carry the H.265 profile, not the H.264 one.
#[test]
#[ignore = "needs vulkanh265enc (patched GStreamer); run via ci/harness.sh gpu"]
fn vulkan_h265_nv12_encode_to_eos() {
    init();
    let Some(node) = render_node_for(&["amdgpu", "nvidia"]) else {
        skip!("no AMD/NVIDIA render node")
    };
    if !have("vulkanh265enc") {
        skip!("no vulkanh265enc (needs the patched GStreamer)");
    }
    run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} vulkan=true num-buffers=60 ! \
         video/x-raw(memory:VulkanImage),format=NV12,width=1280,height=720,framerate=60/1 \
         ! vulkanh265enc ! fakesink"
    ))
    .expect("Vulkan H.265 Main encode");
}

/// NV12 `memory:VulkanImage` into `vulkanav1enc` Main (the element takes 8-bit input only):
/// the encode-src images must carry the AV1 profile.
#[test]
#[ignore = "needs vulkanav1enc (patched GStreamer); run via ci/harness.sh gpu"]
fn vulkan_av1_encode_to_eos() {
    init();
    let Some(node) = render_node_for(&["amdgpu", "nvidia"]) else {
        skip!("no AMD/NVIDIA render node")
    };
    if !have("vulkanav1enc") {
        skip!("no vulkanav1enc (needs the patched GStreamer)");
    }
    run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} vulkan=true num-buffers=60 ! \
         video/x-raw(memory:VulkanImage),format=NV12,width=1280,height=720,framerate=60/1 \
         ! vulkanav1enc ! fakesink"
    ))
    .expect("Vulkan AV1 encode");
}

/// Dynamic HDR end to end: a client declaring PQ through `wp_color_management_v1` makes the
/// source post `wolf-hdr-state` (hdr=true) and retag its P010 caps as BT.2100 PQ
/// with mastering metadata, after the debounce and without renegotiating format or size.
#[test]
#[ignore = "needs a GPU with a Vulkan driver; run via ci/harness.sh gpu"]
fn dynamic_hdr_flips_caps_for_a_pq_client() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    init();
    let Some(node) = any_render_node() else {
        skip!("no render node")
    };
    let pipeline = gst::parse::launch(&format!(
        "waylanddisplaysrc name=src render-node={node} dynamic-hdr=true ! \
         video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=P010,width=320,height=240,framerate=30/1 ! \
         fakesink"
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();
    let src = pipeline.by_name("src").unwrap();
    let bus = pipeline.bus().unwrap();
    pipeline.set_state(gst::State::Playing).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let mut client = None;
    let mut flipped = false;
    for msg in bus.iter_timed(gst::ClockTime::from_seconds(20)) {
        match msg.view() {
            gst::MessageView::Application(app) => {
                let s = app.structure().unwrap();
                if s.name() == "wayland.src" && client.is_none() {
                    let socket = std::path::Path::new(&std::env::var("XDG_RUNTIME_DIR").unwrap())
                        .join(s.get::<String>("WAYLAND_DISPLAY").unwrap());
                    let stop = stop.clone();
                    client = Some(std::thread::spawn(move || {
                        hdr_client::run_pq_client(socket, stop)
                    }));
                } else if s.name() == "wolf-hdr-state" {
                    assert!(
                        s.get::<bool>("hdr").unwrap(),
                        "first transition must be to HDR"
                    );
                    flipped = true;
                    break;
                }
            }
            gst::MessageView::Error(err) => panic!("{}: {:?}", err.error(), err.debug()),
            _ => {}
        }
    }
    assert!(flipped, "no wolf-hdr-state message within 20s");
    // The retag happens before the first HDR frame is pushed; give it a frame to land.
    let mut caps = None;
    for _ in 0..50 {
        caps = src.static_pad("src").and_then(|p| p.current_caps());
        if caps_field(caps.as_ref().unwrap(), "colorimetry").as_deref() == Some("bt2100-pq") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let caps = caps.unwrap();
    stop.store(true, Ordering::Relaxed);
    let _ = pipeline.set_state(gst::State::Null);
    if let Some(c) = client {
        let _ = c.join();
    }
    assert_eq!(
        caps_field(&caps, "colorimetry").as_deref(),
        Some("bt2100-pq")
    );
    assert!(caps_field(&caps, "mastering-display-info").is_some());
    let s = caps.structure(0).unwrap();
    assert_eq!(s.get::<i32>("width").unwrap(), 320);
    assert!(s.get::<String>("drm-format").unwrap().starts_with("P010"));
}

/// Two sessions in one process (as in Wolf): stopping one must not break the other, for each
/// output type. On NVIDIA every compositor on a device shares one `EGLDisplay`, and the NV12
/// and Vulkan outputs also open a GBM device on it, so a teardown that terminates the display
/// would kill the surviving session's renderer.
#[test]
#[ignore = "needs a GPU render node; run via ci/harness.sh gpu"]
fn second_session_survives_first_teardown() {
    init();
    let Some(node) = any_render_node() else {
        skip!("no render node")
    };
    let mut outputs = vec![
        ("RGBx", "video/x-raw,format=RGBx".to_string()),
        (
            "NV12 dmabuf",
            "video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12".to_string(),
        ),
    ];
    if have("vulkanh264enc") {
        outputs.push((
            "Vulkan",
            "video/x-raw(memory:VulkanImage),format=NV12 ! vulkanh264enc".to_string(),
        ));
    }
    for (label, tail) in outputs {
        let vulkan = if label == "Vulkan" { "vulkan=true" } else { "" };
        second_session_survives(
            &format!(
                "waylanddisplaysrc render-node={node} {vulkan} ! {tail},width=320,height=240,framerate=30/1"
            ),
            label,
        );
    }
}

fn second_session_survives(head: &str, label: &str) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    // The caps filter ends with the size; a trailing encoder takes the rest of `head`.
    let desc = |name: &str| {
        let (caps_and_rest, size) = head.rsplit_once(",width=").unwrap();
        let (caps, rest) = match caps_and_rest.split_once(" ! vulkanh264enc") {
            Some((caps, _)) => (caps.to_string(), " ! vulkanh264enc"),
            None => (caps_and_rest.to_string(), ""),
        };
        format!("{caps},width={size}{rest} ! fakesink name={name} signal-handoffs=true")
    };
    let launch = |name: &str| {
        let p = gst::parse::launch(&desc(name))
            .unwrap()
            .downcast::<gst::Pipeline>()
            .unwrap();
        let count = Arc::new(AtomicU64::new(0));
        let c = count.clone();
        p.by_name(name)
            .unwrap()
            .connect("handoff", false, move |_| {
                c.fetch_add(1, Ordering::Relaxed);
                None
            });
        p.set_state(gst::State::Playing).unwrap();
        (p, count)
    };
    let (a, _) = launch("a");
    let (b, b_frames) = launch("b");
    std::thread::sleep(std::time::Duration::from_secs(2));
    a.set_state(gst::State::Null).unwrap();
    drop(a);
    let before = b_frames.load(Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_secs(2));
    let after = b_frames.load(Ordering::Relaxed);
    let errors: Vec<String> = b
        .bus()
        .unwrap()
        .iter()
        .filter_map(|m| match m.view() {
            gst::MessageView::Error(e) => Some(format!("{}: {:?}", e.error(), e.debug())),
            _ => None,
        })
        .collect();
    let _ = b.set_state(gst::State::Null);
    assert!(before > 0, "{label}: second session never produced a frame");
    assert!(
        errors.is_empty(),
        "{label}: second session errored: {errors:?}"
    );
    assert!(
        after >= before + 30,
        "{label}: second session stalled after the first was torn down ({before} -> {after} frames in 2s)"
    );
}

/// A mid-stream resolution change rebuilds the output (and, for NV12, the converter and its
/// GBM device) while the session's renderer stays in use; frames must keep coming at the new
/// size.
#[test]
#[ignore = "needs a GPU render node; run via ci/harness.sh gpu"]
fn resolution_change_keeps_rendering() {
    use std::sync::{Arc, Mutex};
    init();
    let Some(node) = any_render_node() else {
        skip!("no render node")
    };
    let mut outputs = vec![
        ("video/x-raw,format=RGBx", false),
        (
            "video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12",
            false,
        ),
    ];
    if have("vulkanh264enc") {
        outputs.push(("video/x-raw(memory:VulkanImage),format=NV12", true));
    }
    for (caps, vulkan) in outputs {
        let vk = if vulkan { "vulkan=true" } else { "" };
        let p = gst::parse::launch(&format!(
            "waylanddisplaysrc render-node={node} {vk} ! capsfilter name=f caps=\"{caps},width=320,height=240,framerate=30/1\" ! \
             fakesink name=s signal-handoffs=true"
        ))
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
        let widths = Arc::new(Mutex::new(Vec::<i32>::new()));
        let w = widths.clone();
        p.by_name("s")
            .unwrap()
            .connect("handoff", false, move |args| {
                let pad = args[2].get::<gst::Pad>().unwrap();
                if let Some(c) = pad.current_caps() {
                    w.lock()
                        .unwrap()
                        .push(c.structure(0).unwrap().get::<i32>("width").unwrap());
                }
                None
            });
        p.set_state(gst::State::Playing).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let new_caps: gst::Caps = format!("{caps},width=640,height=480,framerate=30/1")
            .parse()
            .unwrap();
        p.by_name("f").unwrap().set_property("caps", &new_caps);
        std::thread::sleep(std::time::Duration::from_secs(2));
        let errors: Vec<String> = p
            .bus()
            .unwrap()
            .iter()
            .filter_map(|m| match m.view() {
                gst::MessageView::Error(e) => Some(format!("{}: {:?}", e.error(), e.debug())),
                _ => None,
            })
            .collect();
        let _ = p.set_state(gst::State::Null);
        let widths = widths.lock().unwrap();
        assert!(errors.is_empty(), "{caps}: errored: {errors:?}");
        let at_new = widths.iter().filter(|w| **w == 640).count();
        assert!(
            widths.contains(&320) && at_new >= 30,
            "{caps}: expected frames at 320 then >=30 at 640, got {} at 640 of {}",
            at_new,
            widths.len()
        );
    }
}

/// The converter's output pixels, not just its caps: a client fills its window with a known
/// sRGB colour, the source converts it to NV12 (BT.601) or P010 (BT.709) `memory:VulkanImage`
/// and a Vulkan encoder encodes it; decoding the stream back must give the Y'CbCr the
/// shaders' matrices produce for that colour. Exercises the GLES -> Vulkan import (queue
/// ownership, layouts), the compute conversion and the CONCURRENT encode-src image end to end.
#[test]
#[ignore = "needs Vulkan encode and decode (patched GStreamer); run via ci/harness.sh gpu"]
fn vulkan_encode_round_trips_the_client_colour() {
    use gst_video::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    init();
    let Some(node) = render_node_for(&["amdgpu", "nvidia"]) else {
        skip!("no AMD/NVIDIA render node")
    };
    for e in [
        "vulkanh264enc",
        "vulkanh265enc",
        "vulkanh264dec",
        "vulkanh265dec",
        "vulkandownload",
    ] {
        if !have(e) {
            skip!("no {e}");
        }
    }
    // A saturated red: its BT.709 SDR, SDR -> PQ and PQ-passthrough encodings differ by
    // >= 0.037 in some component, well over the tolerance, so the right shader must have run.
    const RGB: [u8; 3] = [240, 16, 16];
    let [r, g, b] = RGB.map(|c| c as f64 / 255.0);
    // BT.2020 NCL limited-range matrix on PQ code values (rgba_to_p010_bt2020.comp and
    // rgba_pqpass_to_p010.comp).
    let bt2020 = |[r, g, b]: [f64; 3]| {
        [
            0.2256 * r + 0.5823 * g + 0.0509 * b + 0.0627,
            -0.1227 * r - 0.3166 * g + 0.4392 * b + 0.5,
            0.4392 * r - 0.4039 * g - 0.0353 * b + 0.5,
        ]
    };
    // SDR -> PQ as rgba_to_p010_bt2020.comp does it, at the default 203-nit SDR white.
    let sdr_to_pq = |c: f64| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let pq_oetf = |l: f64| {
        let (m1, m2, c1, c2, c3) = (0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875);
        let lm = l.powf(m1);
        ((c1 + c2 * lm) / (1.0 + c3 * lm)).powf(m2)
    };
    let lin = [sdr_to_pq(r), sdr_to_pq(g), sdr_to_pq(b)];
    let lin2020 = [
        0.627404 * lin[0] + 0.329283 * lin[1] + 0.043313 * lin[2],
        0.069097 * lin[0] + 0.919541 * lin[1] + 0.011362 * lin[2],
        0.016391 * lin[0] + 0.088013 * lin[1] + 0.895595 * lin[2],
    ];
    let tone_mapped = lin2020.map(|v| pq_oetf(v.clamp(0.0, 1.0) * 203.0 / 10000.0));
    // (label, format, codec, source properties, client declares PQ, expected Y'CbCr).
    let cases = [
        (
            "NV12 SDR (BT.601)",
            "NV12",
            "h264",
            "",
            false,
            [
                0.257 * r + 0.504 * g + 0.098 * b + 0.0625,
                -0.148 * r - 0.291 * g + 0.439 * b + 0.5,
                0.439 * r - 0.368 * g - 0.071 * b + 0.5,
            ],
        ),
        (
            "P010 SDR (BT.709)",
            "P010_10LE",
            "h265",
            "",
            false,
            [
                0.183 * r + 0.614 * g + 0.062 * b + 0.0625,
                -0.101 * r - 0.339 * g + 0.439 * b + 0.5,
                0.439 * r - 0.399 * g - 0.040 * b + 0.5,
            ],
        ),
        (
            "P010 static HDR (SDR -> PQ)",
            "P010_10LE",
            "h265",
            "hdr=true",
            false,
            bt2020(tone_mapped),
        ),
        (
            "P010 dynamic HDR (PQ passthrough)",
            "P010_10LE",
            "h265",
            "dynamic-hdr=true",
            true,
            bt2020([r, g, b]),
        ),
    ];
    for (label, format, codec, props, client_pq, [ey, ecb, ecr]) in cases {
        let p = gst::parse::launch(&format!(
            "waylanddisplaysrc render-node={node} vulkan=true {props} ! \
             video/x-raw(memory:VulkanImage),format={format},width=320,height=240,framerate=30/1 ! \
             vulkan{codec}enc ! {codec}parse ! vulkan{codec}dec ! vulkandownload ! \
             video/x-raw,format={format} ! fakesink name=s signal-handoffs=true"
        ))
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
        let p010 = format == "P010_10LE";
        let last = Arc::new(Mutex::new(None::<(f64, f64, f64)>));
        let l = last.clone();
        p.by_name("s")
            .unwrap()
            .connect("handoff", false, move |args| {
                // Never panic here: this runs inside a C callback.
                let (Ok(buffer), Ok(pad)) =
                    (args[1].get::<gst::Buffer>(), args[2].get::<gst::Pad>())
                else {
                    return None;
                };
                let Some(info) = pad
                    .current_caps()
                    .and_then(|c| gst_video::VideoInfo::from_caps(&c).ok())
                else {
                    return None;
                };
                let Ok(frame) = gst_video::VideoFrame::from_buffer_readable(buffer, &info) else {
                    return None;
                };
                let read = |plane: u32, x: usize, y: usize| -> f64 {
                    let data = frame.plane_data(plane).unwrap();
                    let at = y * frame.plane_stride()[plane as usize] as usize;
                    if p010 {
                        let i = at + x * 2;
                        (u16::from_le_bytes([data[i], data[i + 1]]) >> 6) as f64 / 1023.0
                    } else {
                        data[at + x] as f64 / 255.0
                    }
                };
                // Away from the centre, where the compositor draws its cursor.
                let y = read(0, 80, 60);
                let (cb, cr) = (read(1, 80, 30), read(1, 81, 30));
                *l.lock().unwrap() = Some((y, cb, cr));
                None
            });
        p.set_state(gst::State::Playing).unwrap();
        let bus = p.bus().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let mut client = None;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while std::time::Instant::now() < deadline && client.is_none() {
            if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100)) {
                match msg.view() {
                    gst::MessageView::Application(app)
                        if app.structure().unwrap().name() == "wayland.src" =>
                    {
                        let display = app
                            .structure()
                            .unwrap()
                            .get::<String>("WAYLAND_DISPLAY")
                            .unwrap();
                        let socket =
                            std::path::Path::new(&std::env::var("XDG_RUNTIME_DIR").unwrap())
                                .join(display);
                        let stop = stop.clone();
                        client = Some(std::thread::spawn(move || {
                            hdr_client::run_client(socket, stop, RGB, client_pq)
                        }));
                    }
                    gst::MessageView::Error(e) => {
                        panic!("{label}: {}: {:?}", e.error(), e.debug())
                    }
                    _ => {}
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(3));
        let got = *last.lock().unwrap();
        stop.store(true, Ordering::Relaxed);
        let _ = p.set_state(gst::State::Null);
        if let Some(c) = client {
            let _ = c.join();
        }
        let (y, cb, cr) = got.unwrap_or_else(|| panic!("{label}: no decoded frame"));
        let tol = 4.0 / 255.0;
        assert!(
            (y - ey).abs() < tol && (cb - ecb).abs() < tol && (cr - ecr).abs() < tol,
            "{label}: decoded Y'CbCr ({y:.4}, {cb:.4}, {cr:.4}), expected ({ey:.4}, {ecb:.4}, {ecr:.4})"
        );
    }
}

/// NVIDIA CUDA output into NVENC, run to EOS and torn down. The CUDA context comes from
/// downstream here (nvh265enc creates it and answers the source's context query), the order
/// in which the source used to drop a reference it never owned: a GLib CRITICAL (fatal in
/// these tests) in g_object_unref when the pipeline disposed its stored GstContext.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an NVIDIA GPU with nvh265enc; run via ci/harness.sh gpu"]
fn nvidia_cuda_encode_to_eos() {
    init();
    let Some(node) = render_node_for(&["nvidia"]) else {
        skip!("no NVIDIA render node")
    };
    if !have("nvh265enc") {
        skip!("no nvh265enc");
    }
    for _ in 0..2 {
        run_to_eos(&format!(
            "waylanddisplaysrc render-node={node} num-buffers=30 ! \
             video/x-raw(memory:CUDAMemory),width=1280,height=720,framerate=60/1 ! \
             nvh265enc ! fakesink"
        ))
        .expect("CUDA -> NVENC encode");
    }
}

/// The CUDA context the source answers a context query with right now, if any.
#[cfg(feature = "cuda")]
fn source_cuda_context(src: &gst::Element) -> Option<gst::Context> {
    let mut query = gst::query::Context::new("gst.cuda.context");
    if !src.static_pad("src").unwrap().query(&mut query) {
        return None;
    }
    query.context_owned()
}

/// Identity of the `GstCudaContext` inside a "gst.cuda.context" context.
#[cfg(feature = "cuda")]
fn cuda_context_ptr(context: &gst::Context) -> usize {
    context
        .structure()
        .get::<gst::Object>("gst.cuda.context")
        .expect("no GstCudaContext in the context")
        .as_ptr() as usize
}

/// Runs `src` (num-buffers set) into `encoder` in a new pipeline to EOS, stops it and takes the
/// source back out. Returns the CUDA context the source held while streaming.
#[cfg(feature = "cuda")]
fn run_cuda_source(src: &gst::Element, encoder: &str) -> gst::Context {
    let caps = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            "video/x-raw(memory:CUDAMemory),width=1280,height=720,framerate=60/1"
                .parse::<gst::Caps>()
                .unwrap(),
        )
        .build()
        .unwrap();
    let encoder = gst::ElementFactory::make(encoder).build().unwrap();
    let sink = gst::ElementFactory::make("fakesink").build().unwrap();
    let pipeline = gst::Pipeline::new();
    pipeline.add_many([src, &caps, &encoder, &sink]).unwrap();
    gst::Element::link_many([src, &caps, &encoder, &sink]).unwrap();
    pipeline.set_state(gst::State::Playing).unwrap();
    let bus = pipeline.bus().unwrap();
    let msg = bus
        .timed_pop_filtered(
            gst::ClockTime::from_seconds(30),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        )
        .expect("timed out before EOS");
    if let gst::MessageView::Error(err) = msg.view() {
        panic!("{}: {:?}", err.error(), err.debug());
    }
    let context = source_cuda_context(src).expect("no CUDA context while streaming");
    pipeline.set_state(gst::State::Null).unwrap();
    pipeline.remove(src).unwrap();
    context
}

/// A CUDA context the source acquired from its pipeline (here nvh26xenc's, via the context
/// query) belongs to that run: it is released at stop, and the restarted source, now in another
/// pipeline with another encoder, acquires one again. One the application set before start is
/// kept across restarts and used by every run.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an NVIDIA GPU with nvh264enc and nvh265enc; run via ci/harness.sh gpu"]
fn nvidia_cuda_context_ownership_across_restarts() {
    init();
    let Some(node) = render_node_for(&["nvidia"]) else {
        skip!("no NVIDIA render node")
    };
    if !have("nvh264enc") || !have("nvh265enc") {
        skip!("no nvh264enc / nvh265enc");
    }
    let src = gst::ElementFactory::make("waylanddisplaysrc")
        .property("render-node", &node)
        .property("num-buffers", 30i32)
        .build()
        .unwrap();

    let acquired = run_cuda_source(&src, "nvh265enc");
    assert!(
        source_cuda_context(&src).is_none(),
        "kept the pipeline's CUDA context after stop"
    );
    run_cuda_source(&src, "nvh264enc");
    assert!(
        source_cuda_context(&src).is_none(),
        "kept the second pipeline's CUDA context after stop"
    );

    // As an application would: set a context on the stopped source.
    let app = cuda_context_ptr(&acquired);
    src.set_context(&acquired);
    drop(acquired);
    for encoder in ["nvh264enc", "nvh265enc"] {
        let used = run_cuda_source(&src, encoder);
        assert_eq!(
            cuda_context_ptr(&used),
            app,
            "{encoder}: not the application's context"
        );
        let kept = source_cuda_context(&src).expect("dropped the application's context at stop");
        assert_eq!(cuda_context_ptr(&kept), app);
    }
}

/// An application that answers the source's NEED_CONTEXT with one CUDA context it keeps
/// for every pipeline (Wolf does) gets it back with its reference count intact after each
/// pipeline is torn down and the element finalized, i.e. through the element's real
/// `Settings::drop`. The source used to drop one reference per pipeline here, until the
/// application's context was finalized under a live pipeline (`GST_IS_CUDA_CONTEXT`
/// CRITICALs, then a SIGSEGV in `cuStreamDestroy`). A leak shows up as a count that grows.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an NVIDIA GPU; run via ci/harness.sh gpu"]
fn nvidia_application_cuda_context_survives_pipelines() {
    use waylanddisplaycore::utils::allocator::cuda::{GstCudaContext, init_cuda};

    unsafe extern "C" {
        fn gst_cuda_context_new(device_id: std::ffi::c_int) -> *mut GstCudaContext;
        fn gst_context_new_cuda_context(cuda_ctx: *mut GstCudaContext)
        -> *mut gst::ffi::GstContext;
    }
    fn ref_count(ctx: *mut GstCudaContext) -> u32 {
        unsafe { (*(ctx as *mut gst::glib::gobject_ffi::GObject)).ref_count }
    }

    init();
    let Some(node) = render_node_for(&["nvidia"]) else {
        skip!("no NVIDIA render node")
    };
    init_cuda().expect("init CUDA");
    let app_cuda = unsafe { gst_cuda_context_new(0) };
    assert!(!app_cuda.is_null(), "create the application's CUDA context");
    let app_context =
        unsafe { gst::Context::from_glib_full(gst_context_new_cuda_context(app_cuda)) };
    let baseline = ref_count(app_cuda);

    for i in 0..6 {
        let pipeline = gst::parse::launch(&format!(
            "waylanddisplaysrc render-node={node} num-buffers=10 ! \
             video/x-raw(memory:CUDAMemory),width=640,height=480,framerate=30/1 ! fakesink"
        ))
        .expect("parse")
        .downcast::<gst::Pipeline>()
        .expect("pipeline");
        let bus = pipeline.bus().expect("bus");
        let answer = app_context.clone();
        bus.set_sync_handler(move |_, msg| {
            if let gst::MessageView::NeedContext(nc) = msg.view()
                && nc.context_type() == "gst.cuda.context"
                && let Some(el) = msg.src().and_then(|s| s.downcast_ref::<gst::Element>())
            {
                el.set_context(&answer);
            }
            gst::BusSyncReply::Pass
        });

        pipeline
            .set_state(gst::State::Playing)
            .expect("set Playing");
        let mut saw_eos = false;
        for msg in bus.iter_timed(gst::ClockTime::from_seconds(30)) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    saw_eos = true;
                    break;
                }
                gst::MessageView::Error(err) => panic!("pipeline {i}: {}", err.error()),
                _ => {}
            }
        }
        assert!(saw_eos, "pipeline {i}: no EOS");
        pipeline.set_state(gst::State::Null).expect("set Null");
        bus.unset_sync_handler();
        // Finalize the element: its Settings::drop releases the context slot
        drop(pipeline);
        assert_eq!(
            ref_count(app_cuda),
            baseline,
            "pipeline {i}: the application's context reference count changed"
        );
    }

    drop(app_context);
    unsafe { gst::ffi::gst_object_unref(app_cuda as *mut gst::ffi::GstObject) };
}

/// P010 dmabuf into the VA H.265 encoder (`vah265enc`, or Intel's low-power
/// `vah265lpenc`; Main-10) for longer than the encoder's surface pool lasts
/// without surface reuse: P010 exports must be VA-allocator buffers like NV12, or every frame
/// imports a new surface and the encode fails with "Failed to create the reconstruct
/// picture" after a few seconds.
#[test]
#[ignore = "needs an AMD or Intel GPU with VA H.265 encode; run via ci/harness.sh gpu"]
fn va_p010_encode_outlives_the_surface_pool() {
    init();
    let Some(node) = render_node_for(&["amdgpu", "radeon", "i915", "xe"]) else {
        skip!("no AMD/Intel render node")
    };
    if render_node_for(&["i915", "xe"]).is_some() {
        // SAFETY: single-threaded test setup.
        unsafe { std::env::set_var("LIBVA_DRIVER_NAME", "iHD") };
    }
    let Some((encoder, factory)) = ["vah265enc", "vah265lpenc"]
        .into_iter()
        .find_map(|e| gst::ElementFactory::find(e).map(|f| (e, f)))
    else {
        skip!("no vah265enc / vah265lpenc");
    };
    // The P010 modifier the encoder imports (it does not take LINEAR P010).
    let drm_format = factory
        .static_pad_templates()
        .into_iter()
        .filter(|t| t.direction() == gst::PadDirection::Sink)
        .flat_map(|t| {
            let caps = t.caps();
            (0..caps.size())
                .filter_map(|i| {
                    let s = caps.structure(i)?;
                    let v = s.value("drm-format").ok()?;
                    let list: Vec<String> = v
                        .get::<gst::List>()
                        .map(|l| l.iter().filter_map(|x| x.get::<String>().ok()).collect())
                        .or_else(|_| v.get::<String>().map(|x| vec![x]))
                        .ok()?;
                    Some(list)
                })
                .flatten()
                .collect::<Vec<_>>()
        })
        .find(|f| f.starts_with("P010:"));
    let Some(drm_format) = drm_format else {
        skip!("{encoder} takes no P010 dmabuf");
    };
    run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} hdr=true num-buffers=360 ! \
         video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format={drm_format},width=1280,height=720,framerate=60/1 ! \
         {encoder} ! fakesink"
    ))
    .expect("P010 VA encode");
}
