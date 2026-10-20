//! Real-GPU encode integration tests, generalised across vendors.
//!
//! Every test is `#[ignore]` and auto-detects its hardware: it skips cleanly
//! (printing why) when the matching GPU or encoder element is absent, so a plain
//! `cargo test` stays green on a machine with no GPU. The harness
//! (`ci/harness.sh`, phase `gpu`) runs them with `--ignored` on the boxes that
//! do have GPUs. A runner that is meant to have the hardware sets
//! `GST_WAYLAND_DISPLAY_REQUIRE_GPU=1`, which turns every skip into a failure so a
//! missing GPU or encoder cannot pass silently.
//!
//! What they cover that the unit tests don't:
//!   * the source negotiates an NV12 modifier the encoder accepts and drives a
//!     real VA / CUDA encoder to EOS (AMD, Intel, Nvidia);
//!   * teardown is clean -- any GLib `CRITICAL` (e.g. the CUDA-context
//!     double-unref class) is made fatal, so it fails the test;
//!   * `supported_nv12_modifiers()` returns a usable set for a present GPU.

use gst::prelude::*;
use std::sync::Once;

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
    let entries = std::fs::read_dir("/dev/dri").ok()?;
    for e in entries.flatten() {
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
        if std::env::var_os("GST_WAYLAND_DISPLAY_REQUIRE_GPU").is_some() {
            panic!("required hardware missing: {why}");
        }
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
    for msg in bus.iter_timed(gst::ClockTime::from_seconds(30)) {
        match msg.view() {
            gst::MessageView::Eos(..) => break,
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
         video/x-raw(memory:VulkanImage),format=NV12,width=320,height=240 ! fakesink"
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
         video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12,width=1280,height=720 ! \
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
         video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=P010,width=1280,height=720 ! \
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
         video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=P010,width=1280,height=720 ! \
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
