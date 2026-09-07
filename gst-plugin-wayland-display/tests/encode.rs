//! Real-GPU encode integration tests, generalised across vendors.
//!
//! Every test is `#[ignore]` and auto-detects its hardware: it skips cleanly
//! (printing why) when the matching GPU or encoder element is absent, so a plain
//! `cargo test` stays green on a machine with no GPU. The harness
//! (`ci/harness.sh`, phase `gpu`) runs them with `--ignored` on the boxes that
//! do have GPUs.
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
    ($($a:tt)*) => {{ eprintln!("skip: {}", format!($($a)*)); return; }};
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
/// NV12 `memory:VulkanImage` and drives it to EOS. Needs an nvidia GPU (the only
/// `vulkanh264enc`-capable driver in our fleet).
#[test]
#[ignore = "needs an nvidia GPU with vulkanh264enc; run via ci/harness.sh gpu"]
fn nvidia_vulkan_encode_to_eos() {
    init();
    let Some(node) = render_node_for(&["nvidia"]) else {
        skip!("no nvidia render node")
    };
    if !have("vulkanh264enc") {
        skip!("no vulkanh264enc (needs Vulkan video-encode)");
    }
    run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} vulkan=true num-buffers=20 ! vulkanh264enc ! fakesink"
    ))
    .expect("nvidia Vulkan encode");
}

/// Regression for the device-sharing race: forcing `memory:VulkanImage` with **no**
/// encoder downstream means the source can never absorb a shared `GstVulkanDevice`.
/// This used to abort the process with a Rust panic in the compositor thread
/// (`GsVulkanBuf .expect("...no shared GstVulkanDevice?")`). It must now wait for the
/// device and, when it never arrives, stop with a clean bus error instead of panicking.
#[test]
#[ignore = "needs a GPU render node; run via ci/harness.sh gpu"]
fn vulkan_without_encoder_errors_not_panics() {
    init();
    let Some(node) = any_render_node() else {
        skip!("no render node")
    };
    let res = run_to_eos(&format!(
        "waylanddisplaysrc render-node={node} vulkan=true num-buffers=5 ! \
         video/x-raw(memory:VulkanImage),format=NV12,width=320,height=240 ! fakesink"
    ));
    // A clean bus error (Err) is the pass condition; reaching EOS would be wrong, and a
    // panic/abort would crash the test binary rather than return here at all.
    assert!(
        res.is_err(),
        "expected a clean error with no encoder downstream, got EOS"
    );
}
