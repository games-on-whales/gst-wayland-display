use crate::utils::{CAT, GstLayer};
use gst::message::Application;
use gst::query::Allocation;
use gst::subclass::prelude::*;
use gst::{Context, Event, Fraction, glib};
use gst::{LibraryError, LoggableError};
use gst::{Structure, prelude::*};
use gst_base::prelude::BaseSrcExt;
use gst_base::prelude::BaseSrcExtManual;
use gst_base::subclass::base_src::CreateSuccess;
use gst_base::subclass::prelude::*;
use gst_video::{NavigationEvent, VideoCapsBuilder, VideoFormat, VideoInfo, VideoInfoDmaDrm};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, LazyLock, Mutex};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;
#[cfg(feature = "cuda")]
use waylanddisplaycore::utils::allocator::{
    cuda,
    cuda::{CUDABufferPool, CUDAContext},
    gst_video_format_name_to_drm_fourcc,
};
#[cfg(feature = "cuda")]
use waylanddisplaycore::utils::video_info::CUDAParams;
use waylanddisplaycore::utils::video_info::VulkanCodec;
use waylanddisplaycore::{
    ButtonState, Channel, Command, DrmFormat, DrmModifier, GstVideoInfo, KeyState, PendingFrame,
    Sender, WaylandDisplay, channel, utils::device::PCIVendor,
};

pub struct WaylandDisplaySrc {
    state: Mutex<Option<State>>,
    settings: Mutex<Settings>,
    // Replaced on every start(); the display owns the receiver and drops it on stop(), so commands
    // sent while stopped fail instead of queueing stale input for the next run. Send through
    // send_command(), never a cloned sender.
    command_tx: Mutex<Sender<Command>>,
    /// Live HDR-colorimetry state (dynamic-hdr only). Mirrors the latest
    /// `poll_hdr_state()` value: `true` => the P010 output caps carry BT.2100 PQ
    /// colorimetry + mastering/CLL, `false` => BT.709 SDR (no static metadata).
    /// Read in `caps()`, driven per-frame in `create()`. Default `false`. When
    /// dynamic-hdr is off this is never consulted (the static `hdr` property governs).
    hdr_active: AtomicBool,
    /// Live HDR static metadata (dynamic-hdr only): the active surface's REAL
    /// `(mastering-display-info, content-light-level)` gst caps strings, as reported by the
    /// compositor (frog for gamescope, `wp_color_management_v1` for sway). `(None, None)` =>
    /// `caps()` falls back to the hardcoded `HDR_MASTERING` / `HDR_CLL` defaults. Driven in
    /// `create()`, read in `caps()`. Never consulted when dynamic-hdr is off.
    hdr_meta: Mutex<(Option<String>, Option<String>)>,
    /// This element's OWN Vulkan-encode device share, replacing the process-global
    /// `vulkan_share` slots so N concurrent Vulkan sessions in one process each mint, own and
    /// destroy their own `VkDevice` (an isolated failure domain). Lives for the element's whole
    /// lifetime (created in `Default`) so `set_context`, the context query and `set_caps` can
    /// reach it regardless of start/stop ordering; a clone goes to this element's compositor
    /// thread in `start()`. The device is retired by ownership: when the element is finalized
    /// its `Arc` drops, so N sessions do not leak N devices.
    vulkan_share: Arc<waylanddisplaycore::utils::vulkan_share::VulkanShare>,
    /// VA display owner shared only with this source's compositor and output ring.
    va_share: Arc<waylanddisplaycore::utils::va_share::VaShare>,
    // Set by unlock(): create() stops waiting for its frame so GstBaseSrc can join the streaming
    // thread before stop() instead of waiting out the compositor reply timeout.
    flushing: AtomicBool,
    // Set while change_state() runs PLAYING->PAUSED; see there.
    pausing: AtomicBool,
    // A frame whose wait unlock() cut short; the next create() waits on it instead of requesting
    // another, since dropping it would shut the compositor down.
    pending_frame: Mutex<Option<PendingFrame>>,
}

impl Default for WaylandDisplaySrc {
    fn default() -> Self {
        // No display yet: drop the receiver so commands sent before start() are discarded.
        let (command_tx, _) = channel();
        WaylandDisplaySrc {
            state: Mutex::new(None),
            settings: Mutex::new(Settings::default()),
            command_tx: Mutex::new(command_tx),
            hdr_active: AtomicBool::new(false),
            hdr_meta: Mutex::new((None, None)),
            va_share: Arc::new(Default::default()),
            vulkan_share: waylanddisplaycore::utils::vulkan_share::VulkanShare::new(),
            flushing: AtomicBool::new(false),
            pausing: AtomicBool::new(false),
            pending_frame: Mutex::new(None),
        }
    }
}

impl WaylandDisplaySrc {
    fn send_command(&self, command: Command) {
        let _ = self.command_tx.lock().unwrap().send(command);
    }

    /// Installs a fresh command channel and returns its receiver.
    fn new_command_channel(&self) -> Channel<Command> {
        let (command_tx, command_rx) = channel();
        *self.command_tx.lock().unwrap() = command_tx;
        command_rx
    }

    /// Installs a fresh command channel with the devices a new display starts with already
    /// queued on it, and returns its receiver.
    fn new_startup_channel(&self, settings: &Settings) -> Channel<Command> {
        let command_rx = self.new_command_channel();
        for path in settings.startup_input_devices() {
            self.send_command(Command::InputDevice(path));
        }
        command_rx
    }
}

impl Settings {
    /// Devices a new display is given on start(): configured ones, then announced ones.
    fn startup_input_devices(&self) -> Vec<String> {
        self.input_devices
            .iter()
            .cloned()
            .chain(
                self.virtual_input_devices
                    .iter()
                    .map(|device| device.path.clone()),
            )
            .collect()
    }

    /// Forgets announced devices that are gone or whose path now belongs to another device, so
    /// start() doesn't hand the display an input it was never given.
    fn prune_virtual_input_devices(&mut self) {
        self.virtual_input_devices.retain(|device| {
            let unchanged =
                device.identity.is_none() || input_device_identity(&device.path) == device.identity;
            if !unchanged {
                tracing::info!(
                    path = device.path,
                    "Dropping a virtual input device that is gone."
                );
            }
            unchanged
        });
    }
}

/// A device announced with VirtualDevicesReady.
#[derive(Debug)]
struct VirtualDevice {
    path: String,
    /// What the path pointed at when announced; None if it couldn't be told, which keeps it.
    identity: Option<PathBuf>,
}

/// The kernel input device behind an evdev node: its sysfs `inputN` directory. Unlike
/// /dev/input/eventN numbers, `inputN` numbers are never handed out twice.
fn input_device_identity(path: &str) -> Option<PathBuf> {
    let node = std::fs::canonicalize(path).ok()?;
    let name = node.file_name()?.to_str()?;
    std::fs::canonicalize(sysfs_root().join("class/input").join(name).join("device")).ok()
}

#[cfg(not(test))]
fn sysfs_root() -> PathBuf {
    PathBuf::from("/sys")
}

#[cfg(test)]
fn sysfs_root() -> PathBuf {
    tests::SYSFS_ROOT.with(|root| root.borrow().clone())
}

#[derive(Debug, Default)]
pub struct Settings {
    render_node: Option<String>,
    input_devices: Vec<String>,
    // Devices announced with VirtualDevicesReady; re-added on every start() while unchanged.
    virtual_input_devices: Vec<VirtualDevice>,
    disable_intel_workaround: bool,
    nv12: bool,
    /// Opt into NV12 `memory:VulkanImage` output on a downstream encoder's shared
    /// `GstVulkanDevice` (zero-copy into `vulkanh264enc`). Default off.
    vulkan: bool,
    /// Tag the P010 10-bit output as HDR (BT.2100 PQ): BT.2020 primaries + SMPTE 2084 (PQ)
    /// transfer + BT.2020 matrix colorimetry, plus static mastering-display-info and
    /// content-light-level caps fields, so a downstream `vulkanh265enc` emits the matching
    /// VUI + mastering/CLL SEI. Only affects the P010 path; NV12/SDR is unchanged. Default off.
    hdr: bool,
    /// Dynamic HDR color management. When true the compositor advertises wp_color_management_v1 +
    /// frog_color_management_v1, renders into an fp16 target, tracks the active surface's
    /// HDR state and flips the P010 caps colorimetry mid-stream (bt709 <-> bt2100-pq),
    /// posting `wolf-hdr-state` bus messages on transitions. Set before starting the
    /// pipeline (the Wayland globals are created at start). Must be per element -- not an
    /// env var -- so one process can stream an HDR client and an SDR client side by side.
    hdr_cm: bool,
    /// Manual override of the HDR mastering-display-info caps string (gst format:
    /// "Rx:Ry:Gx:Gy:Bx:By:Wx:Wy:maxLum:minLum"). `None` => the hardcoded BT.2020/1000-nit
    /// default, or the game's real metadata under dynamic HDR.
    mastering_display_info: Option<String>,
    /// Manual override of the HDR content-light-level caps string ("maxCLL:maxFALL" in
    /// cd/m^2). `None` => the hardcoded 1000:400 default, or the game's real metadata
    /// under dynamic HDR.
    content_light_level: Option<String>,
    vulkan_profile: Option<String>,
    /// Codec of the Vulkan encoder consuming `memory:VulkanImage` output (`h264`, `h265`,
    /// `av1`). `None` => the directly linked encoder's, else H.264 for NV12 / H.265 for P010.
    vulkan_codec: Option<String>,
    #[cfg(feature = "cuda")]
    cuda_context: Option<Arc<Mutex<cuda::CUDAContext>>>,
    #[cfg(feature = "cuda")]
    cuda_raw_ptr: AtomicPtr<cuda::GstCudaContext>,
}

#[cfg(feature = "cuda")]
impl Drop for Settings {
    fn drop(&mut self) {
        // The slot holds the reference the gst_cuda helpers transferred into it; the
        // `cuda_context` wrapper holds its own (see CUDAContext::from_slot).
        unsafe { CUDAContext::release_slot(self.cuda_raw_ptr.as_ptr()) };
    }
}

pub struct State {
    display: WaylandDisplay,
}

#[glib::object_subclass]
impl ObjectSubclass for WaylandDisplaySrc {
    const NAME: &'static str = "GstWaylandDisplaySrc";
    type Type = super::WaylandDisplaySrc;
    type ParentType = gst_base::PushSrc;
    type Interfaces = ();
}

trait EventHandler {
    fn handle_event(&self, event: &Event) -> bool;
}

impl EventHandler for WaylandDisplaySrc {
    fn handle_event(&self, event: &Event) -> bool {
        tracing::debug!("Received event: {:?}", event);

        match event.view() {
            gst::EventView::CustomUpstream(e) => {
                let structure = e.structure().expect("Unable to get message structure");
                if structure.has_name("VirtualDevicesReady") {
                    let path = structure
                        .get::<String>("path")
                        .expect("Should contain the path to the device as a String");
                    // Held across the send so it can't interleave with start() swapping the
                    // channel: the path is either in start()'s snapshot or sent to the new display.
                    let mut settings = self.settings.lock().unwrap();
                    // Always forward: a reconnected device can reuse its old /dev/input path.
                    let identity = input_device_identity(&path);
                    match settings
                        .virtual_input_devices
                        .iter_mut()
                        .find(|device| device.path == path)
                    {
                        Some(device) => device.identity = identity,
                        None => settings.virtual_input_devices.push(VirtualDevice {
                            path: path.clone(),
                            identity,
                        }),
                    }
                    self.send_command(Command::InputDevice(path));
                    return true;
                } else if structure.has_name("MouseMoveAbsolute") {
                    let x = structure
                        .get::<f64>("pointer_x")
                        .expect("Should contain pointer_x");
                    let y = structure
                        .get::<f64>("pointer_y")
                        .expect("Should contain pointer_y");

                    self.send_command(Command::PointerMotionAbsolute((x, y).into()));

                    return true;
                } else if structure.has_name("MouseMoveRelative") {
                    let x = structure
                        .get::<f64>("pointer_x")
                        .expect("Should contain pointer_x");
                    let y = structure
                        .get::<f64>("pointer_y")
                        .expect("Should contain pointer_y");

                    self.send_command(Command::PointerMotion((x, y).into()));

                    return true;
                } else if structure.has_name("MouseButton") {
                    let button = structure
                        .get::<u32>("button")
                        .expect("Should contain button");
                    let pressed = structure
                        .get::<bool>("pressed")
                        .expect("Should contain pressed");

                    self.send_command(Command::PointerButton(
                        button,
                        if pressed {
                            ButtonState::Pressed
                        } else {
                            ButtonState::Released
                        },
                    ));

                    return true;
                } else if structure.has_name("MouseAxis") {
                    let x = structure.get::<f64>("x").expect("Should contain x");
                    let y = structure.get::<f64>("y").expect("Should contain y");

                    self.send_command(Command::PointerAxis(x, y));

                    return true;
                } else if structure.has_name("KeyboardKey") {
                    let key = structure.get::<u32>("key").expect("Should contain key");
                    let pressed = structure
                        .get::<bool>("pressed")
                        .expect("Should contain pressed");

                    self.send_command(Command::KeyboardInput(
                        key,
                        if pressed {
                            KeyState::Pressed
                        } else {
                            KeyState::Released
                        },
                    ));

                    return true;
                } else if structure.has_name("TouchDown") {
                    let x = structure.get::<f64>("x").expect("Should contain x");
                    let y = structure.get::<f64>("y").expect("Should contain y");
                    let id = structure.get::<u32>("id").expect("Should contain id");
                    self.send_command(Command::TouchDown(id, (x, y).into()));
                    return true;
                } else if structure.has_name("TouchUp") {
                    let id = structure.get::<u32>("id").expect("Should contain id");
                    self.send_command(Command::TouchUp(id));
                    return true;
                } else if structure.has_name("TouchMotion") {
                    let x = structure.get::<f64>("x").expect("Should contain x");
                    let y = structure.get::<f64>("y").expect("Should contain y");
                    let id = structure.get::<u32>("id").expect("Should contain id");
                    self.send_command(Command::TouchMotion(id, (x, y).into()));
                    return true;
                } else if structure.has_name("TouchFrame") {
                    self.send_command(Command::TouchFrame);
                    return true;
                } else if structure.has_name("TouchCancel") {
                    self.send_command(Command::TouchCancel);
                    return true;
                }
            }
            gst::EventView::Navigation(n) => {
                let navigation_event = gst_video::NavigationEvent::parse(n).unwrap();

                match navigation_event {
                    NavigationEvent::MouseMove { x, y, .. } => {
                        self.send_command(Command::PointerMotionAbsolute((x, y).into()));

                        return true;
                    }
                    NavigationEvent::MouseButtonPress { button, .. } => {
                        if let Some(cmd) = gst_button_to_msg(button, ButtonState::Pressed) {
                            self.send_command(cmd);
                        } else {
                            tracing::warn!("Unknown mouse button pressed: {:?}", button);
                        }

                        return true;
                    }
                    NavigationEvent::MouseButtonRelease { button, .. } => {
                        if let Some(cmd) = gst_button_to_msg(button, ButtonState::Released) {
                            self.send_command(cmd);
                        } else {
                            tracing::warn!("Unknown mouse button released: {:?}", button);
                        }

                        return true;
                    }
                    NavigationEvent::KeyPress { key, .. } => {
                        if let Some(scancode) = gst_key_to_scancode(&key) {
                            self.send_command(Command::KeyboardInput(scancode, KeyState::Pressed));
                        } else {
                            tracing::warn!("Unknown keyboard key pressed: {:?}", key);
                        }

                        return true;
                    }
                    NavigationEvent::KeyRelease { key, .. } => {
                        if let Some(scancode) = gst_key_to_scancode(&key) {
                            self.send_command(Command::KeyboardInput(scancode, KeyState::Released));
                        } else {
                            tracing::warn!("Unknown keyboard key pressed: {:?}", key);
                        }

                        return true;
                    }
                    _ => {
                        tracing::warn!("Unhandled event: {:?}", navigation_event);
                    }
                };
            }
            _ => (),
        }
        false
    }
}

impl ObjectImpl for WaylandDisplaySrc {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecString::builder("render-node")
                    .nick("DRM Render Node")
                    .blurb("DRM Render Node to use (e.g. /dev/dri/renderD128")
                    .construct()
                    .build(),
                #[cfg(feature = "cuda")]
                glib::ParamSpecInt::builder("cuda-device-id")
                    .nick("CUDA Device ID")
                    .blurb("CUDA Device ID to use")
                    .construct()
                    .default_value(-1)
                    .build(),
                glib::ParamSpecString::builder("mouse")
                    .nick("Input Device")
                    .blurb("Input device to use (e.g. /dev/input/event0")
                    .construct()
                    .build(),
                glib::ParamSpecString::builder("keyboard")
                    .nick("Input Device")
                    .blurb("Input device to use (e.g. /dev/input/event0")
                    .construct()
                    .build(),
                glib::ParamSpecBoolean::builder("disable-intel-workaround")
                    .nick("Disable Intel workaround")
                    .blurb(
                        "Disable workaround for Intel GPUs that tries to fix DRM modifier issues",
                    )
                    .default_value(false)
                    .build(),
                glib::ParamSpecBoolean::builder("nv12")
                    .nick("Prefer NV12 output")
                    .blurb(
                        "Advertise the Vulkan-converted NV12 dmabuf formats first, so a \
                         format-agnostic downstream (e.g. an interpipesink) negotiates NV12 \
                         instead of RGBA. RGBA stays offered as a fallback.",
                    )
                    .default_value(false)
                    .build(),
                glib::ParamSpecBoolean::builder("vulkan")
                    .nick("Prefer NV12 Vulkan output")
                    .blurb(
                        "Advertise NV12/P010 memory:VulkanImage output on the GstVulkanDevice \
                         shared with a downstream Vulkan encoder (zero-copy), offered first so a \
                         format-agnostic interpipesink negotiates it. If the device offered \
                         lacks the converter's extensions, or a directly linked encoder runs on \
                         another device, memory:VulkanImage is withdrawn and negotiation falls \
                         back to the dmabuf/RGBA caps (when downstream accepts them). \
                         Default off.",
                    )
                    .default_value(false)
                    .build(),
                glib::ParamSpecUInt::builder("ring-depth")
                    .minimum(1).maximum(4).default_value(4)
                    .blurb("Conversion ring depth; 4 recommended, smaller rings depend on downstream buffer retention")
                    .mutable_ready().build(),
                glib::ParamSpecBoolean::builder("linear-encode-src")
                    .blurb("Prefer linear Vulkan encode input; retry optimal tiling if allocation fails")
                    .mutable_ready().build(),
                glib::ParamSpecFloat::builder("sdr-reference-white")
                    .minimum(1.0).maximum(10000.0).default_value(203.0)
                    .blurb("SDR reference white in nits for SDR to HDR conversion")
                    .mutable_ready().build(),
                glib::ParamSpecUInt::builder("hdr-peak-luminance")
                    .minimum(1).maximum(10000).default_value(1000)
                    .blurb(
                        "Peak luminance in cd/m^2 of the HDR display presented to clients under \
                         dynamic-hdr (what games calibrate to), and of the default \
                         mastering-display-info / content-light-level",
                    )
                    .mutable_ready().build(),
                glib::ParamSpecString::builder("vulkan-codec")
                    .blurb("Codec of the Vulkan encoder consuming memory:VulkanImage output (h264, h265 or av1). Empty = the directly linked encoder's, else h264 for NV12 and h265 for P010; set it when the encoder sits behind interpipe")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("vulkan-profile")
                    .blurb("Encode image profile, matching the downstream encoder: h264 high/main/baseline/constrained-baseline (NV12); h265 main (NV12) or main-10 (P010); av1 main (NV12)")
                    .mutable_ready().build(),
                glib::ParamSpecBoolean::builder("dynamic-hdr")
                    .blurb(
                        "Dynamic HDR color management: advertise wp_color_management_v1 + \
                         frog_color_management_v1, render into an fp16 target, follow the active \
                         surface's HDR state and flip the P010 caps colorimetry mid-stream \
                         (bt709 <-> bt2100-pq), posting wolf-hdr-state bus messages on \
                         transitions. Per element (an HDR and an SDR session can share a \
                         process). Set before starting the pipeline.",
                    )
                    .default_value(false)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("mastering-display-info")
                    .blurb(
                        "Override the HDR mastering-display-info caps string \
                         (Rx:Ry:Gx:Gy:Bx:By:Wx:Wy:maxLum:minLum). Empty = the hardcoded \
                         BT.2020/1000-nit default or, under dynamic-hdr, the game's real \
                         metadata.",
                    )
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("content-light-level")
                    .blurb(
                        "Override the HDR content-light-level caps string (maxCLL:maxFALL in \
                         cd/m^2). Empty = the hardcoded 1000:400 default or, under dynamic-hdr, \
                         the game's real metadata.",
                    )
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("hdr")
                    .nick("Tag P010 output as HDR (BT.2100 PQ)")
                    .blurb(
                        "On the P010 10-bit path, tag the output caps as HDR: BT.2020 primaries, \
                         SMPTE 2084 (PQ) transfer and BT.2020 matrix colorimetry, plus static \
                         mastering-display-info and content-light-level fields, and convert with \
                         the BT.2020 matrix. A downstream vulkanh265enc then emits the matching \
                         VUI + mastering/CLL SEI. No effect on NV12/SDR output. Default off.",
                    )
                    .default_value(false)
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "ring-depth" | "linear-encode-src" | "sdr-reference-white" | "hdr-peak-luminance" => {
                let mut options = self.vulkan_share.conversion_options();
                match pspec.name() {
                    "ring-depth" => options.ring_depth = value.get::<u32>().unwrap() as usize,
                    "linear-encode-src" => options.linear_encode_src = value.get().unwrap(),
                    "hdr-peak-luminance" => options.hdr_peak_nits = value.get().unwrap(),
                    _ => options.sdr_reference_white = value.get().unwrap(),
                }
                self.vulkan_share.set_conversion_options(options);
            }
            "vulkan-profile" => self.settings.lock().unwrap().vulkan_profile = value.get().unwrap(),
            "vulkan-codec" => {
                self.settings.lock().unwrap().vulkan_codec = value
                    .get::<Option<String>>()
                    .unwrap()
                    .filter(|v| !v.trim().is_empty())
            }
            "dynamic-hdr" => {
                let mut settings = self.settings.lock().unwrap();
                settings.hdr_cm = value.get::<bool>().expect("Type checked upstream");
                // The compositor thread and converter snapshot this via the shared
                // conversion options, so keep them in sync with the property.
                let mut options = self.vulkan_share.conversion_options();
                options.hdr_cm = settings.hdr_cm;
                self.vulkan_share.set_conversion_options(options);
            }
            "mastering-display-info" => {
                let mut settings = self.settings.lock().unwrap();
                settings.mastering_display_info = value
                    .get::<Option<String>>()
                    .expect("Type checked upstream")
                    .filter(|v| !v.trim().is_empty());
            }
            "content-light-level" => {
                let mut settings = self.settings.lock().unwrap();
                settings.content_light_level = value
                    .get::<Option<String>>()
                    .expect("Type checked upstream")
                    .filter(|v| !v.trim().is_empty());
            }

            "render-node" => {
                let mut settings = self.settings.lock().unwrap();
                settings.render_node = value
                    .get::<Option<String>>()
                    .expect("Type checked upstream");
            }
            #[cfg(feature = "cuda")]
            "cuda-device-id" => {
                let mut cuda_context = {
                    let device_id = value.get().unwrap();
                    if device_id != -1 {
                        match CUDAContext::new(device_id) {
                            Ok(ctx) => Some(ctx),
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to create CUDA context with device ID 0: {}",
                                    e
                                );
                                None
                            }
                        }
                    } else {
                        None
                    }
                };
                let mut settings = self.settings.lock().unwrap();
                settings.cuda_context = if cuda_context.is_some() {
                    Some(Arc::new(Mutex::new(cuda_context.take().unwrap())))
                } else {
                    None
                };
            }
            "mouse" => {
                let actual_val = value
                    .get::<Option<String>>()
                    .expect("Type checked upstream");
                if actual_val.is_some() {
                    let mut settings = self.settings.lock().unwrap();
                    settings.input_devices.push(actual_val.unwrap());
                }
            }
            "keyboard" => {
                let actual_val = value
                    .get::<Option<String>>()
                    .expect("Type checked upstream");
                if actual_val.is_some() {
                    let mut settings = self.settings.lock().unwrap();
                    settings.input_devices.push(actual_val.unwrap());
                }
            }
            "disable-intel-workaround" => {
                let mut settings = self.settings.lock().unwrap();
                settings.disable_intel_workaround =
                    value.get::<bool>().expect("Type checked upstream");
            }
            "vulkan" => {
                let mut settings = self.settings.lock().unwrap();
                settings.vulkan = value.get::<bool>().expect("Type checked upstream");
            }
            "nv12" => {
                let mut settings = self.settings.lock().unwrap();
                settings.nv12 = value.get::<bool>().expect("Type checked upstream");
            }
            "hdr" => {
                let mut settings = self.settings.lock().unwrap();
                settings.hdr = value.get::<bool>().expect("Type checked upstream");
            }
            _ => unreachable!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "ring-depth" => (self.vulkan_share.conversion_options().ring_depth as u32).to_value(),
            "linear-encode-src" => self
                .vulkan_share
                .conversion_options()
                .linear_encode_src
                .to_value(),
            "sdr-reference-white" => self
                .vulkan_share
                .conversion_options()
                .sdr_reference_white
                .to_value(),
            "hdr-peak-luminance" => self
                .vulkan_share
                .conversion_options()
                .hdr_peak_nits
                .to_value(),
            "vulkan-profile" => self.settings.lock().unwrap().vulkan_profile.to_value(),
            "vulkan-codec" => self.settings.lock().unwrap().vulkan_codec.to_value(),
            "dynamic-hdr" => self.settings.lock().unwrap().hdr_cm.to_value(),
            "mastering-display-info" => self
                .settings
                .lock()
                .unwrap()
                .mastering_display_info
                .to_value(),
            "content-light-level" => self.settings.lock().unwrap().content_light_level.to_value(),

            "render-node" => {
                let settings = self.settings.lock().unwrap();
                settings
                    .render_node
                    .clone()
                    .unwrap_or_else(|| String::from("/dev/dri/renderD128"))
                    .to_value()
            }
            #[cfg(feature = "cuda")]
            "cuda-device-id" => {
                let settings = self.settings.lock().unwrap();
                match settings.cuda_context {
                    Some(ref _cuda_context) => "Set".into(),
                    None => "None".into(),
                }
            }
            "mouse" => {
                let settings = self.settings.lock().unwrap();
                settings.input_devices.join(",").to_value()
            }
            "keyboard" => {
                let settings = self.settings.lock().unwrap();
                settings.input_devices.join(",").to_value()
            }
            "disable-intel-workaround" => {
                let settings = self.settings.lock().unwrap();
                settings.disable_intel_workaround.to_value()
            }
            "nv12" => {
                let settings = self.settings.lock().unwrap();
                settings.nv12.to_value()
            }
            "vulkan" => {
                let settings = self.settings.lock().unwrap();
                settings.vulkan.to_value()
            }
            "hdr" => {
                let settings = self.settings.lock().unwrap();
                settings.hdr.to_value()
            }
            _ => unreachable!(),
        }
    }

    fn constructed(&self) {
        self.parent_constructed();

        let obj = self.obj();
        obj.set_element_flags(gst::ElementFlags::SOURCE);
        obj.set_live(true);
        obj.set_format(gst::Format::Time);
        obj.set_automatic_eos(false);
        obj.set_do_timestamp(true);
    }
}

impl GstObjectImpl for WaylandDisplaySrc {}

impl ElementImpl for WaylandDisplaySrc {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Wayland display source",
                "Source/Video",
                "GStreamer video src running a wayland compositor",
                "Victoria Brekenfeld <wayland@drakulix.de>, ABeltramo <https://github.com/ABeltramo>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    fn send_event(&self, event: Event) -> bool {
        if self.handle_event(&event) {
            return true;
        }
        self.parent_send_event(event)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst_video::VideoCapsBuilder::new()
                .format(VideoFormat::Rgbx)
                .height_range(..i32::MAX)
                .width_range(..i32::MAX)
                .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                .build();

            let mut dmabuf_caps = gst_video::VideoCapsBuilder::new()
                .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
                .format(VideoFormat::DmaDrm)
                // we can let the drm-format field absent to mean the super set of all formats
                // we'll negotiate the actual format with the pads
                .height_range(..i32::MAX)
                .width_range(..i32::MAX)
                .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                .build();

            dmabuf_caps.merge(caps);

            // NV12/P010 memory:VulkanImage (the shared-device encode path; offered when
            // `vulkan` is set and a downstream encoder shares its GstVulkanDevice). NV12 ⇒
            // vulkanh264enc 8-bit; P010 ⇒ vulkanh265enc Main-10.
            let vulkan_caps = gst_video::VideoCapsBuilder::new()
                .features(["memory:VulkanImage"])
                .format_list([VideoFormat::Nv12, VideoFormat::P01010le])
                .height_range(..i32::MAX)
                .width_range(..i32::MAX)
                .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                .build();
            dmabuf_caps.merge(vulkan_caps);

            #[cfg(feature = "cuda")]
            {
                let cuda_caps = gst_video::VideoCapsBuilder::new()
                    .features([cuda::CAPS_FEATURE_MEMORY_CUDA_MEMORY])
                    .format_list([VideoFormat::Bgra, VideoFormat::Rgba])
                    .height_range(..i32::MAX)
                    .width_range(..i32::MAX)
                    .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                    .build();
                dmabuf_caps.merge(cuda_caps);
            }

            let src_pad_template = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &dmabuf_caps,
            )
            .unwrap();

            vec![src_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }

    fn change_state(
        &self,
        transition: gst::StateChange,
    ) -> Result<gst::StateChangeSuccess, gst::StateChangeError> {
        // GstBaseSrc unlock()s a live source on pause, but a create() cancelled by it can return
        // FLUSHING after a quick resume already restarted the streaming task, which then stops
        // for good. A pause doesn't join the task, so let unlock() ignore it.
        let pausing = transition == gst::StateChange::PlayingToPaused;
        self.pausing.store(pausing, Ordering::Release);
        let res = self.parent_change_state(transition);
        if pausing {
            self.pausing.store(false, Ordering::Release);
        }
        match res {
            Ok(gst::StateChangeSuccess::Success) => {
                if transition.next() == gst::State::Paused {
                    // this is a live source
                    Ok(gst::StateChangeSuccess::NoPreroll)
                } else {
                    Ok(gst::StateChangeSuccess::Success)
                }
            }
            x => x,
        }
    }

    fn set_context(&self, context: &Context) {
        // Absorb a downstream encoder's shared GstVulkanDevice so the Vulkan-encode path can
        // mint encode-src images on the same device (best-effort; no-op for other contexts).
        // Returns true whenever a device is shared afterwards, including when the guard kept
        // the one this element already had; vulkan_share logs which of the two happened.
        let was_usable = self.vulkan_share.usable();
        self.vulkan_share.handle_set_context(context);
        if was_usable && !self.vulkan_share.usable() && self.settings.lock().unwrap().vulkan {
            // Renegotiate without memory:VulkanImage.
            self.obj().src_pad().mark_reconfigure();
        }

        // Absorb a downstream VA encoder's GstVaDisplay context so our NV12 buffers can
        // attach VA surfaces on the same display (best-effort).
        let render_path = {
            let settings = self.settings.lock().unwrap();
            settings
                .render_node
                .clone()
                .unwrap_or_else(|| "/dev/dri/renderD128".into())
        };
        if render_node_backs_va_display(&render_path) {
            let elem_ptr =
                self.obj().upcast_ref::<gst::Element>().as_ptr() as *mut std::ffi::c_void;
            let ctx_ptr = context.as_ptr() as *mut std::ffi::c_void;
            self.va_share
                .handle_set_context(elem_ptr, ctx_ptr, &render_path);
        }

        #[cfg(feature = "cuda")]
        {
            let elem = self.obj().upcast_ref::<gst::Element>().to_owned();
            let cuda_raw_ptr = {
                let settings = self.settings.lock().unwrap();
                settings.cuda_raw_ptr.as_ptr()
            };
            match CUDAContext::new_from_set_context(&elem, &context, -1, cuda_raw_ptr) {
                Ok(ctx) => {
                    let mut settings = self.settings.lock().unwrap();
                    if settings.cuda_context.is_none() {
                        settings.cuda_context = Some(Arc::new(Mutex::new(ctx)));
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to create CUDA context: {}", e);
                }
            }
        }
        self.parent_set_context(context)
    }
}

impl WaylandDisplaySrc {
    /// Guard the NV12 export path: refuse a negotiated modifier the Vulkan converter can't
    /// export, so a bad negotiation fails here with a clear error instead of green/garbled
    /// frames downstream. Non-NV12 DMA caps (the compositor's RGBA dmabuf) pass untouched.
    /// The codec and (if fixed in its caps) profile of a directly linked Vulkan encoder.
    /// Encoded profile constraints are downstream of the raw sink pad: follow simple
    /// filters/queues to the encoder. interpipe has no graph edge, so that case uses the
    /// explicit vulkan-codec / vulkan-profile properties.
    fn downstream_vulkan_encoder(&self) -> Option<(VulkanCodec, Option<String>)> {
        let mut pad = self.obj().static_pad("src")?;
        for _ in 0..32 {
            let element = pad.peer()?.parent_element()?;
            let factory = element.factory()?;
            let src = element.static_pad("src")?;
            if let Some(codec) = VulkanCodec::from_encoder_factory(factory.name().as_str()) {
                let profile = src
                    .current_caps()
                    .or_else(|| src.allowed_caps())
                    .and_then(|caps| caps.structure(0)?.get::<String>("profile").ok());
                return Some((codec, profile));
            }
            if !matches!(factory.name().as_str(), "capsfilter" | "queue" | "identity") {
                return None;
            }
            pad = src;
        }
        None
    }

    /// Guard the NV12 export path: refuse a negotiated modifier the Vulkan converter can't
    /// export, so a bad negotiation fails here with a clear error instead of green/garbled
    /// frames downstream. Non-NV12 DMA caps (the compositor's RGBA dmabuf) pass untouched.
    fn check_nv12_export(
        &self,
        caps: &gst::Caps,
        info: &VideoInfoDmaDrm,
    ) -> Result<(), gst::LoggableError> {
        let drm_format = caps
            .structure(0)
            .and_then(|s| s.get::<String>("drm-format").ok());
        let is_nv12 = drm_format.as_deref().is_some_and(|f| f.starts_with("NV12"));
        let is_p010 = drm_format.as_deref().is_some_and(|f| f.starts_with("P010"));
        if !is_nv12 && !is_p010 {
            return Ok(());
        }
        let (render_path, prefer_nv12) = {
            let s = self.settings.lock().unwrap();
            (
                s.render_node
                    .clone()
                    .unwrap_or_else(|| "/dev/dri/renderD128".into()),
                s.nv12,
            )
        };
        let minor = waylanddisplaycore::utils::vulkan_nv12::render_node_minor(&render_path);
        let modifier = info.modifier();
        let (label, exportable) = if is_p010 {
            (
                "P010",
                waylanddisplaycore::utils::vulkan_nv12::supported_p010_modifiers(minor),
            )
        } else {
            (
                "NV12",
                waylanddisplaycore::utils::vulkan_nv12::supported_nv12_modifiers(minor),
            )
        };
        let encoder_pref = waylanddisplaycore::utils::va_query::import_nv12_modifier(&render_path);
        tracing::info!(
            "waylandsrc: {label} export modifier {modifier:#x} on {render_path} \
             (encoder imports {encoder_pref:#x?}, exportable {exportable:#x?})"
        );
        if !exportable.contains(&modifier) {
            return Err(gst::loggable_error!(
                CAT,
                "negotiated {label} modifier {modifier:#x} is not Vulkan-exportable on \
                 {render_path} (exportable: {exportable:#x?})"
            ));
        }
        // The encoder-modifier-match warning below is NV12/VA-specific; skip it for P010.
        if is_p010 {
            return Ok(());
        }
        // Direct `! vah265enc`: exporting anything but the encoder's own modifier makes it
        // re-import (and radeonsi-VA then fails). Behind interpipe (`nv12=true`) a
        // LINEAR/encoder mismatch is expected -- the consumer's vapostproc imports it -- so
        // only flag it on the direct path.
        if !prefer_nv12 {
            if let Some(pref) = encoder_pref {
                if pref != modifier {
                    tracing::warn!(
                        "waylandsrc: exporting NV12 modifier {modifier:#x} but the VA encoder \
                         on {render_path} imports {pref:#x}; a direct encode needs a vapostproc \
                         bridge"
                    );
                }
            }
        }
        Ok(())
    }
}

impl WaylandDisplaySrc {
    /// When `vulkan=true`, answer a downstream `gst.vulkan.{instance,device}` context query
    /// with the `GstVulkanDevice` we own — created with the external-memory extensions the
    /// converter needs. The encoder then adopts our device (one shared device, no zero-copy
    /// gap, no gstreamer fork). Mirrors how the CUDA path shares its context.
    fn handle_vulkan_context_query(&self, query: &mut gst::QueryRef) -> bool {
        let (vulkan_on, render_node) = {
            let s = self.settings.lock().unwrap();
            (s.vulkan, s.render_node.clone())
        };
        if !vulkan_on {
            return false;
        }
        let node = render_node.unwrap_or_else(|| "/dev/dri/renderD128".into());
        let minor = waylanddisplaycore::utils::vulkan_nv12::render_node_minor(&node);
        self.vulkan_share
            .provide_context(self.obj().upcast_ref::<gst::Element>(), query, minor)
    }
}

impl BaseSrcImpl for WaylandDisplaySrc {
    #[cfg(feature = "cuda")]
    fn query(&self, query: &mut gst::QueryRef) -> bool {
        if query.type_() == gst::QueryType::Context {
            if self.handle_vulkan_context_query(query) {
                return true;
            }
            let settings = self.settings.lock().unwrap();
            match settings.cuda_context {
                Some(ref cuda_context) => {
                    tracing::info!("Handling context query with CUDA");
                    let cuda_context = cuda_context.lock().unwrap();
                    cuda::gst_cuda_handle_context_query_wrapped(
                        self.obj().as_ref().as_ref(),
                        query,
                        &cuda_context,
                    )
                }
                None => BaseSrcImplExt::parent_query(self, query),
            }
        } else {
            BaseSrcImplExt::parent_query(self, query)
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn query(&self, query: &mut gst::QueryRef) -> bool {
        if query.type_() == gst::QueryType::Context && self.handle_vulkan_context_query(query) {
            return true;
        }
        BaseSrcImplExt::parent_query(self, query)
    }

    fn caps(&self, filter: Option<&gst::Caps>) -> Option<gst::Caps> {
        let mut caps = VideoCapsBuilder::new()
            .format(VideoFormat::Rgbx)
            .height_range(..i32::MAX)
            .width_range(..i32::MAX)
            .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
            .build();

        #[cfg(feature = "cuda")]
        {
            let cuda_caps = gst_video::VideoCapsBuilder::new()
                .features([cuda::CAPS_FEATURE_MEMORY_CUDA_MEMORY])
                .format_list([VideoFormat::Bgra, VideoFormat::Rgba])
                .height_range(..i32::MAX)
                .width_range(..i32::MAX)
                .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                .build();

            caps.merge(cuda_caps);
        }

        let state = self.state.lock().unwrap();
        let gst_dma_formats: Vec<String> = match state.as_ref() {
            None => Default::default(),
            Some(state) => {
                let dma_formats = match state.display.get_supported_dma_formats() {
                    Ok(dma_formats) => dma_formats,
                    Err(err) => {
                        gst::element_imp_error!(
                            self,
                            gst::ResourceError::Failed,
                            ["Compositor did not report its DMA formats: {}", err]
                        );
                        return None;
                    }
                };

                let settings = self.settings.lock().unwrap();
                let mut disable_workaround = settings.disable_intel_workaround;
                if let Some(render_device) = state.display.get_render_device() {
                    // Only enable workaround for DG2 (Alchemist) Intel GPUs, Battlemage and later
                    // have reportedly no issues with the DRM modifier and don't require workaround.
                    if !disable_workaround && *render_device.pci_vendor() == PCIVendor::Intel {
                        if !render_device.device_name().contains("DG2") {
                            tracing::info!(
                                "Disabling workaround for non-Alchemist (DG2) Intel GPU"
                            );
                            disable_workaround = true;
                        } else if !disable_workaround {
                            tracing::info!("Enabling workaround for Alchemist (DG2) Intel GPU");
                        }
                    }
                }

                dma_formats
                    .iter()
                    .filter_map(|format| drm_to_gst_format(format, disable_workaround))
                    .collect()
            }
        };

        tracing::info!("Supported DMA formats: {:?}", gst_dma_formats);

        if gst_dma_formats.is_empty() {
            let dmabuf_caps = gst_video::VideoCapsBuilder::new()
                .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
                .format(VideoFormat::DmaDrm)
                .height_range(..i32::MAX)
                .width_range(..i32::MAX)
                .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                .build();
            caps.merge(dmabuf_caps);
        } else {
            for format in gst_dma_formats {
                let dmabuf_caps = gst_video::VideoCapsBuilder::new()
                    .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
                    .format(VideoFormat::DmaDrm)
                    .field("drm-format", &format)
                    .height_range(..i32::MAX)
                    .width_range(..i32::MAX)
                    .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                    .build();
                caps.merge(dmabuf_caps);
            }
        }

        // NV12 via the in-process Vulkan converter: the compositor renders RGBA and Vulkan
        // converts to an NV12 dmabuf. The modifier we advertise is the one downstream can
        // import without a re-import, and we pick it deterministically rather than offering
        // a list and hoping negotiation lands right (it can't behind interpipe -- there's no
        // encoder in the pipeline to intersect with).
        const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
        const DRM_FORMAT_MOD_LINEAR: u64 = 0;
        let (render_path, prefer_nv12) = {
            let s = self.settings.lock().unwrap();
            (
                s.render_node
                    .clone()
                    .unwrap_or_else(|| "/dev/dri/renderD128".into()),
                s.nv12,
            )
        };
        // Match the GPU the converter runs on (the compositor's render node), so advertised
        // modifiers are ones we can actually export on a multi-GPU host.
        let nv12_minor = waylanddisplaycore::utils::vulkan_nv12::render_node_minor(&render_path);

        // `nv12=true` is Wolf: `waylanddisplaysrc ! interpipesink`, the encoder sits behind
        // interpipe so no caps negotiation reaches us. Pin LINEAR -- the one NV12 modifier
        // every VA importer (the consumer's vapostproc) takes -- instead of a tiled/DCC
        // modifier radeonsi-VA mis-imports into green frames. Direct (`! vah265enc`): order
        // the encoder's own modifier (queried from the driver) first so negotiation lands on
        // exactly what it wants, then LINEAR, then the rest the GPU can export.
        let nv12_mods: Vec<u64> = if prefer_nv12 {
            vec![DRM_FORMAT_MOD_LINEAR]
        } else {
            let exportable =
                waylanddisplaycore::utils::vulkan_nv12::supported_nv12_modifiers(nv12_minor);
            let encoder_pref =
                waylanddisplaycore::utils::va_query::import_nv12_modifier(&render_path);
            let mut ordered: Vec<u64> = Vec::new();
            if let Some(p) = encoder_pref {
                if p != DRM_FORMAT_MOD_INVALID && exportable.contains(&p) {
                    ordered.push(p);
                }
            }
            if exportable.contains(&DRM_FORMAT_MOD_LINEAR) {
                ordered.push(DRM_FORMAT_MOD_LINEAR);
            }
            let rest: Vec<u64> = exportable
                .iter()
                .copied()
                .filter(|m| *m != DRM_FORMAT_MOD_INVALID && !ordered.contains(m))
                .collect();
            ordered.extend(rest);
            ordered
        };

        let mut nv12_caps = gst::Caps::new_empty();
        for m in nv12_mods {
            let drm = if m == DRM_FORMAT_MOD_LINEAR {
                "NV12".to_string()
            } else {
                format!("NV12:0x{m:016x}")
            };
            let one = gst_video::VideoCapsBuilder::new()
                .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
                .format(VideoFormat::DmaDrm)
                .field("drm-format", drm)
                .field("colorimetry", nv12_colorimetry())
                .height_range(..i32::MAX)
                .width_range(..i32::MAX)
                .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                .build();
            nv12_caps.merge(one);
        }

        // With `nv12` set, offer the NV12 formats first so a format-agnostic downstream
        // (e.g. an unconstrained interpipesink, as in Wolf) fixates on NV12 instead of
        // RGBA; RGBA stays appended as a fallback. Default: RGBA first (current behaviour),
        // with NV12 still offered for encoders that request it directly.
        let mut caps = if prefer_nv12 {
            nv12_caps.merge(caps);
            nv12_caps
        } else {
            caps.merge(nv12_caps);
            caps
        };

        // HDR (BT.2100 PQ) signalling for the P010 path; see `p010_hdr_metadata`.
        let hdr_meta = self.p010_hdr_metadata();

        // P010 (10-bit 4:2:0) via the same Vulkan converter, for a downstream that asks for
        // it (e.g. a Main-10 dmabuf encoder). Offered as a fallback after NV12/RGBA -- NV12
        // stays the default; P010 is selected only when downstream constrains the format.
        let p010_exportable =
            waylanddisplaycore::utils::vulkan_nv12::supported_p010_modifiers(nv12_minor);
        let mut p010_mods: Vec<u64> = Vec::new();
        if p010_exportable.contains(&DRM_FORMAT_MOD_LINEAR) {
            p010_mods.push(DRM_FORMAT_MOD_LINEAR);
        }
        p010_mods.extend(
            p010_exportable
                .iter()
                .copied()
                .filter(|m| *m != DRM_FORMAT_MOD_INVALID && *m != DRM_FORMAT_MOD_LINEAR),
        );
        for m in p010_mods {
            let drm = if m == DRM_FORMAT_MOD_LINEAR {
                "P010".to_string()
            } else {
                format!("P010:0x{m:016x}")
            };
            let mut b = gst_video::VideoCapsBuilder::new()
                .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
                .format(VideoFormat::DmaDrm)
                .field("drm-format", drm)
                .height_range(..i32::MAX)
                .width_range(..i32::MAX)
                .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1));
            let mut one = b.build();
            set_p010_color(one.get_mut().unwrap().structure_mut(0).unwrap(), &hdr_meta);
            caps.merge(one);
        }

        // `vulkan=true`: advertise NV12 memory:VulkanImage FIRST, so a format-agnostic
        // interpipesink (Wolf) fixates on it and we hand the encoder a shared-device
        // encode-src image. Requires the encoder to share its GstVulkanDevice (set_context);
        // the dmabuf/RGBA caps stay as fallback.
        // NV12 is listed first so a format-agnostic interpipesink (Wolf) fixates on it by
        // default; a downstream Main-10 encoder (vulkanh265enc) constrains the format to
        // P010_10LE, so negotiation intersects to the P010 path on demand.
        // Only while the shared device can actually serve the encoder: a refused device or a
        // directly linked encoder on another VkDevice drops memory:VulkanImage, so
        // negotiation falls back to the dmabuf/RGBA caps below.
        let vulkan_on = self.settings.lock().unwrap().vulkan
            && self.vulkan_share.check_peer_device(&self.obj().src_pad());
        if vulkan_on {
            // NV12 (8-bit SDR) first so a format-agnostic interpipesink fixates on it; P010 is
            // a separate structure so the HDR fields apply ONLY to it (an `hdr` NV12 stream
            // would be wrong). A downstream vulkanh265enc constrains to P010_10LE on demand.
            let nv12_vk = gst_video::VideoCapsBuilder::new()
                .features(["memory:VulkanImage"])
                .format(VideoFormat::Nv12)
                .field("colorimetry", nv12_colorimetry())
                .height_range(..i32::MAX)
                .width_range(..i32::MAX)
                .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                .build();
            let mut p010_vk = gst_video::VideoCapsBuilder::new()
                .features(["memory:VulkanImage"])
                .format(VideoFormat::P01010le)
                .height_range(..i32::MAX)
                .width_range(..i32::MAX)
                .framerate_range(Fraction::new(1, 1)..Fraction::new(i32::MAX, 1))
                .build();
            set_p010_color(
                p010_vk.get_mut().unwrap().structure_mut(0).unwrap(),
                &hdr_meta,
            );
            let mut merged = nv12_vk;
            merged.merge(p010_vk);
            merged.merge(caps);
            caps = merged;
        }

        if let Some(filter) = filter {
            caps = caps.intersect(filter);
        }

        Some(caps)
    }

    fn negotiate(&self) -> Result<(), gst::LoggableError> {
        self.parent_negotiate()
    }

    fn event(&self, event: &Event) -> bool {
        if self.handle_event(&event) {
            return true;
        }
        // A pad-level flush doesn't take the state lock, so it can arrive mid-pause, where the
        // unlock() it triggers is ignored; cancel the frame wait here instead. Before chaining up:
        // the parent blocks on the live lock until the pause finishes.
        if let gst::EventView::FlushStart(_) = event.view() {
            self.flushing.store(true, Ordering::Release);
        }
        self.parent_event(event)
    }

    #[cfg(feature = "cuda")]
    fn decide_allocation(&self, query: &mut Allocation) -> Result<(), LoggableError> {
        // No caps, no allocation
        let (outcaps, _need_pool) = query.get();
        if outcaps.is_none() {
            return self.parent_decide_allocation(query);
        }

        tracing::debug!("Handling allocation query {}", outcaps.unwrap());
        // If it's not CUDA we don't need to share a pool
        let is_cuda = outcaps
            .unwrap()
            .features(0)
            .expect("Failed to get features")
            .contains(cuda::CAPS_FEATURE_MEMORY_CUDA_MEMORY);
        let settings = self.settings.lock().unwrap();
        if settings.cuda_context.is_none() || !is_cuda {
            return self.parent_decide_allocation(query);
        }
        let cuda_ctx = settings.cuda_context.as_ref().unwrap().lock().unwrap();

        // Let's get the pool from the query, if it's not there, we'll create one.
        // `allocation_pools()` is an iterator as of gstreamer-rs 0.24; bind the
        // first item to a local so the query borrow is dropped before we mutate
        // the query further down.
        let first_pool = query.allocation_pools().next();
        let (pool, update_pool, size, min, max) = if let Some((pool, size, min, max)) = first_pool {
            tracing::info!("Found existing allocation pools");
            let wrapped_pool = match pool {
                Some(pool) => match CUDABufferPool::from(pool.as_ptr()) {
                    Ok(pool) => Ok(pool),
                    Err(err) => {
                        tracing::info!(
                            "Failed to get CUDA buffer pool from allocation pool: {}",
                            err
                        );
                        // Nothing to release here. `query.allocation_pools()` parses with
                        // `gst_query_parse_nth_allocation_pool`, which is (transfer full), so
                        // `pools` owns this `BufferPool` and drops it at the end of the scope.
                        CUDABufferPool::new(&cuda_ctx)
                    }
                },
                None => {
                    tracing::info!("Failed to get CUDA buffer pool from allocation pool");
                    CUDABufferPool::new(&cuda_ctx)
                }
            };
            (wrapped_pool, true, size, min, max)
        } else {
            tracing::info!("No allocation pools, creating one");
            let video_info = VideoInfo::from_caps(outcaps.unwrap())?;
            let size = video_info.size() as u32;
            (CUDABufferPool::new(&cuda_ctx), false, size, 0, 0)
        };

        match pool {
            Ok(pool) => {
                // The allocation query only lends us its caps (`query.get()` borrows), so
                // adopt it with `from_glib_none`. `from_glib_full` consumed a reference we
                // never owned, leaving the negotiated caps one short. Paired with the
                // `gst_structure_free` in `CUDABufferPool::get_updated_size()`: the pool
                // config holds a ref on this same caps, so releasing the config copy while
                // the over-unref is present drops the caps below its true count.
                let caps = unsafe { gst::Caps::from_glib_none(outcaps.unwrap().as_ptr()) };
                let stream = cuda_ctx.stream().expect("failed to get CUDA stream");
                pool.configure(&caps, &stream, size, min, max)
                    .expect("failed to configure CUDA pool");

                let updated_size = pool.get_updated_size().expect("failed to get updated size");
                tracing::info!("Configured CUDA buffer pool");

                // This will update the query and activate the pool internally
                if update_pool {
                    pool.set_nth_allocation_pool(query, 0, updated_size, min, max);
                } else {
                    pool.add_allocation_pool(query, updated_size, min, max);
                }

                // Send the pool to the compositor
                self.send_command(Command::UpdateCUDABufferPool(Arc::new(Mutex::new(Some(
                    pool,
                )))));
            }
            Err(err) => {
                tracing::warn!("Failed to create CUDA buffer pool: {}", err);
            }
        }

        self.parent_decide_allocation(query)
    }

    fn set_caps(&self, caps: &gst::Caps) -> Result<(), gst::LoggableError> {
        // NV12 memory:VulkanImage: hand the compositor a VULKAN video-info so it builds the
        // shared-device encode-src converter. The H.264 profile isn't in our (raw) caps --
        // it lives in encoded output caps, potentially across interpipe. The
        // vulkan-profile property must match the consumer (defaults: high/main-10).
        let is_vulkan = caps
            .features(0)
            .is_some_and(|f| f.contains("memory:VulkanImage"));
        if is_vulkan {
            // Ensure OUR shared GstVulkanDevice exists before the compositor allocates on it.
            // We create it (with the external-memory extensions the converter needs) and hand
            // it to the downstream encoder via our context-query answer (see query()), so both
            // sides share one device -- no zero-copy gap and no gstreamer fork.
            let node = self
                .settings
                .lock()
                .unwrap()
                .render_node
                .clone()
                .unwrap_or_else(|| "/dev/dri/renderD128".into());
            let minor = waylanddisplaycore::utils::vulkan_nv12::render_node_minor(&node);
            let base_video_info = gst_video::VideoInfo::from_caps(caps)
                .map_err(|e| gst::loggable_error!(CAT, "Invalid Vulkan caps: {e}"))?;
            // The producer's raw caps don't carry the codec, so it comes from the property, the
            // linked encoder, or the format (P010 ⇒ vulkanh265enc Main-10, NV12 ⇒
            // vulkanh264enc); the encode-src image's video profile is built from it.
            let format = base_video_info.format();
            let (configured_codec, configured_profile) = {
                let s = self.settings.lock().unwrap();
                (s.vulkan_codec.clone(), s.vulkan_profile.clone())
            };
            let downstream = self.downstream_vulkan_encoder();
            let codec = match configured_codec {
                Some(name) => VulkanCodec::from_name(&name)
                    .ok_or_else(|| gst::loggable_error!(CAT, "Unknown vulkan-codec {name}"))?,
                None => downstream
                    .as_ref()
                    .map(|(codec, _)| *codec)
                    .unwrap_or_else(|| VulkanCodec::default_for(format)),
            };
            let profile = configured_profile
                .or_else(|| {
                    downstream
                        .filter(|(c, _)| *c == codec)
                        .and_then(|(_, profile)| profile)
                })
                .unwrap_or_else(|| codec.default_profile(format).into());
            if !codec.accepts(format, &profile) {
                return Err(gst::loggable_error!(
                    CAT,
                    "Unsupported Vulkan {codec:?} profile {profile} for {format:?}"
                ));
            }
            self.vulkan_share.ensure_owned_device(minor);
            let video_info =
                GstVideoInfo::VULKAN(waylanddisplaycore::utils::video_info::VulkanParams {
                    video_info: base_video_info,
                    codec,
                    profile,
                });
            self.send_command(Command::VideoInfo(video_info));
            return self.parent_set_caps(caps);
        }

        let video_info = match VideoInfoDmaDrm::from_caps(caps) {
            Ok(dma_video_info) => {
                self.check_nv12_export(caps, &dma_video_info)?;
                GstVideoInfo::DMA(dma_video_info)
            }
            #[cfg(feature = "cuda")]
            Err(_) => {
                let base_video_info = gst_video::VideoInfo::from_caps(caps)
                    .map_err(|e| gst::loggable_error!(CAT, "Invalid video caps: {e}"))?;
                let is_cuda = caps
                    .features(0)
                    .expect("Failed to get features")
                    .contains(cuda::CAPS_FEATURE_MEMORY_CUDA_MEMORY);
                let cuda_context = {
                    let settings = self.settings.lock().unwrap();
                    settings.cuda_context.clone()
                };
                if is_cuda && cuda_context.is_some() {
                    // memory:CUDAMemory will only get us a base format without modifiers,
                    // let's pick the first DRM format that matches the base format
                    let state = self.state.lock().unwrap();
                    let dma_formats = state
                        .as_ref()
                        .ok_or_else(|| {
                            gst::loggable_error!(
                                CAT,
                                "Source must be started before CUDA negotiation"
                            )
                        })?
                        .display
                        .get_supported_dma_formats()
                        .map_err(|err| {
                            gst::loggable_error!(
                                CAT,
                                "Compositor did not report its DMA formats: {}",
                                err
                            )
                        })?;
                    let chosen_format =
                        gst_video_format_name_to_drm_fourcc(base_video_info.format().to_string())
                            .ok_or_else(|| {
                            gst::loggable_error!(CAT, "Unsupported CUDA video format")
                        })?;
                    let format = dma_formats
                        .iter()
                        .filter(|dma_format| dma_format.code == chosen_format)
                        .next()
                        .ok_or_else(|| gst::loggable_error!(CAT, "No matching CUDA DRM format"))?;
                    let modifier: u64 = format.modifier.into();
                    let video_info =
                        VideoInfoDmaDrm::new(base_video_info, format.code as u32, modifier);
                    GstVideoInfo::CUDA(CUDAParams {
                        video_info,
                        cuda_context: cuda_context.unwrap(),
                    })
                } else {
                    GstVideoInfo::RAW(base_video_info)
                }
            }
            #[cfg(not(feature = "cuda"))]
            Err(_) => GstVideoInfo::RAW(
                VideoInfo::from_caps(caps)
                    .map_err(|e| gst::loggable_error!(CAT, "Invalid video caps: {e}"))?,
            ),
        };

        self.send_command(Command::VideoInfo(video_info));

        self.parent_set_caps(caps)
    }

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let mut state = self.state.lock().unwrap();
        if state.is_some() {
            return Ok(());
        }

        // The channel is swapped under the settings lock so a concurrent VirtualDevicesReady lands
        // either in the queued startup devices or after them, never both or neither.
        let (render_node, command_rx, command_tx) = {
            let mut settings = self.settings.lock().unwrap();
            settings.prune_virtual_input_devices();
            let command_rx = self.new_startup_channel(&settings);
            let command_tx = self.command_tx.lock().unwrap().clone();
            (settings.render_node.clone(), command_rx, command_tx)
        };
        // A new compositor starts SDR and only reports HDR changes, so a previous run's dynamic-HDR
        // state must not carry over into this one.
        self.hdr_active.store(false, Ordering::Relaxed);
        *self.hdr_meta.lock().unwrap() = (None, None);
        self.flushing.store(false, Ordering::Release);
        #[cfg(feature = "cuda")]
        let have_cuda_context = self.settings.lock().unwrap().cuda_context.is_some();

        let elem = self.obj().upcast_ref::<gst::Element>().to_owned();
        let subscriber = Registry::default().with(GstLayer);

        let Ok(mut display) = tracing::subscriber::with_default(subscriber, || {
            WaylandDisplay::new_with_channel(
                render_node.clone(),
                command_tx,
                command_rx,
                // Hand this element's compositor thread a clone of OUR share, so producer +
                // compositor + encoder resolve THIS element's device.
                Arc::clone(&self.vulkan_share),
                Arc::clone(&self.va_share),
            )
        }) else {
            return Err(gst::error_msg!(
                LibraryError::Failed,
                (
                    "Failed to open drm node {}, if you want to utilize software rendering set `render-node=software`.",
                    render_node.unwrap_or("".into())
                )
            ));
        };

        // For a downstream VA encoder, adopt its GstVaDisplay so our NV12 buffers can
        // carry a VA surface on the *same* display -- the encoder then reuses one surface
        // instead of importing (and leaking) a new one every frame. Best-effort.
        //
        // Resolve the same node the compositor falls back to when `render-node` is unset
        // (see WaylandDisplay::new_with_channel), so VA sharing engages on the default
        // path too -- otherwise the encoder imports (and leaks) a surface per frame and
        // starves its reconstruct pool after a few frames.
        let va_node = render_node
            .clone()
            .unwrap_or_else(|| "/dev/dri/renderD128".into());
        if render_node_backs_va_display(&va_node) {
            let elem_ptr = elem.as_ptr() as *mut std::ffi::c_void;
            self.va_share.ensure_shared_display(elem_ptr, &va_node);
        }

        #[cfg(feature = "cuda")]
        match display.get_render_device() {
            Some(render_device) => {
                if *render_device.pci_vendor() == PCIVendor::NVIDIA && !have_cuda_context {
                    tracing::info!(
                        "Acquiring a CudaContext from the pipeline, you can manually set the `cuda-device-id` property to override this behavior"
                    );
                    let cuda_raw_ptr = {
                        let settings = self.settings.lock().unwrap();
                        settings.cuda_raw_ptr.as_ptr()
                    };
                    match CUDAContext::new_from_gstreamer(&elem, -1, cuda_raw_ptr) {
                        Ok(cuda_context) => {
                            let mut settings = self.settings.lock().unwrap();
                            if settings.cuda_context.is_none() {
                                tracing::info!("Acquired a CudaContext via new_from_gstreamer");
                                settings.cuda_context = Some(Arc::new(Mutex::new(cuda_context)));
                            } else {
                                tracing::info!("Acquired a CudaContext via set_context");
                            }
                        }
                        Err(err) => {
                            gst::warning!(CAT, "Failed to acquire a CudaContext: {}", err);
                        }
                    }
                }
            }
            None => {}
        }

        let mut structure = Structure::builder("wayland.src");
        let reply_timeout = display.reply_timeout();
        let env_vars = match display.env_vars() {
            Ok(env_vars) => env_vars,
            Err(RecvTimeoutError::Timeout) => {
                return Err(gst::error_msg!(
                    LibraryError::Init,
                    (
                        "The compositor did not finish initializing within {:?}.",
                        reply_timeout
                    )
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(gst::error_msg!(
                    LibraryError::Init,
                    ("The compositor exited during initialization (its panic message is on stderr); check `render-node`, or set `render-node=software` for software rendering.")
                ));
            }
        };
        for (key, var) in env_vars.flat_map(|var| var.split_once("=")) {
            structure = structure.field(key, var);
        }
        let structure = structure.build();
        if let Err(err) = elem.post_message(Application::builder(structure).src(&elem).build()) {
            gst::warning!(CAT, "Failed to post environment to gstreamer bus: {}", err);
        }

        *state = Some(State { display });

        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let mut state = self.state.lock().unwrap();
        if let Some(state) = state.take() {
            let subscriber = Registry::default().with(GstLayer);
            tracing::subscriber::with_default(subscriber, || drop(state.display));
        }
        // Only after the display: dropping a frame the compositor still owes makes it exit, and
        // Drop must still be able to send Quit and join it.
        self.pending_frame.lock().unwrap().take();
        Ok(())
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn unlock(&self) -> Result<(), gst::ErrorMessage> {
        if !self.pausing.load(Ordering::Acquire) {
            self.flushing.store(true, Ordering::Release);
        }
        Ok(())
    }

    fn unlock_stop(&self) -> Result<(), gst::ErrorMessage> {
        self.flushing.store(false, Ordering::Release);
        Ok(())
    }
}

impl PushSrcImpl for WaylandDisplaySrc {
    fn create(
        &self,
        _buffer: Option<&mut gst::BufferRef>,
    ) -> Result<CreateSuccess, gst::FlowError> {
        let hdr_cm = self.settings.lock().unwrap().hdr_cm;
        loop {
            let subscriber = Registry::default().with(GstLayer);
            let buffer = tracing::subscriber::with_default(subscriber, || {
                // Only the request is made under the state lock: waiting for the frame under it
                // would stall caps queries and stop() for up to the compositor reply timeout.
                if self.flushing.load(Ordering::Acquire) {
                    return Err(gst::FlowError::Flushing);
                }
                let stashed = self.pending_frame.lock().unwrap().take();
                let pending = match stashed {
                    Some(pending) => pending,
                    None => match self.state.lock().unwrap().as_ref() {
                        Some(state) => state.display.request_frame()?,
                        None => return Err(gst::FlowError::Eos),
                    },
                };
                match pending.wait_unless(&self.flushing) {
                    Err(gst::FlowError::Flushing) => {
                        *self.pending_frame.lock().unwrap() = Some(pending);
                        Err(gst::FlowError::Flushing)
                    }
                    frame => frame,
                }
            })?;
            // dynamic-hdr: drained AFTER the frame. The compositor signals a change right
            // before it renders the first frame in the new state, so this is exactly the
            // state `buffer` was converted for.
            let hdr_change = hdr_cm
                .then(|| {
                    let state = self.state.lock().unwrap();
                    state
                        .as_ref()
                        .and_then(|state| state.display.poll_hdr_state())
                })
                .flatten();
            match hdr_change {
                Some((hdr, mastering, cll)) => {
                    if self.apply_hdr_state(hdr, mastering, cll)? {
                        return Ok(CreateSuccess::NewBuffer(buffer));
                    }
                    // Downstream refused HDR and the compositor now renders SDR: this frame
                    // was converted for HDR, so replace it.
                }
                None => return Ok(CreateSuccess::NewBuffer(buffer)),
            }
        }
    }
}

impl WaylandDisplaySrc {
    /// Effective HDR static metadata `(mastering-display-info, content-light-level)` of the
    /// P010 output, or `None` when it is SDR. Static `hdr` uses the defaults; under
    /// dynamic-hdr the live state and the active surface's own metadata (frog/gamescope or
    /// wp_color_management/sway) apply, so the game's "HDR Luminance" slider reaches the
    /// encoder's SEI. The defaults describe a BT.2020 display of `hdr-peak-luminance`; the
    /// mastering-display-info / content-light-level properties override either.
    fn p010_hdr_metadata(&self) -> Option<(String, String)> {
        let settings = self.settings.lock().unwrap();
        let hdr = if settings.hdr_cm {
            self.hdr_active.load(Ordering::Relaxed)
        } else {
            settings.hdr
        };
        if !hdr {
            return None;
        }
        let peak = self.vulkan_share.conversion_options().hdr_peak_nits;
        let default_mastering = format!("{HDR_PRIMARIES}:{}:1", u64::from(peak) * 10000);
        let default_cll = format!("{peak}:{}", (peak * 2 / 5).max(1));
        let (mastering, cll) = if settings.hdr_cm {
            self.hdr_meta.lock().unwrap().clone()
        } else {
            (None, None)
        };
        Some((
            settings
                .mastering_display_info
                .clone()
                .or(mastering)
                .unwrap_or(default_mastering),
            settings
                .content_light_level
                .clone()
                .or(cll)
                .unwrap_or(default_cll),
        ))
    }

    /// Whether the negotiated output is P010, the only format whose caps carry the dynamic
    /// HDR colorimetry.
    fn negotiated_p010(&self) -> Option<gst::Caps> {
        let caps = self.obj().src_pad().current_caps()?;
        // Only P010 caps carry the dynamic HDR colorimetry.
        let s = caps.structure(0)?;
        let p010 = s.get::<&str>("format").ok() == Some("P010_10LE")
            || s.get::<&str>("drm-format")
                .is_ok_and(|f| f.starts_with("P010"));
        p010.then_some(caps)
    }

    /// Apply a compositor HDR-state change to the element: post the bus message on a
    /// transition and retag the P010 caps before the frame converted for the new state is
    /// pushed. Only the colour fields of the current caps change, so the format, size and
    /// converter stay put. Returns `Ok(false)` when downstream refused HDR: the element is back
    /// on SDR caps, the compositor has been told to stay SDR, and the caller must render a
    /// replacement frame.
    fn apply_hdr_state(
        &self,
        hdr: bool,
        mastering: Option<String>,
        cll: Option<String>,
    ) -> Result<bool, gst::FlowError> {
        let prev_hdr = self.hdr_active.swap(hdr, Ordering::Relaxed);
        let prev_meta = std::mem::replace(&mut *self.hdr_meta.lock().unwrap(), (mastering, cll));
        let Some(current) = self.negotiated_p010() else {
            self.post_hdr_transition(prev_hdr, hdr);
            return Ok(true);
        };
        if self.retag_p010(&current) {
            self.post_hdr_transition(prev_hdr, hdr);
            gst::info!(
                CAT,
                imp = self,
                "dynamic-hdr: output caps retagged (hdr={hdr})"
            );
            return Ok(true);
        }
        if prev_hdr == hdr {
            // Updated metadata only; the pixels are unchanged, keep the previous metadata.
            gst::warning!(
                CAT,
                imp = self,
                "dynamic-hdr: downstream refused updated HDR metadata"
            );
            *self.hdr_meta.lock().unwrap() = prev_meta;
            return Ok(true);
        }
        if !hdr {
            gst::error!(CAT, imp = self, "dynamic-hdr: downstream refused SDR caps");
            return Err(gst::FlowError::NotNegotiated);
        }
        gst::warning!(
            CAT,
            imp = self,
            "dynamic-hdr: downstream refused HDR caps; staying SDR for this stream"
        );
        self.hdr_active.store(false, Ordering::Relaxed);
        *self.hdr_meta.lock().unwrap() = (None, None);
        if let Some(state) = self.state.lock().unwrap().as_ref() {
            state.display.refuse_hdr();
        }
        if !self.retag_p010(&current) {
            return Err(gst::FlowError::NotNegotiated);
        }
        Ok(false)
    }

    /// Notify the host only after downstream accepted the new output state.
    fn post_hdr_transition(&self, prev_hdr: bool, hdr: bool) {
        if prev_hdr != hdr {
            let elem = self.obj().upcast_ref::<gst::Element>().to_owned();
            let structure = Structure::builder(HDR_STATE_MESSAGE)
                .field("hdr", hdr)
                .build();
            if let Err(err) = elem.post_message(Application::builder(structure).src(&elem).build())
            {
                gst::warning!(CAT, "Failed to post {HDR_STATE_MESSAGE} message: {}", err);
            }
        }
    }

    /// Re-send `current` with the P010 colour fields of the present HDR state, if downstream
    /// accepts it.
    fn retag_p010(&self, current: &gst::Caps) -> bool {
        let mut caps = current.copy();
        set_p010_color(
            caps.get_mut().unwrap().structure_mut(0).unwrap(),
            &self.p010_hdr_metadata(),
        );
        if caps == *current {
            return true;
        }
        let obj = self.obj();
        obj.src_pad().peer_query_accept_caps(&caps)
            && obj
                .upcast_ref::<gst_base::BaseSrc>()
                .set_caps(&caps)
                .is_ok()
    }
}

/// Colorimetry of the converter's NV12 output: the rgba_to_nv12 shader applies the BT.601
/// limited-range matrix (bit-comparable with the GLES/VA/CUDA paths) to sRGB/BT.709 content.
/// Without it, downstream assumes BT.709 for HD sizes and the colours shift.
fn nv12_colorimetry() -> String {
    gst_video::VideoColorimetry::new(
        gst_video::VideoColorRange::Range16_235,
        gst_video::VideoColorMatrix::Bt601,
        gst_video::VideoTransferFunction::Bt709,
        gst_video::VideoColorPrimaries::Bt709,
    )
    .to_string()
}

/// Bus message posted on dynamic-hdr output transitions, with a boolean `hdr` field.
const HDR_STATE_MESSAGE: &str = "wolf-hdr-state";
/// BT.2020 R/G/B + D65 white chromaticities (x, y * 50000), the default mastering primaries.
const HDR_PRIMARIES: &str = "35400:14600:8500:39850:6550:2300:15635:16450";

/// Stamp the P010 colorimetry onto one caps structure: BT.2100 PQ (primaries=bt2020,
/// transfer=smpte2084, matrix=bt2020, range=tv) plus mastering-display-info (R:G:B:W
/// chromaticity x,y * 50000 + max:min luminance * 10000 cd/m^2) and content-light-level
/// (MaxCLL:MaxFALL) for HDR, so a downstream vulkanh265enc emits the matching VUI + SEI; or
/// BT.709 without static metadata for SDR. The converter's SDR output uses the BT.709 matrix
/// whether or not dynamic HDR is enabled.
fn set_p010_color(s: &mut gst::StructureRef, hdr: &Option<(String, String)>) {
    match hdr {
        Some((mastering, cll)) => {
            s.set("colorimetry", "bt2100-pq");
            s.set("mastering-display-info", mastering.as_str());
            s.set("content-light-level", cll.as_str());
        }
        None => {
            s.set("colorimetry", "bt709");
            s.remove_field("mastering-display-info");
            s.remove_field("content-light-level");
        }
    }
}

/// A `/dev/dri/*` render node backs a real `GstVaDisplay`; the `software` (llvmpipe)
/// target and any other value do not, so VA-display sharing is skipped for them.
fn render_node_backs_va_display(path: &str) -> bool {
    waylanddisplaycore::utils::va_query::render_node_backs_va_display(path)
}

fn drm_to_gst_format(format: &DrmFormat, disable_workaround: bool) -> Option<String> {
    let video_format = format.code.to_string();
    let video_format = video_format.trim();
    if format.modifier == DrmModifier::Linear {
        Some(format!("{:<4}", video_format))
    } else {
        match format.modifier {
            DrmModifier::Invalid => None,
            DrmModifier::Unrecognized(0x0100000000000009) if !disable_workaround => {
                // NOTE: This is a workaround for the i915 4-tiled modifiers
                //       not being advertised by gstreamer elements.
                // - In this part we tell we map any 4-tiled modifiers
                //   to y-tiled ones for compatibility with gstreamer.
                // Continued in wayland-display-core allocator/mod.rs.
                let modifier: u64 = DrmModifier::I915_y_tiled.into();
                Some(format!("{:<4}:0x{:016x}", video_format, modifier))
            }
            modifier => {
                let modifier: u64 = modifier.into();
                Some(format!("{:<4}:0x{:016x}", video_format, modifier))
            }
        }
    }
}

fn gst_button_to_msg(button: i32, state: ButtonState) -> Option<Command> {
    match button as u32 {
        // X11 buttons are internally mapped to some values
        1 => Some(Command::PointerButton(
            input_event_codes_sys::BTN_LEFT,
            state,
        )),
        2 => Some(Command::PointerButton(
            input_event_codes_sys::BTN_MIDDLE,
            state,
        )),
        3 => Some(Command::PointerButton(
            input_event_codes_sys::BTN_RIGHT,
            state,
        )),
        4 => Some(Command::PointerAxis(0.0, -10.0)),
        5 => Some(Command::PointerAxis(0.0, 10.0)),
        // TODO: should we handle these?
        8 => Some(Command::PointerButton(
            input_event_codes_sys::BTN_BACK,
            state,
        )),
        9 => Some(Command::PointerButton(
            input_event_codes_sys::BTN_FORWARD,
            state,
        )),
        // Wayland buttons are just copies from linux input-event-codes.h, so handle them transparently
        input_event_codes_sys::BTN_LEFT => Some(Command::PointerButton(
            input_event_codes_sys::BTN_LEFT,
            state,
        )),
        input_event_codes_sys::BTN_RIGHT => Some(Command::PointerButton(
            input_event_codes_sys::BTN_RIGHT,
            state,
        )),
        input_event_codes_sys::BTN_MIDDLE => Some(Command::PointerButton(
            input_event_codes_sys::BTN_MIDDLE,
            state,
        )),
        input_event_codes_sys::BTN_SIDE => Some(Command::PointerButton(
            input_event_codes_sys::BTN_SIDE,
            state,
        )),
        input_event_codes_sys::BTN_EXTRA => Some(Command::PointerButton(
            input_event_codes_sys::BTN_EXTRA,
            state,
        )),
        input_event_codes_sys::BTN_FORWARD => Some(Command::PointerButton(
            input_event_codes_sys::BTN_FORWARD,
            state,
        )),
        input_event_codes_sys::BTN_BACK => Some(Command::PointerButton(
            input_event_codes_sys::BTN_BACK,
            state,
        )),
        input_event_codes_sys::BTN_WHEEL => Some(Command::PointerButton(
            input_event_codes_sys::BTN_WHEEL,
            state,
        )),
        // TODO: should we handle others?
        _ => None,
    }
}

fn gst_key_to_scancode(key: &str) -> Option<u32> {
    static KEYMAP: LazyLock<HashMap<&str, u32>> = LazyLock::new(|| {
        let mut m = HashMap::new();
        m.insert("Escape", input_event_codes_sys::KEY_ESC);
        m.insert("1", input_event_codes_sys::KEY_1);
        m.insert("exclam", input_event_codes_sys::KEY_1);
        m.insert("2", input_event_codes_sys::KEY_2);
        m.insert("at", input_event_codes_sys::KEY_2);
        m.insert("3", input_event_codes_sys::KEY_3);
        m.insert("numbersign", input_event_codes_sys::KEY_3);
        m.insert("4", input_event_codes_sys::KEY_4);
        m.insert("dollar", input_event_codes_sys::KEY_4);
        m.insert("5", input_event_codes_sys::KEY_5);
        m.insert("percent", input_event_codes_sys::KEY_5);
        m.insert("6", input_event_codes_sys::KEY_6);
        m.insert("asciicircum", input_event_codes_sys::KEY_6);
        m.insert("7", input_event_codes_sys::KEY_7);
        m.insert("ampersand", input_event_codes_sys::KEY_7);
        m.insert("8", input_event_codes_sys::KEY_8);
        m.insert("asterisk", input_event_codes_sys::KEY_8);
        m.insert("9", input_event_codes_sys::KEY_9);
        m.insert("parenleft", input_event_codes_sys::KEY_9);
        m.insert("0", input_event_codes_sys::KEY_0);
        m.insert("parenright", input_event_codes_sys::KEY_0);
        m.insert("minus", input_event_codes_sys::KEY_MINUS);
        m.insert("underscore", input_event_codes_sys::KEY_MINUS);
        m.insert("equal", input_event_codes_sys::KEY_EQUAL);
        m.insert("plus", input_event_codes_sys::KEY_EQUAL);
        m.insert("BackSpace", input_event_codes_sys::KEY_BACKSPACE);
        m.insert("Tab", input_event_codes_sys::KEY_TAB);
        m.insert("Q", input_event_codes_sys::KEY_Q);
        m.insert("q", input_event_codes_sys::KEY_Q);
        m.insert("W", input_event_codes_sys::KEY_W);
        m.insert("w", input_event_codes_sys::KEY_W);
        m.insert("E", input_event_codes_sys::KEY_E);
        m.insert("e", input_event_codes_sys::KEY_E);
        m.insert("R", input_event_codes_sys::KEY_R);
        m.insert("r", input_event_codes_sys::KEY_R);
        m.insert("T", input_event_codes_sys::KEY_T);
        m.insert("t", input_event_codes_sys::KEY_T);
        m.insert("Y", input_event_codes_sys::KEY_Y);
        m.insert("y", input_event_codes_sys::KEY_Y);
        m.insert("U", input_event_codes_sys::KEY_U);
        m.insert("u", input_event_codes_sys::KEY_U);
        m.insert("I", input_event_codes_sys::KEY_I);
        m.insert("i", input_event_codes_sys::KEY_I);
        m.insert("O", input_event_codes_sys::KEY_O);
        m.insert("o", input_event_codes_sys::KEY_O);
        m.insert("P", input_event_codes_sys::KEY_P);
        m.insert("p", input_event_codes_sys::KEY_P);
        m.insert("bracketleft", input_event_codes_sys::KEY_LEFTBRACE);
        m.insert("braceleft", input_event_codes_sys::KEY_LEFTBRACE);
        m.insert("bracketright", input_event_codes_sys::KEY_RIGHTBRACE);
        m.insert("braceright", input_event_codes_sys::KEY_RIGHTBRACE);
        m.insert("Return", input_event_codes_sys::KEY_ENTER);
        m.insert("Control_L", input_event_codes_sys::KEY_LEFTCTRL);
        m.insert("Control_L", input_event_codes_sys::KEY_LEFTCTRL);
        m.insert("A", input_event_codes_sys::KEY_A);
        m.insert("a", input_event_codes_sys::KEY_A);
        m.insert("S", input_event_codes_sys::KEY_S);
        m.insert("s", input_event_codes_sys::KEY_S);
        m.insert("D", input_event_codes_sys::KEY_D);
        m.insert("d", input_event_codes_sys::KEY_D);
        m.insert("F", input_event_codes_sys::KEY_F);
        m.insert("f", input_event_codes_sys::KEY_F);
        m.insert("G", input_event_codes_sys::KEY_G);
        m.insert("g", input_event_codes_sys::KEY_G);
        m.insert("H", input_event_codes_sys::KEY_H);
        m.insert("h", input_event_codes_sys::KEY_H);
        m.insert("J", input_event_codes_sys::KEY_J);
        m.insert("j", input_event_codes_sys::KEY_J);
        m.insert("K", input_event_codes_sys::KEY_K);
        m.insert("k", input_event_codes_sys::KEY_K);
        m.insert("L", input_event_codes_sys::KEY_L);
        m.insert("l", input_event_codes_sys::KEY_L);
        m.insert("semicolon", input_event_codes_sys::KEY_SEMICOLON);
        m.insert("colon", input_event_codes_sys::KEY_SEMICOLON);
        m.insert("apostrophe", input_event_codes_sys::KEY_APOSTROPHE);
        m.insert("quotedbl", input_event_codes_sys::KEY_APOSTROPHE);
        m.insert("grave", input_event_codes_sys::KEY_GRAVE);
        m.insert("grave", input_event_codes_sys::KEY_GRAVE);
        m.insert("asciitilde", input_event_codes_sys::KEY_GRAVE);
        m.insert("asciitilde", input_event_codes_sys::KEY_GRAVE);
        m.insert("Shift_L", input_event_codes_sys::KEY_LEFTSHIFT);
        m.insert("backslash", input_event_codes_sys::KEY_BACKSLASH);
        m.insert("backslash", input_event_codes_sys::KEY_BACKSLASH);
        m.insert("bar", input_event_codes_sys::KEY_BACKSLASH);
        m.insert("bar", input_event_codes_sys::KEY_BACKSLASH);
        m.insert("backslash", input_event_codes_sys::KEY_BACKSLASH);
        m.insert("backslash", input_event_codes_sys::KEY_BACKSLASH);
        m.insert("bar", input_event_codes_sys::KEY_BACKSLASH);
        m.insert("bar", input_event_codes_sys::KEY_BACKSLASH);
        m.insert("Z", input_event_codes_sys::KEY_Z);
        m.insert("z", input_event_codes_sys::KEY_Z);
        m.insert("X", input_event_codes_sys::KEY_X);
        m.insert("x", input_event_codes_sys::KEY_X);
        m.insert("C", input_event_codes_sys::KEY_C);
        m.insert("c", input_event_codes_sys::KEY_C);
        m.insert("V", input_event_codes_sys::KEY_V);
        m.insert("v", input_event_codes_sys::KEY_V);
        m.insert("B", input_event_codes_sys::KEY_B);
        m.insert("b", input_event_codes_sys::KEY_B);
        m.insert("N", input_event_codes_sys::KEY_N);
        m.insert("n", input_event_codes_sys::KEY_N);
        m.insert("M", input_event_codes_sys::KEY_M);
        m.insert("m", input_event_codes_sys::KEY_M);
        m.insert("comma", input_event_codes_sys::KEY_COMMA);
        m.insert("less", input_event_codes_sys::KEY_COMMA);
        m.insert("period", input_event_codes_sys::KEY_DOT);
        m.insert("greater", input_event_codes_sys::KEY_DOT);
        m.insert("slash", input_event_codes_sys::KEY_SLASH);
        m.insert("question", input_event_codes_sys::KEY_SLASH);
        m.insert("Shift_R", input_event_codes_sys::KEY_RIGHTSHIFT);
        m.insert("KP_Multiplymultiply", input_event_codes_sys::KEY_KPASTERISK);
        m.insert("multiply", input_event_codes_sys::KEY_KPASTERISK);
        m.insert("KP_Multiply", input_event_codes_sys::KEY_KPASTERISK);
        m.insert("multiply", input_event_codes_sys::KEY_KPASTERISK);
        m.insert("Alt_L", input_event_codes_sys::KEY_LEFTALT);
        m.insert("Alt_L", input_event_codes_sys::KEY_LEFTALT);
        m.insert("space", input_event_codes_sys::KEY_SPACE);
        m.insert("Caps_Lock", input_event_codes_sys::KEY_CAPSLOCK);
        m.insert("F1", input_event_codes_sys::KEY_F1);
        m.insert("F2", input_event_codes_sys::KEY_F2);
        m.insert("F3", input_event_codes_sys::KEY_F3);
        m.insert("F4", input_event_codes_sys::KEY_F4);
        m.insert("F5", input_event_codes_sys::KEY_F5);
        m.insert("F6", input_event_codes_sys::KEY_F6);
        m.insert("F7", input_event_codes_sys::KEY_F7);
        m.insert("F8", input_event_codes_sys::KEY_F8);
        m.insert("F9", input_event_codes_sys::KEY_F9);
        m.insert("F10", input_event_codes_sys::KEY_F10);
        m.insert("Num_Lock", input_event_codes_sys::KEY_NUMLOCK);
        m.insert("Scroll_Lock", input_event_codes_sys::KEY_SCROLLLOCK);
        m.insert("KP_Home", input_event_codes_sys::KEY_KP7);
        m.insert("KP_7", input_event_codes_sys::KEY_KP7);
        m.insert("KP_Up", input_event_codes_sys::KEY_KP8);
        m.insert("KP_8", input_event_codes_sys::KEY_KP8);
        m.insert("KP_Prior", input_event_codes_sys::KEY_KP9);
        m.insert("KP_9", input_event_codes_sys::KEY_KP9);
        m.insert("KP_Subtract", input_event_codes_sys::KEY_KPMINUS);
        m.insert("KP_Left", input_event_codes_sys::KEY_KP4);
        m.insert("KP_4", input_event_codes_sys::KEY_KP4);
        m.insert("KP_Begin", input_event_codes_sys::KEY_KP5);
        m.insert("KP_5", input_event_codes_sys::KEY_KP5);
        m.insert("KP_Right", input_event_codes_sys::KEY_KP6);
        m.insert("KP_6", input_event_codes_sys::KEY_KP6);
        m.insert("KP_Add", input_event_codes_sys::KEY_KPPLUS);
        m.insert("KP_End", input_event_codes_sys::KEY_KP1);
        m.insert("KP_1", input_event_codes_sys::KEY_KP1);
        m.insert("KP_Down", input_event_codes_sys::KEY_KP2);
        m.insert("KP_2", input_event_codes_sys::KEY_KP2);
        m.insert("KP_Next", input_event_codes_sys::KEY_KP3);
        m.insert("KP_3", input_event_codes_sys::KEY_KP3);
        m.insert("KP_Insert", input_event_codes_sys::KEY_KP0);
        m.insert("KP_0", input_event_codes_sys::KEY_KP0);
        m.insert("KP_Delete", input_event_codes_sys::KEY_KPDOT);
        m.insert("KP_Delete", input_event_codes_sys::KEY_KPDOT);
        m.insert("KP_Decimal", input_event_codes_sys::KEY_KPDOT);
        m.insert("KP_Decimal", input_event_codes_sys::KEY_KPDOT);
        m.insert("Zenkaku_Hankaku", input_event_codes_sys::KEY_ZENKAKUHANKAKU);
        m.insert("F11", input_event_codes_sys::KEY_F11);
        m.insert("F12", input_event_codes_sys::KEY_F12);
        m.insert("underscore", input_event_codes_sys::KEY_RO);
        m.insert("Katakana", input_event_codes_sys::KEY_KATAKANA);
        m.insert("Katakana", input_event_codes_sys::KEY_KATAKANA);
        m.insert("Hiragana", input_event_codes_sys::KEY_HIRAGANA);
        m.insert("Hiragana", input_event_codes_sys::KEY_HIRAGANA);
        m.insert("Henkan", input_event_codes_sys::KEY_HENKAN);
        m.insert(
            "Hiragana_Katakana",
            input_event_codes_sys::KEY_KATAKANAHIRAGANA,
        );
        m.insert("Muhenkan", input_event_codes_sys::KEY_MUHENKAN);
        m.insert("Muhenkan", input_event_codes_sys::KEY_MUHENKAN);
        m.insert("KP_Separator", input_event_codes_sys::KEY_KPJPCOMMA);
        m.insert("KP_Separator", input_event_codes_sys::KEY_KPJPCOMMA);
        m.insert("KP_Enter", input_event_codes_sys::KEY_KPENTER);
        m.insert("Control_R", input_event_codes_sys::KEY_RIGHTCTRL);
        m.insert("KP_Divide", input_event_codes_sys::KEY_KPSLASH);
        m.insert("Sys_Req", input_event_codes_sys::KEY_SYSRQ);
        m.insert("Sys_Req", input_event_codes_sys::KEY_SYSRQ);
        m.insert("Alt_R", input_event_codes_sys::KEY_RIGHTALT);
        m.insert("Alt_R", input_event_codes_sys::KEY_RIGHTALT);
        m.insert("Home", input_event_codes_sys::KEY_HOME);
        m.insert("Up", input_event_codes_sys::KEY_UP);
        m.insert("Prior", input_event_codes_sys::KEY_PAGEUP);
        m.insert("Page_Up", input_event_codes_sys::KEY_PAGEUP);
        m.insert("Left", input_event_codes_sys::KEY_LEFT);
        m.insert("Right", input_event_codes_sys::KEY_RIGHT);
        m.insert("End", input_event_codes_sys::KEY_END);
        m.insert("Down", input_event_codes_sys::KEY_DOWN);
        m.insert("Next", input_event_codes_sys::KEY_PAGEDOWN);
        m.insert("Page_Down", input_event_codes_sys::KEY_PAGEDOWN);
        m.insert("Insert", input_event_codes_sys::KEY_INSERT);
        m.insert("Delete", input_event_codes_sys::KEY_DELETE);
        m.insert("Delete", input_event_codes_sys::KEY_DELETE);
        m.insert("KP_Equal", input_event_codes_sys::KEY_KPEQUAL);
        m.insert("Pause", input_event_codes_sys::KEY_PAUSE);
        m.insert("Meta_L", input_event_codes_sys::KEY_LEFTMETA);
        m.insert("Meta_L", input_event_codes_sys::KEY_LEFTMETA);
        m.insert("Super_L", input_event_codes_sys::KEY_LEFTMETA);
        m.insert("Meta_R", input_event_codes_sys::KEY_RIGHTMETA);
        m.insert("Meta_R", input_event_codes_sys::KEY_RIGHTMETA);
        m.insert("Super_R", input_event_codes_sys::KEY_RIGHTMETA);
        m.insert("Help", input_event_codes_sys::KEY_HELP);
        m.insert("Select", input_event_codes_sys::KEY_SELECT);

        m
    });
    KEYMAP.get(key).copied()
}

#[cfg(test)]
mod tests {
    use waylanddisplaycore::DrmFormat;
    use waylanddisplaycore::utils::tests::INIT;

    /// Paths the compositor logged as "Adding input device.", across all tests in this binary.
    static ADDED_INPUT_DEVICES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

    thread_local! {
        /// Where input_device_identity() looks devices up. Per thread, as start() and event
        /// handling run on the test's thread, so a fake tree doesn't leak into other tests.
        pub(super) static SYSFS_ROOT: std::cell::RefCell<std::path::PathBuf> =
            std::cell::RefCell::new("/sys".into());
    }

    /// Records the paths of the compositor's "Adding input device." events.
    struct AddedInputDevices;

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for AddedInputDevices {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            #[derive(Default)]
            struct Fields {
                message: String,
                path: Option<String>,
            }
            impl tracing::field::Visit for Fields {
                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    if field.name() == "path" {
                        self.path = Some(value.to_string());
                    }
                }
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.message = format!("{value:?}");
                    }
                }
            }
            let mut fields = Fields::default();
            event.record(&mut fields);
            if let (true, Some(path)) = (fields.message == "Adding input device.", fields.path) {
                ADDED_INPUT_DEVICES.lock().unwrap().push(path);
            }
        }
    }

    fn test_init() -> () {
        use tracing_subscriber::Layer;
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        INIT.call_once(|| {
            // Global, so the compositor threads' events reach AddedInputDevices too.
            tracing_subscriber::registry()
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_filter(tracing_subscriber::filter::LevelFilter::INFO),
                )
                .with(AddedInputDevices)
                .try_init()
                .ok();
            gst::init().expect("Failed to initialize GStreamer");
        });
    }

    #[test]
    fn rejects_invalid_caps_without_starting_a_device() {
        use gst::prelude::*;
        use gst::subclass::prelude::*;
        use gst_base::subclass::prelude::*;
        test_init();
        crate::plugin_register_static().ok();
        let src = gst::ElementFactory::make("waylanddisplaysrc")
            .build()
            .unwrap()
            .downcast::<crate::waylandsrc::WaylandDisplaySrc>()
            .unwrap();
        let caps: gst::Caps = "video/x-raw(memory:VulkanImage),format=NV12"
            .parse()
            .unwrap();
        assert!(src.imp().set_caps(&caps).is_err());
        assert!(src.imp().vulkan_share.shared_device().is_none());
        let caps: gst::Caps = "video/x-raw,format=RGBx".parse().unwrap();
        assert!(src.imp().set_caps(&caps).is_err());
    }

    #[test]
    fn conversion_properties_are_per_source() {
        use gst::prelude::*;
        use gst::subclass::prelude::*;
        test_init();
        crate::plugin_register_static().ok();
        let make = || {
            gst::ElementFactory::make("waylanddisplaysrc")
                .build()
                .unwrap()
                .downcast::<crate::waylandsrc::WaylandDisplaySrc>()
                .unwrap()
        };
        let first = make();
        let second = make();
        let original = second.imp().vulkan_share.conversion_options();
        first.set_property("ring-depth", 2u32);
        first.set_property("linear-encode-src", true);
        first.set_property("sdr-reference-white", 150f32);
        assert_eq!(first.property::<u32>("ring-depth"), 2);
        assert!(first.property::<bool>("linear-encode-src"));
        assert_eq!(first.property::<f32>("sdr-reference-white"), 150.0);
        let other = second.imp().vulkan_share.conversion_options();
        assert_eq!(other.ring_depth, original.ring_depth);
        assert_eq!(other.sdr_reference_white, original.sdr_reference_white);
        assert_eq!(other.linear_encode_src, original.linear_encode_src);
    }

    #[test]
    fn test_drm_format_to_gstreamer() {
        test_init();

        assert_eq!(
            super::drm_to_gst_format(
                &DrmFormat {
                    code: waylanddisplaycore::Fourcc::Abgr8888,
                    modifier: waylanddisplaycore::DrmModifier::Linear
                },
                false
            ),
            Some("AB24".to_string())
        );

        assert_eq!(
            super::drm_to_gst_format(
                &DrmFormat {
                    code: waylanddisplaycore::Fourcc::R8,
                    modifier: waylanddisplaycore::DrmModifier::Linear
                },
                false
            ),
            Some("R8  ".to_string())
        );

        assert_eq!(
            super::drm_to_gst_format(
                &DrmFormat {
                    code: waylanddisplaycore::Fourcc::Rgba8888,
                    modifier: waylanddisplaycore::DrmModifier::Nvidia_16bx2_block_eight_gob
                },
                false
            ),
            Some("RA24:0x0300000000000013".to_string())
        );
    }

    // the first start() consumed the command receiver and stop() never replaced it,
    // so the second start() panicked and the state change failed.
    #[test]
    fn restart_reaches_playing_again() {
        use gst::prelude::*;
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>()
            .property("render-node", "software")
            .build()
            .upcast::<gst::Element>();
        let caps = gst::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gst::Caps::builder("video/x-raw")
                    .field("width", 320)
                    .field("height", 240)
                    .field("framerate", gst::Fraction::new(30, 1))
                    .build(),
            )
            .build()
            .unwrap();
        let sink = gst::ElementFactory::make("fakesink").build().unwrap();
        let pipeline = gst::Pipeline::new();
        pipeline.add_many([&src, &caps, &sink]).unwrap();
        gst::Element::link_many([&src, &caps, &sink]).unwrap();

        for run in 0..2 {
            pipeline
                .set_state(gst::State::Playing)
                .unwrap_or_else(|err| panic!("run {run}: set_state(Playing) failed: {err}"));
            let (result, current, _) = pipeline.state(gst::ClockTime::from_seconds(10));
            assert!(result.is_ok(), "run {run}: state change failed");
            assert_eq!(current, gst::State::Playing, "run {run}");
            pipeline.set_state(gst::State::Null).unwrap();
        }
    }

    // A start() that fails after consuming the receiver must not break the next start().
    #[test]
    fn start_after_failed_start_reaches_playing() {
        use gst::prelude::*;
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>()
            .property("render-node", "/nonexistent/renderD128")
            .build()
            .upcast::<gst::Element>();
        let sink = gst::ElementFactory::make("fakesink").build().unwrap();
        let pipeline = gst::Pipeline::new();
        pipeline.add_many([&src, &sink]).unwrap();
        src.link(&sink).unwrap();

        assert!(pipeline.set_state(gst::State::Playing).is_err());
        pipeline.set_state(gst::State::Null).unwrap();

        src.set_property("render-node", "software");
        pipeline.set_state(gst::State::Playing).unwrap();
        let (result, current, _) = pipeline.state(gst::ClockTime::from_seconds(10));
        assert!(result.is_ok(), "state change failed");
        assert_eq!(current, gst::State::Playing);
        pipeline.set_state(gst::State::Null).unwrap();
    }

    // Commands sent while no display runs must fail rather than queue and replay into the next
    // run, and VirtualDevicesReady devices must be kept for the next start().
    #[test]
    fn stopped_element_drops_input_and_keeps_virtual_devices() {
        use gst::prelude::*;
        use gst::subclass::prelude::*;
        use std::sync::atomic::Ordering;
        use waylanddisplaycore::Command;
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>()
            .property("render-node", "software")
            .build();
        let imp = src.imp();
        let sink = gst::ElementFactory::make("fakesink").build().unwrap();
        let pipeline = gst::Pipeline::new();
        pipeline.add_many([src.upcast_ref(), &sink]).unwrap();
        src.link(&sink).unwrap();

        let stopped_send = |imp: &super::WaylandDisplaySrc| {
            imp.command_tx.lock().unwrap().send(Command::TouchFrame)
        };
        assert!(
            stopped_send(imp).is_err(),
            "queued before the first start()"
        );

        let device = gst::event::CustomUpstream::new(
            gst::Structure::builder("VirtualDevicesReady")
                .field("path", "/nonexistent/input/event0")
                .build(),
        );
        for _ in 0..2 {
            // As if the previous run ended in dynamic HDR.
            imp.hdr_active.store(true, Ordering::Relaxed);
            *imp.hdr_meta.lock().unwrap() = (Some("m".into()), Some("c".into()));
            pipeline.set_state(gst::State::Playing).unwrap();
            let (result, _, _) = pipeline.state(gst::ClockTime::from_seconds(10));
            assert!(result.is_ok());
            assert!(
                !imp.hdr_active.load(Ordering::Relaxed),
                "HDR state kept across start()"
            );
            assert_eq!(*imp.hdr_meta.lock().unwrap(), (None, None));
            assert!(src.send_event(device.clone()));
            pipeline.set_state(gst::State::Null).unwrap();
            assert!(stopped_send(imp).is_err(), "queued after stop()");
        }
        assert_eq!(
            imp.settings.lock().unwrap().startup_input_devices(),
            ["/nonexistent/input/event0"]
        );
    }

    // A reconnected device can reuse its old /dev/input path, so every announcement must reach
    // the running display, not only the first one for a path.
    #[test]
    fn virtual_devices_ready_forwards_every_announcement() {
        use super::EventHandler;
        use gst::subclass::prelude::*;
        use waylanddisplaycore::Command;
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>().build();
        let imp = src.imp();
        // Stands in for a running display's receiver.
        let command_rx = imp.new_command_channel();
        let device = gst::event::CustomUpstream::new(
            gst::Structure::builder("VirtualDevicesReady")
                .field("path", "/dev/input/event7")
                .build(),
        );
        for _ in 0..2 {
            assert!(imp.handle_event(&device));
        }

        let mut forwarded = Vec::new();
        while let Ok(command) = command_rx.try_recv() {
            if let Command::InputDevice(path) = command {
                forwarded.push(path);
            }
        }
        assert_eq!(forwarded, ["/dev/input/event7", "/dev/input/event7"]);
        assert_eq!(
            imp.settings.lock().unwrap().startup_input_devices(),
            ["/dev/input/event7"]
        );
    }

    // start() re-added every announced path, including unplugged devices and event
    // numbers the kernel had since given to another device.
    #[test]
    #[ignore = "needs /dev/uinput and udev; run with --ignored"]
    fn restart_drops_virtual_devices_that_changed() {
        use super::EventHandler;
        use gst::subclass::prelude::*;
        use waylanddisplaycore::utils::uinput::UinputKeyboard;
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>().build();
        let imp = src.imp();
        let _command_rx = imp.new_command_channel();
        let announce = |path: &str| {
            assert!(
                imp.handle_event(&gst::event::CustomUpstream::new(
                    gst::Structure::builder("VirtualDevicesReady")
                        .field("path", path)
                        .build(),
                ))
            );
        };
        let startup_devices = || {
            let mut settings = imp.settings.lock().unwrap();
            settings.prune_virtual_input_devices();
            settings.startup_input_devices()
        };

        let kept = UinputKeyboard::new();
        let unplugged = UinputKeyboard::new();
        let unplugged_path = unplugged.path.clone();
        announce(&kept.path);
        announce(&unplugged_path);
        drop(unplugged);
        assert_eq!(startup_devices(), [kept.path.clone()]);

        // The kernel hands out the lowest free event number, so this takes the dropped one.
        let other = UinputKeyboard::new();
        announce(&other.path);
        drop(other);
        let reused = UinputKeyboard::new();
        assert_eq!(
            startup_devices(),
            [kept.path.clone()],
            "re-added {} (now another device: {})",
            reused.path,
            reused.path == unplugged_path
        );

        // Announcing the device that now owns the path keeps it again.
        announce(&reused.path);
        assert_eq!(startup_devices(), [kept.path.clone(), reused.path.clone()]);
    }

    // start() re-added every announced path, including unplugged devices and event
    // numbers since given to another device. Runs without hardware on a fake sysfs tree.
    #[test]
    fn restart_skips_virtual_devices_that_are_gone_or_reassigned() {
        use gst::prelude::*;
        use std::os::unix::fs::symlink;
        use std::time::{Duration, Instant};
        test_init();

        let root = std::env::temp_dir().join(format!("wlsrc-sysfs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let node = |n: u32| root.join(format!("dev/event{n}"));
        let link = |n: u32| root.join(format!("sys/class/input/event{n}/device"));
        std::fs::create_dir_all(root.join("dev")).unwrap();
        for n in 4..=7 {
            std::fs::write(node(n), "").unwrap();
            std::fs::create_dir_all(root.join(format!("sys/devices/input{n}"))).unwrap();
            std::fs::create_dir_all(link(n).parent().unwrap()).unwrap();
            symlink(root.join(format!("sys/devices/input{n}")), link(n)).unwrap();
        }
        SYSFS_ROOT.with(|sysfs| *sysfs.borrow_mut() = root.join("sys"));

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>()
            .property("render-node", "software")
            .build()
            .upcast::<gst::Element>();
        let sink = gst::ElementFactory::make("fakesink").build().unwrap();
        let pipeline = gst::Pipeline::new();
        pipeline.add_many([&src, &sink]).unwrap();
        src.link(&sink).unwrap();
        let path = |n: u32| node(n).to_str().unwrap().to_owned();
        let adds = |n: u32| {
            ADDED_INPUT_DEVICES
                .lock()
                .unwrap()
                .iter()
                .filter(|added| **added == path(n))
                .count()
        };
        let wait_for_adds = |n: u32, count: usize| {
            let deadline = Instant::now() + Duration::from_secs(5);
            while adds(n) < count {
                assert!(Instant::now() < deadline, "event{n} was not added");
                std::thread::sleep(Duration::from_millis(10));
            }
        };

        let announce = |path: &str| {
            assert!(
                src.send_event(gst::event::CustomUpstream::new(
                    gst::Structure::builder("VirtualDevicesReady")
                        .field("path", path)
                        .build(),
                ))
            );
        };
        let reassign = |n: u32, to: u32| {
            std::fs::create_dir_all(root.join(format!("sys/devices/input{to}"))).unwrap();
            std::fs::remove_file(link(n)).unwrap();
            symlink(root.join(format!("sys/devices/input{to}")), link(n)).unwrap();
        };
        // No sysfs entry, so its identity is unknown: kept, as before.
        let unknown = root.join("dev/unknown").to_str().unwrap().to_owned();

        pipeline.set_state(gst::State::Playing).unwrap();
        // Unplugged, silently reassigned, reconnected, unknown, kept: added in this order.
        for n in [6, 4, 7] {
            announce(&path(n));
        }
        announce(&unknown);
        announce(&path(5));
        wait_for_adds(5, 1);
        // event7 reconnects on its old path while streaming and is announced again.
        reassign(7, 17);
        announce(&path(7));
        wait_for_adds(7, 2);
        pipeline.set_state(gst::State::Null).unwrap();

        std::fs::remove_file(node(6)).unwrap();
        reassign(4, 14);
        pipeline.set_state(gst::State::Playing).unwrap();
        wait_for_adds(5, 2);
        pipeline.set_state(gst::State::Null).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        let unknown_adds = ADDED_INPUT_DEVICES
            .lock()
            .unwrap()
            .iter()
            .filter(|added| **added == unknown)
            .count();
        assert_eq!(adds(6), 1, "re-added an unplugged device");
        assert_eq!(
            adds(4),
            1,
            "re-added a path now belonging to another device"
        );
        assert_eq!(
            adds(7),
            3,
            "dropped a device announced again on its old path"
        );
        assert_eq!(
            unknown_adds, 2,
            "dropped a device whose identity was unknown"
        );
    }

    // create() must not hold the state lock while it waits for the frame, or caps queries
    // queue behind a slow or wedged compositor for up to the reply timeout.
    #[test]
    fn create_releases_state_while_waiting_for_frame() {
        use gst::subclass::prelude::*;
        use gst_base::subclass::prelude::*;
        use std::time::{Duration, Instant};
        use waylanddisplaycore::{Command, WaylandDisplay, channel};
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>().build();
        // The display sends to `unserviced`, which only this test reads; the compositor runs on
        // its own channel, so frame requests stay pending until the test drops them.
        let (command_tx, unserviced) = channel::<Command>();
        let (compositor_tx, compositor_rx) = channel::<Command>();
        let display = WaylandDisplay::new_with_channel(
            Some("software".into()),
            command_tx,
            compositor_rx,
            std::sync::Arc::clone(&src.imp().vulkan_share),
            std::sync::Arc::clone(&src.imp().va_share),
        )
        .unwrap();
        *src.imp().state.lock().unwrap() = Some(super::State { display });

        let creating = src.clone();
        let create = std::thread::spawn(move || PushSrcImpl::create(creating.imp(), None).is_err());

        let deadline = Instant::now() + Duration::from_secs(5);
        let pending_frame = loop {
            if let Ok(command @ Command::Buffer(..)) = unserviced.try_recv() {
                break command;
            }
            assert!(
                Instant::now() < deadline,
                "create() never requested a frame"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        // Give a regressed create() time to take the lock back around its wait.
        std::thread::sleep(Duration::from_millis(100));
        let state_free = src.imp().state.try_lock().is_ok();

        // Dropping the request ends create()'s wait; then shut the compositor down.
        drop(pending_frame);
        assert!(
            create.join().unwrap(),
            "an unanswered frame must be an error"
        );
        compositor_tx.send(Command::Quit).unwrap();
        // Drop the display while `unserviced` still exists: its own Quit must send for Drop to
        // join the compositor thread, or the thread outlives the test and races process exit.
        src.imp().state.lock().unwrap().take();
        drop(unserviced);
        assert!(
            state_free,
            "create() held the state lock while waiting for the frame"
        );
    }

    // start() must hand a new display the configured devices and the announced ones;
    // nothing caught dropping that hand-off.
    #[test]
    fn startup_channel_carries_configured_and_announced_devices() {
        use super::EventHandler;
        use gst::subclass::prelude::*;
        use waylanddisplaycore::Command;
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>()
            .property("keyboard", "/dev/input/event1")
            .property("mouse", "/dev/input/event2")
            .build();
        let imp = src.imp();
        assert!(
            imp.handle_event(&gst::event::CustomUpstream::new(
                gst::Structure::builder("VirtualDevicesReady")
                    .field("path", "/dev/input/event3")
                    .build(),
            ))
        );

        let command_rx = imp.new_startup_channel(&imp.settings.lock().unwrap());
        let mut queued = Vec::new();
        while let Ok(command) = command_rx.try_recv() {
            if let Command::InputDevice(path) = command {
                queued.push(path);
            }
        }
        // The builder sets properties in no particular order.
        queued.sort();
        assert_eq!(
            queued,
            [
                "/dev/input/event1",
                "/dev/input/event2",
                "/dev/input/event3"
            ]
        );
    }

    /// Puts a display in `src`'s state whose commands go to the returned channel, which only the
    /// test reads; the compositor runs on the returned sender's channel. Pass both to
    /// `drop_unserviced_display` when done.
    fn unserviced_display(
        src: &super::super::WaylandDisplaySrc,
    ) -> (
        waylanddisplaycore::Channel<waylanddisplaycore::Command>,
        waylanddisplaycore::Sender<waylanddisplaycore::Command>,
    ) {
        use gst::subclass::prelude::*;
        use waylanddisplaycore::{Command, WaylandDisplay, channel};

        let (command_tx, unserviced) = channel::<Command>();
        let (compositor_tx, compositor_rx) = channel::<Command>();
        let display = WaylandDisplay::new_with_channel(
            Some("software".into()),
            command_tx,
            compositor_rx,
            std::sync::Arc::clone(&src.imp().vulkan_share),
            std::sync::Arc::clone(&src.imp().va_share),
        )
        .unwrap();
        *src.imp().state.lock().unwrap() = Some(super::State { display });
        (unserviced, compositor_tx)
    }

    fn drop_unserviced_display(
        src: &super::super::WaylandDisplaySrc,
        unserviced: waylanddisplaycore::Channel<waylanddisplaycore::Command>,
        compositor_tx: waylanddisplaycore::Sender<waylanddisplaycore::Command>,
    ) {
        use gst::subclass::prelude::*;
        compositor_tx
            .send(waylanddisplaycore::Command::Quit)
            .unwrap();
        // Drop the display while `unserviced` still exists: its own Quit must send for Drop to
        // join the compositor thread, or the thread outlives the test and races process exit.
        src.imp().state.lock().unwrap().take();
        drop(unserviced);
    }

    /// Waits for the display to send a command matching `matches` to `unserviced`.
    fn next_command(
        unserviced: &waylanddisplaycore::Channel<waylanddisplaycore::Command>,
        matches: impl Fn(&waylanddisplaycore::Command) -> bool,
    ) -> waylanddisplaycore::Command {
        use std::time::{Duration, Instant};
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok(command) = unserviced.try_recv() {
                if matches(&command) {
                    return command;
                }
            }
            assert!(Instant::now() < deadline, "the expected command never came");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    // A caps query the compositor fails to answer must post an element error and fail the query,
    // not report caps without DMA formats.
    #[test]
    fn caps_fails_when_compositor_does_not_report_dma_formats() {
        use gst::prelude::*;
        use gst_base::subclass::prelude::*;
        use waylanddisplaycore::Command;
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>().build();
        let pipeline = gst::Pipeline::new();
        pipeline.add(&src).unwrap();
        let bus = pipeline.bus().unwrap();
        let (unserviced, compositor_tx) = unserviced_display(&src);

        let querying = src.clone();
        let caps = std::thread::spawn(move || BaseSrcImpl::caps(querying.imp(), None));
        // Dropping the request unanswered is what a compositor that dies mid-query does.
        drop(next_command(&unserviced, |command| {
            matches!(command, Command::GetSupportedDmaFormats(_))
        }));
        assert!(caps.join().unwrap().is_none());
        let error = bus
            .timed_pop_filtered(gst::ClockTime::from_seconds(5), &[gst::MessageType::Error])
            .expect("no element error posted");
        let gst::MessageView::Error(error) = error.view() else {
            unreachable!()
        };
        assert!(
            error.error().matches(gst::ResourceError::Failed),
            "{:?}",
            error.error()
        );

        drop_unserviced_display(&src, unserviced, compositor_tx);
    }

    // GstBaseSrc joins the streaming thread before stop(), after calling unlock(); a
    // create() still waiting on a wedged compositor held that up for the full reply timeout.
    #[test]
    fn unlock_ends_create_waiting_for_frame() {
        use gst_base::subclass::prelude::*;
        use std::time::{Duration, Instant};
        use waylanddisplaycore::Command;
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>().build();
        let (unserviced, compositor_tx) = unserviced_display(&src);

        let creating = src.clone();
        let create = std::thread::spawn(move || PushSrcImpl::create(creating.imp(), None).err());
        // Held, never answered: the frame request stays pending like on a wedged compositor.
        let pending_frame = next_command(&unserviced, |command| {
            matches!(command, Command::Buffer(..))
        });

        let unlocked = Instant::now();
        BaseSrcImpl::unlock(src.imp()).unwrap();
        assert_eq!(create.join().unwrap(), Some(gst::FlowError::Flushing));
        assert!(unlocked.elapsed() < Duration::from_secs(1));
        // Still flushing: a create() racing in after unlock() must not start a new wait.
        assert_eq!(
            PushSrcImpl::create(src.imp(), None).err(),
            Some(gst::FlowError::Flushing)
        );
        // unlock_stop() resumes waiting on the same frame: dropping it would make the compositor
        // fail to deliver it and shut down.
        BaseSrcImpl::unlock_stop(src.imp()).unwrap();
        let creating = src.clone();
        let create = std::thread::spawn(move || PushSrcImpl::create(creating.imp(), None).is_ok());
        let Command::Buffer(buffer_tx, _) = pending_frame else {
            unreachable!()
        };
        buffer_tx.send(Ok(gst::Buffer::new())).unwrap();
        assert!(create.join().unwrap(), "the kept frame was not delivered");
        assert!(
            unserviced.try_recv().is_err(),
            "requested another frame instead of waiting on the kept one"
        );
        drop_unserviced_display(&src, unserviced, compositor_tx);
    }

    // a pad-level flush doesn't take the state lock, so it can land inside a
    // PLAYING->PAUSED change_state(), where unlock() ignores the call; flush_stop then blocked
    // for the compositor reply timeout behind the create() it failed to cancel.
    #[test]
    fn pad_flush_during_pause_ends_create_waiting_for_frame() {
        use gst_base::subclass::prelude::*;
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};
        use waylanddisplaycore::Command;
        test_init();

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>().build();
        let (unserviced, compositor_tx) = unserviced_display(&src);

        let creating = src.clone();
        let create = std::thread::spawn(move || PushSrcImpl::create(creating.imp(), None).err());
        let pending_frame = next_command(&unserviced, |command| {
            matches!(command, Command::Buffer(..))
        });

        // As if change_state(PLAYING->PAUSED) were still running.
        src.imp().pausing.store(true, Ordering::Release);
        let flushed = Instant::now();
        // What the src pad's event handler calls; the pad itself drops events while inactive.
        BaseSrcImpl::event(src.imp(), &gst::event::FlushStart::new());
        assert_eq!(create.join().unwrap(), Some(gst::FlowError::Flushing));
        assert!(flushed.elapsed() < Duration::from_secs(1));
        src.imp().pausing.store(false, Ordering::Release);

        // The flush's stop must lift it, or create() would return FLUSHING for good.
        BaseSrcImpl::event(src.imp(), &gst::event::FlushStop::new(true));
        let creating = src.clone();
        let create = std::thread::spawn(move || PushSrcImpl::create(creating.imp(), None).is_ok());
        let Command::Buffer(buffer_tx, _) = pending_frame else {
            unreachable!()
        };
        buffer_tx.send(Ok(gst::Buffer::new())).unwrap();
        assert!(
            create.join().unwrap(),
            "create() failed after the flush stopped"
        );
        drop_unserviced_display(&src, unserviced, compositor_tx);
    }

    /// A software-rendered live pipeline whose sink counts the frames it receives.
    fn counting_pipeline() -> (
        gst::Pipeline,
        gst::Element,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use gst::prelude::*;
        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>()
            .property("render-node", "software")
            .build()
            .upcast::<gst::Element>();
        counting_pipeline_with(src, true)
    }

    /// `src` (optionally through a 320x240@30 raw video capsfilter) into a fakesink that counts
    /// the buffers it receives. The sink doesn't sync: a flush-stop resets the running time, and a
    /// syncing sink on a live pipeline then waits on the clock instead of counting.
    fn counting_pipeline_with(
        src: gst::Element,
        video_caps: bool,
    ) -> (
        gst::Pipeline,
        gst::Element,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use gst::prelude::*;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let sink = gst::ElementFactory::make("fakesink")
            .property("signal-handoffs", true)
            .property("sync", false)
            .build()
            .unwrap();
        let frames = Arc::new(AtomicUsize::new(0));
        let counter = frames.clone();
        sink.connect("handoff", false, move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            None
        });
        let pipeline = gst::Pipeline::new();
        let mut chain = vec![src.clone()];
        if video_caps {
            chain.push(
                gst::ElementFactory::make("capsfilter")
                    .property(
                        "caps",
                        gst::Caps::builder("video/x-raw")
                            .field("width", 320)
                            .field("height", 240)
                            .field("framerate", gst::Fraction::new(30, 1))
                            .build(),
                    )
                    .build()
                    .unwrap(),
            );
        }
        chain.push(sink);
        pipeline.add_many(&chain).unwrap();
        gst::Element::link_many(&chain).unwrap();
        (pipeline, src, frames)
    }

    /// Flushes a playing counting pipeline both ways an application can, the source element's
    /// send_event and the pipeline's (which forwards flushes to its sources), and asserts frames
    /// flow again after each flush and nothing posted an error.
    fn assert_flushes_recover(
        pipeline: &gst::Pipeline,
        src: &gst::Element,
        frames: &std::sync::atomic::AtomicUsize,
    ) {
        use gst::prelude::*;
        assert_frames_flow(frames, "start");
        for run in 0..10 {
            let target: &gst::Element = if run % 2 == 0 {
                src
            } else {
                pipeline.upcast_ref()
            };
            assert!(target.send_event(gst::event::FlushStart::new()));
            assert!(target.send_event(gst::event::FlushStop::new(true)));
            assert_frames_flow(frames, &format!("flush {run} via {}", target.name()));
        }
        let bus = pipeline.bus().unwrap();
        assert!(
            bus.pop_filtered(&[gst::MessageType::Error]).is_none(),
            "flushing posted an error"
        );
    }

    // Checks the flush oracle itself on core/base live sources, so a waylanddisplaysrc failure
    // under it is the element's and not the harness's. Only element- and pipeline-level flushes
    // restart a paused push-mode GstBaseSrc task; a bare upstream flush from a downstream pad
    // leaves it paused on any GstBaseSrc (videotestsrc included), so it is no oracle.
    #[test]
    fn flush_oracle_recovers_reference_live_sources() {
        use gst::prelude::*;
        test_init();

        let mut checked = 0;
        for (factory, video_caps) in [("fakesrc", false), ("videotestsrc", true)] {
            let Ok(src) = gst::ElementFactory::make(factory)
                .property("is-live", true)
                .build()
            else {
                eprintln!("skip: no {factory}");
                continue;
            };
            let (pipeline, src, frames) = counting_pipeline_with(src, video_caps);
            pipeline.set_state(gst::State::Playing).unwrap();
            assert_flushes_recover(&pipeline, &src, &frames);
            pipeline.set_state(gst::State::Null).unwrap();
            checked += 1;
        }
        assert!(checked > 0, "no reference live source to validate against");
    }

    /// Asserts frames keep flowing: the count grows within a few seconds.
    fn assert_frames_flow(frames: &std::sync::atomic::AtomicUsize, what: &str) {
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};
        let start = frames.load(Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(5);
        while frames.load(Ordering::SeqCst) < start + 3 {
            assert!(Instant::now() < deadline, "no frames after {what}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    // unlock() on a flush dropped the frame create() was waiting for, the compositor
    // shut down when it could not deliver it, and the stream died.
    #[test]
    fn frames_flow_after_flushes() {
        use gst::prelude::*;
        test_init();

        let (pipeline, src, frames) = counting_pipeline();
        pipeline.set_state(gst::State::Playing).unwrap();
        assert_flushes_recover(&pipeline, &src, &frames);
        // The flushed pipeline still tears down and starts again.
        pipeline.set_state(gst::State::Null).unwrap();
        pipeline.set_state(gst::State::Playing).unwrap();
        assert_frames_flow(&frames, "restart after flushes");
        pipeline.set_state(gst::State::Null).unwrap();
    }

    // The source is not seekable: a flushing seek must be refused without flushing or stalling
    // the stream.
    #[test]
    fn flushing_seek_is_refused_and_frames_keep_flowing() {
        use gst::prelude::*;
        test_init();

        let (pipeline, _src, frames) = counting_pipeline();
        pipeline.set_state(gst::State::Playing).unwrap();
        assert_frames_flow(&frames, "start");
        assert!(
            pipeline
                .seek_simple(gst::SeekFlags::FLUSH, gst::ClockTime::ZERO)
                .is_err()
        );
        assert_frames_flow(&frames, "the refused seek");
        let bus = pipeline.bus().unwrap();
        assert!(
            bus.pop_filtered(&[gst::MessageType::Error]).is_none(),
            "the seek posted an error"
        );
        pipeline.set_state(gst::State::Null).unwrap();
    }

    // a create() cancelled by a live pause's unlock() could return FLUSHING after a
    // quick resume had restarted the streaming task, leaving it stopped for good.
    #[test]
    fn frames_flow_after_quick_pause_resume() {
        use gst::prelude::*;
        use gst::subclass::prelude::*;
        test_init();

        let (pipeline, src, frames) = counting_pipeline();
        let imp = src
            .downcast_ref::<super::super::WaylandDisplaySrc>()
            .unwrap()
            .imp();
        pipeline.set_state(gst::State::Playing).unwrap();
        assert_frames_flow(&frames, "start");
        for run in 0..100 {
            pipeline.set_state(gst::State::Paused).unwrap();
            // Left set, it would stop a later flush or stop from cancelling the frame wait.
            assert!(!imp.pausing.load(std::sync::atomic::Ordering::Acquire));
            pipeline.set_state(gst::State::Playing).unwrap();
            assert_frames_flow(&frames, &format!("resume {run}"));
        }
        pipeline.set_state(gst::State::Null).unwrap();
    }

    // start() must give the display it creates the configured devices; a start() that installed
    // a plain command channel instead passed every other test.
    #[test]
    fn started_display_adds_configured_devices() {
        use gst::prelude::*;
        use std::time::{Duration, Instant};
        test_init();

        let path = "/nonexistent/input/started-display-keyboard";
        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>()
            .property("render-node", "software")
            .property("keyboard", path)
            .build()
            .upcast::<gst::Element>();
        let sink = gst::ElementFactory::make("fakesink").build().unwrap();
        let pipeline = gst::Pipeline::new();
        pipeline.add_many([&src, &sink]).unwrap();
        src.link(&sink).unwrap();
        pipeline.set_state(gst::State::Playing).unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        while !ADDED_INPUT_DEVICES
            .lock()
            .unwrap()
            .iter()
            .any(|added| added == path)
        {
            assert!(
                Instant::now() < deadline,
                "the compositor never added the configured device"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        pipeline.set_state(gst::State::Null).unwrap();
    }

    /// A playing pipeline whose source waits on frames its compositor sends only when told to,
    /// like one that wedges mid-stream.
    struct Wedged {
        src: super::super::WaylandDisplaySrc,
        pipeline: gst::Pipeline,
        sink: gst::Element,
        /// Buffers the sink received.
        frames: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        /// Frames create() asked the compositor for.
        requests: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        /// Each send answers the oldest owed frame with an empty buffer.
        deliver: std::sync::mpsc::Sender<()>,
        /// Join after stopping the pipeline.
        wedge: std::thread::JoinHandle<(
            waylanddisplaycore::Channel<waylanddisplaycore::Command>,
            Vec<waylanddisplaycore::Command>,
        )>,
    }

    fn wedged_pipeline() -> Wedged {
        use gst::prelude::*;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;
        use waylanddisplaycore::{Command, FormatSet};

        let src = gst::glib::Object::builder::<super::super::WaylandDisplaySrc>().build();
        // start() keeps a display that is already there, so the pipeline runs on this one.
        let (unserviced, compositor_tx) = unserviced_display(&src);
        // Answers caps queries, but frames only when told to.
        let (frame_requested_tx, frame_requested) = std::sync::mpsc::channel();
        let (deliver, deliver_rx) = std::sync::mpsc::channel::<()>();
        let requests = Arc::new(AtomicUsize::new(0));
        let requested = requests.clone();
        let wedge = std::thread::spawn(move || {
            let mut owed_frames = Vec::new();
            loop {
                match unserviced.try_recv() {
                    Ok(Command::GetSupportedDmaFormats(reply)) => {
                        let _ = reply.send(FormatSet::default());
                    }
                    Ok(command @ Command::Buffer(..)) => {
                        owed_frames.push(command);
                        requested.fetch_add(1, Ordering::SeqCst);
                        let _ = frame_requested_tx.send(());
                    }
                    // stop() is dropping the display: let the real compositor quit so Drop can
                    // join it.
                    Ok(Command::Quit) => {
                        compositor_tx.send(Command::Quit).unwrap();
                        return (unserviced, owed_frames);
                    }
                    Ok(_) => {}
                    Err(_) => {
                        if !owed_frames.is_empty() && deliver_rx.try_recv().is_ok() {
                            if let Command::Buffer(reply, _) = owed_frames.remove(0) {
                                let _ = reply.send(Ok(gst::Buffer::new()));
                            }
                        }
                        std::thread::sleep(Duration::from_millis(5))
                    }
                }
            }
        });

        let (pipeline, _, frames) = counting_pipeline_with(src.clone().upcast(), true);
        let sink = pipeline.iterate_sinks().next().unwrap().unwrap();
        pipeline.set_state(gst::State::Playing).unwrap();
        frame_requested
            .recv_timeout(Duration::from_secs(5))
            .expect("create() never requested a frame");
        Wedged {
            src,
            pipeline,
            sink,
            frames,
            requests,
            deliver,
            wedge,
        }
    }

    // On a real pipeline the base class handles a pad flush only once the pause releases its live
    // lock, so event() must cancel the frame wait before chaining up, not after.
    #[test]
    fn pad_flush_cancels_frame_wait_before_pause_finishes() {
        use gst::prelude::*;
        use gst::subclass::prelude::*;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};
        test_init();

        let Wedged {
            src,
            pipeline,
            sink,
            wedge,
            ..
        } = wedged_pipeline();
        // Hold PLAYING->PAUSED inside GstBaseSrc's live-lock section, after its unlock().
        let hold = Duration::from_secs(2);
        let held = Arc::new(AtomicBool::new(false));
        let src_addr = src.as_ptr() as usize;
        let holding = held.clone();
        // Process-wide: undone on drop, panic included, so later tests don't print basesrc
        // debug output or run a hook keyed on a freed element.
        struct HoldGuard(
            gst::DebugCategory,
            gst::DebugLevel,
            Option<gst::log::DebugLogFunction>,
        );
        impl Drop for HoldGuard {
            fn drop(&mut self) {
                if let Some(hook) = self.2.take() {
                    gst::log::remove_log_function(hook);
                }
                self.0.set_threshold(self.1);
            }
        }
        let basesrc = gst::DebugCategory::get("basesrc").unwrap();
        let mut guard = HoldGuard(basesrc, basesrc.threshold(), None);
        basesrc.set_threshold(gst::DebugLevel::Debug);
        guard.2 = Some(gst::log::add_log_function(
            move |_, _, _, _, _, object, message| {
                if object.map(|o| o.as_ptr() as usize) == Some(src_addr)
                    && message
                        .get()
                        .is_some_and(|m| m.starts_with("live running 0"))
                    && !holding.swap(true, Ordering::SeqCst)
                {
                    std::thread::sleep(hold);
                }
            },
        ));

        let pausing = {
            let pipeline = pipeline.clone();
            std::thread::spawn(move || pipeline.set_state(gst::State::Paused).unwrap())
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while !held.load(Ordering::SeqCst) {
            assert!(
                Instant::now() < deadline,
                "the pause never reached the hold"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let flushed = Instant::now();
        let flushing = {
            let sinkpad = sink.static_pad("sink").unwrap();
            std::thread::spawn(move || sinkpad.push_event(gst::event::FlushStart::new()))
        };
        // create() stashes the frame it was waiting on once it is cancelled.
        while src.imp().pending_frame.lock().unwrap().is_none() {
            assert!(
                Instant::now() < flushed + hold * 3,
                "the flush never cancelled create()"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let cancel_took = flushed.elapsed();

        pausing.join().unwrap();
        assert!(flushing.join().unwrap());
        drop(guard);
        assert!(
            sink.static_pad("sink")
                .unwrap()
                .push_event(gst::event::FlushStop::new(true))
        );
        pipeline.set_state(gst::State::Null).unwrap();
        let (unserviced, owed_frames) = wedge.join().unwrap();
        drop(owed_frames);
        drop(unserviced);
        assert!(
            cancel_took < hold / 2,
            "the flush waited {cancel_took:?} for the pause to finish"
        );
    }

    // A pipeline stop must unlock() a create() whose frame the compositor never sends, going
    // through change_state() like a real stop, not only when unlock() is called directly.
    #[test]
    fn stop_does_not_wait_for_an_unanswered_frame() {
        use gst::prelude::*;
        use std::time::{Duration, Instant};
        test_init();

        let Wedged {
            src: _src,
            pipeline,
            wedge,
            ..
        } = wedged_pipeline();

        let stopping = Instant::now();
        pipeline.set_state(gst::State::Null).unwrap();
        let stop_took = stopping.elapsed();

        let (unserviced, owed_frames) = wedge.join().unwrap();
        drop(owed_frames);
        drop(unserviced);
        assert!(
            stop_took < Duration::from_secs(5),
            "stop waited {stop_took:?} behind an unanswered frame"
        );
    }

    /// Waits up to 5 s for `done`.
    fn wait_for(what: &str, done: impl Fn() -> bool) {
        use std::time::{Duration, Instant};
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    // On a real pipeline: a flush cancels create()'s frame wait and keeps that frame, the task the
    // flush-stop restarts waits on the kept frame rather than requesting another, the frame reaches
    // the sink once the compositor delivers it, and streaming then carries on.
    #[test]
    fn flush_keeps_the_cancelled_frame_and_delivers_it_after() {
        use gst::prelude::*;
        use gst::subclass::prelude::*;
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        test_init();

        let w = wedged_pipeline();
        assert!(w.src.send_event(gst::event::FlushStart::new()));
        wait_for("create() to keep its cancelled frame", || {
            w.src.imp().pending_frame.lock().unwrap().is_some()
        });
        assert!(w.src.send_event(gst::event::FlushStop::new(true)));
        wait_for("the restarted task to take the kept frame", || {
            w.src.imp().pending_frame.lock().unwrap().is_none()
        });
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(
            w.requests.load(Ordering::SeqCst),
            1,
            "requested another frame instead of waiting on the kept one"
        );
        assert_eq!(w.frames.load(Ordering::SeqCst), 0);

        w.deliver.send(()).unwrap();
        wait_for("the kept frame to reach the sink", || {
            w.frames.load(Ordering::SeqCst) == 1
        });
        wait_for("the next frame request", || {
            w.requests.load(Ordering::SeqCst) == 2
        });
        w.deliver.send(()).unwrap();
        wait_for("the next frame to reach the sink", || {
            w.frames.load(Ordering::SeqCst) == 2
        });

        let bus = w.pipeline.bus().unwrap();
        assert!(
            bus.pop_filtered(&[gst::MessageType::Error]).is_none(),
            "the flush posted an error"
        );
        w.pipeline.set_state(gst::State::Null).unwrap();
        let (unserviced, owed_frames) = w.wedge.join().unwrap();
        drop(owed_frames);
        drop(unserviced);
    }

    // A pipeline stopped right after a flush cancelled create(), before any flush-stop, must tear
    // down promptly, drop the kept frame with the display, and start cleanly again.
    #[test]
    fn teardown_right_after_a_cancelled_frame_then_restart() {
        use gst::prelude::*;
        use gst::subclass::prelude::*;
        use std::time::{Duration, Instant};
        test_init();

        let w = wedged_pipeline();
        assert!(w.src.send_event(gst::event::FlushStart::new()));
        wait_for("create() to keep its cancelled frame", || {
            w.src.imp().pending_frame.lock().unwrap().is_some()
        });

        let stopping = Instant::now();
        w.pipeline.set_state(gst::State::Null).unwrap();
        let stop_took = stopping.elapsed();
        let (unserviced, owed_frames) = w.wedge.join().unwrap();
        drop(owed_frames);
        drop(unserviced);
        assert!(
            stop_took < Duration::from_secs(5),
            "stop took {stop_took:?} after a cancel"
        );
        assert!(w.src.imp().pending_frame.lock().unwrap().is_none());
        assert!(w.src.imp().state.lock().unwrap().is_none());

        // The next run gets a real compositor and streams.
        w.src.set_property("render-node", "software");
        w.pipeline.set_state(gst::State::Playing).unwrap();
        assert_frames_flow(&w.frames, "restart after a cancelled teardown");
        w.pipeline.set_state(gst::State::Null).unwrap();
    }
}
