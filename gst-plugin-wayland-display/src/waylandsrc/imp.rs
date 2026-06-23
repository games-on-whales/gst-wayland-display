use crate::utils::{CAT, GstLayer};
use gst::message::Application;
use gst::query::Allocation;
use gst::subclass::prelude::*;
use gst::{Context, Event, Fraction, glib};
use gst::{LibraryError, LoggableError};
use gst::{Structure, prelude::*};
use gst_base::prelude::BaseSrcExt;
use gst_base::subclass::base_src::CreateSuccess;
use gst_base::subclass::prelude::*;
use gst_video::{NavigationEvent, VideoCapsBuilder, VideoFormat, VideoInfo, VideoInfoDmaDrm};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ops::DerefMut;
use std::sync::atomic::AtomicPtr;
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
use waylanddisplaycore::{
    ButtonState, Channel, Command, DrmFormat, DrmModifier, GstVideoInfo, KeyState, Sender,
    WaylandDisplay, channel, utils::device::PCIVendor,
};

pub struct WaylandDisplaySrc {
    state: Mutex<Option<State>>,
    settings: Mutex<Settings>,
    command_tx: Sender<Command>,
    command_rx: Mutex<Option<Channel<Command>>>,
}

impl Default for WaylandDisplaySrc {
    fn default() -> Self {
        let (command_tx, command_rx) = channel();
        WaylandDisplaySrc {
            state: Mutex::new(None),
            settings: Mutex::new(Settings::default()),
            command_tx,
            command_rx: Mutex::new(Some(command_rx)),
        }
    }
}

#[derive(Debug, Default)]
pub struct Settings {
    render_node: Option<String>,
    input_devices: Vec<String>,
    disable_intel_workaround: bool,
    nv12: bool,
    #[cfg(feature = "cuda")]
    cuda_context: Option<Arc<Mutex<cuda::CUDAContext>>>,
    #[cfg(feature = "cuda")]
    cuda_raw_ptr: AtomicPtr<cuda::GstCudaContext>,
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
                    let _ = self.command_tx.send(Command::InputDevice(path));
                    return true;
                } else if structure.has_name("MouseMoveAbsolute") {
                    let x = structure
                        .get::<f64>("pointer_x")
                        .expect("Should contain pointer_x");
                    let y = structure
                        .get::<f64>("pointer_y")
                        .expect("Should contain pointer_y");

                    let _ = self
                        .command_tx
                        .send(Command::PointerMotionAbsolute((x, y).into()));

                    return true;
                } else if structure.has_name("MouseMoveRelative") {
                    let x = structure
                        .get::<f64>("pointer_x")
                        .expect("Should contain pointer_x");
                    let y = structure
                        .get::<f64>("pointer_y")
                        .expect("Should contain pointer_y");

                    let _ = self.command_tx.send(Command::PointerMotion((x, y).into()));

                    return true;
                } else if structure.has_name("MouseButton") {
                    let button = structure
                        .get::<u32>("button")
                        .expect("Should contain button");
                    let pressed = structure
                        .get::<bool>("pressed")
                        .expect("Should contain pressed");

                    let _ = self.command_tx.send(Command::PointerButton(
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

                    let _ = self.command_tx.send(Command::PointerAxis(x, y));

                    return true;
                } else if structure.has_name("KeyboardKey") {
                    let key = structure.get::<u32>("key").expect("Should contain key");
                    let pressed = structure
                        .get::<bool>("pressed")
                        .expect("Should contain pressed");

                    let _ = self.command_tx.send(Command::KeyboardInput(
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
                    let _ = self.command_tx.send(Command::TouchDown(id, (x, y).into()));
                    return true;
                } else if structure.has_name("TouchUp") {
                    let id = structure.get::<u32>("id").expect("Should contain id");
                    let _ = self.command_tx.send(Command::TouchUp(id));
                    return true;
                } else if structure.has_name("TouchMotion") {
                    let x = structure.get::<f64>("x").expect("Should contain x");
                    let y = structure.get::<f64>("y").expect("Should contain y");
                    let id = structure.get::<u32>("id").expect("Should contain id");
                    let _ = self
                        .command_tx
                        .send(Command::TouchMotion(id, (x, y).into()));
                    return true;
                } else if structure.has_name("TouchFrame") {
                    let _ = self.command_tx.send(Command::TouchFrame);
                    return true;
                } else if structure.has_name("TouchCancel") {
                    let _ = self.command_tx.send(Command::TouchCancel);
                    return true;
                }
            }
            gst::EventView::Navigation(n) => {
                let navigation_event = gst_video::NavigationEvent::parse(n).unwrap();

                match navigation_event {
                    NavigationEvent::MouseMove { x, y, .. } => {
                        let _ = self
                            .command_tx
                            .send(Command::PointerMotionAbsolute((x, y).into()));

                        return true;
                    }
                    NavigationEvent::MouseButtonPress { button, .. } => {
                        if let Some(cmd) = gst_button_to_msg(button, ButtonState::Pressed) {
                            let _ = self.command_tx.send(cmd);
                        } else {
                            tracing::warn!("Unknown mouse button pressed: {:?}", button);
                        }

                        return true;
                    }
                    NavigationEvent::MouseButtonRelease { button, .. } => {
                        if let Some(cmd) = gst_button_to_msg(button, ButtonState::Released) {
                            let _ = self.command_tx.send(cmd);
                        } else {
                            tracing::warn!("Unknown mouse button released: {:?}", button);
                        }

                        return true;
                    }
                    NavigationEvent::KeyPress { key, .. } => {
                        if let Some(scancode) = gst_key_to_scancode(&key) {
                            let _ = self
                                .command_tx
                                .send(Command::KeyboardInput(scancode, KeyState::Pressed));
                        } else {
                            tracing::warn!("Unknown keyboard key pressed: {:?}", key);
                        }

                        return true;
                    }
                    NavigationEvent::KeyRelease { key, .. } => {
                        if let Some(scancode) = gst_key_to_scancode(&key) {
                            let _ = self
                                .command_tx
                                .send(Command::KeyboardInput(scancode, KeyState::Released));
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
            ]
        });

        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
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
            "nv12" => {
                let mut settings = self.settings.lock().unwrap();
                settings.nv12 = value.get::<bool>().expect("Type checked upstream");
            }
            _ => unreachable!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
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
        let res = self.parent_change_state(transition);
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
            waylanddisplaycore::utils::va_share::handle_set_context(
                elem_ptr,
                ctx_ptr,
                &render_path,
            );
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
    fn check_nv12_export(
        &self,
        caps: &gst::Caps,
        info: &VideoInfoDmaDrm,
    ) -> Result<(), gst::LoggableError> {
        let is_nv12 = caps
            .structure(0)
            .and_then(|s| s.get::<String>("drm-format").ok())
            .is_some_and(|f| f.starts_with("NV12"));
        if !is_nv12 {
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
        let exportable = waylanddisplaycore::utils::vulkan_nv12::supported_nv12_modifiers(minor);
        let encoder_pref = waylanddisplaycore::utils::va_query::import_nv12_modifier(&render_path);
        tracing::info!(
            "waylandsrc: NV12 export modifier {modifier:#x} on {render_path} \
             (encoder imports {encoder_pref:#x?}, exportable {exportable:#x?})"
        );
        if !exportable.contains(&modifier) {
            return Err(gst::loggable_error!(
                CAT,
                "negotiated NV12 modifier {modifier:#x} is not Vulkan-exportable on \
                 {render_path} (exportable: {exportable:#x?})"
            ));
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

impl BaseSrcImpl for WaylandDisplaySrc {
    #[cfg(feature = "cuda")]
    fn query(&self, query: &mut gst::QueryRef) -> bool {
        if query.type_() == gst::QueryType::Context {
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
                let dma_formats = state.display.get_supported_dma_formats();

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

        // Let's get the pool from the query, if it's not there, we'll create one
        let pools = query.allocation_pools();
        let (pool, update_pool, size, min, max) = if pools.is_empty() {
            tracing::info!("No allocation pools, creating one");
            let video_info = VideoInfo::from_caps(outcaps.unwrap())?;
            let size = video_info.size() as u32;
            (CUDABufferPool::new(&cuda_ctx), false, size, 0, 0)
        } else {
            tracing::info!("Found existing allocation pools");
            let (pool, size, min, max) = pools.get(0).unwrap();
            let wrapped_pool = match pool {
                Some(pool) => match CUDABufferPool::from(pool.as_ptr()) {
                    Ok(pool) => Ok(pool),
                    Err(err) => {
                        tracing::info!(
                            "Failed to get CUDA buffer pool from allocation pool: {}",
                            err
                        );
                        unsafe { gst::ffi::gst_clear_object(pool.as_ptr() as *mut _) };
                        CUDABufferPool::new(&cuda_ctx)
                    }
                },
                None => {
                    tracing::info!("Failed to get CUDA buffer pool from allocation pool");
                    CUDABufferPool::new(&cuda_ctx)
                }
            };
            (wrapped_pool, true, *size, *min, *max)
        };

        match pool {
            Ok(pool) => {
                let caps = unsafe { gst::Caps::from_glib_full(outcaps.unwrap().as_ptr()) };
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
                let _ = self
                    .command_tx
                    .send(Command::UpdateCUDABufferPool(Arc::new(Mutex::new(Some(
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
        let video_info = match VideoInfoDmaDrm::from_caps(caps) {
            Ok(dma_video_info) => {
                self.check_nv12_export(caps, &dma_video_info)?;
                GstVideoInfo::DMA(dma_video_info)
            }
            #[cfg(feature = "cuda")]
            Err(_) => {
                let base_video_info =
                    gst_video::VideoInfo::from_caps(caps).expect("failed to get video info");
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
                    let dma_formats = state.as_ref().unwrap().display.get_supported_dma_formats();
                    let chosen_format =
                        gst_video_format_name_to_drm_fourcc(base_video_info.format().to_string())
                            .expect("failed to get drm format");
                    let format = dma_formats
                        .iter()
                        .filter(|dma_format| dma_format.code == chosen_format)
                        .next()
                        .expect("failed to find a matching DRM format for the CUDA format");
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
            Err(_) => {
                GstVideoInfo::RAW(VideoInfo::from_caps(caps).expect("failed to get video info"))
            }
        };

        let _ = self.command_tx.send(Command::VideoInfo(video_info));

        self.parent_set_caps(caps)
    }

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let mut state = self.state.lock().unwrap();
        if state.is_some() {
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        let (render_node, input_devices, have_cuda_context) = {
            let settings = self.settings.lock().unwrap();
            (
                settings.render_node.clone(),
                settings.input_devices.clone(),
                settings.cuda_context.is_some(),
            )
        };

        #[cfg(not(feature = "cuda"))]
        let (render_node, input_devices) = {
            let settings = self.settings.lock().unwrap();
            (settings.render_node.clone(), settings.input_devices.clone())
        };

        let elem = self.obj().upcast_ref::<gst::Element>().to_owned();
        let subscriber = Registry::default().with(GstLayer);

        let Ok(mut display) = tracing::subscriber::with_default(subscriber, || {
            let mut command_rx = self.command_rx.lock().unwrap();
            WaylandDisplay::new_with_channel(
                render_node.clone(),
                self.command_tx.clone(),
                command_rx.deref_mut().take().unwrap(),
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
            waylanddisplaycore::utils::va_share::ensure_shared_display(elem_ptr, &va_node);
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

        for path in input_devices {
            display.add_input_device(path);
        }

        let mut structure = Structure::builder("wayland.src");
        for (key, var) in display.env_vars().flat_map(|var| var.split_once("=")) {
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
        Ok(())
    }

    fn is_seekable(&self) -> bool {
        false
    }
}

impl PushSrcImpl for WaylandDisplaySrc {
    fn create(
        &self,
        _buffer: Option<&mut gst::BufferRef>,
    ) -> Result<CreateSuccess, gst::FlowError> {
        let mut state_guard = self.state.lock().unwrap();
        let Some(state) = state_guard.as_mut() else {
            return Err(gst::FlowError::Eos);
        };

        let subscriber = Registry::default().with(GstLayer);
        tracing::subscriber::with_default(subscriber, || {
            state.display.frame().map(CreateSuccess::NewBuffer)
        })
    }
}

/// A `/dev/dri/*` render node backs a real `GstVaDisplay`; the `software` (llvmpipe)
/// target and any other value do not, so VA-display sharing is skipped for them.
fn render_node_backs_va_display(path: &str) -> bool {
    path.starts_with("/dev/dri/")
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

    fn test_init() -> () {
        INIT.call_once(|| {
            tracing_subscriber::fmt::try_init().ok();
            gst::init().expect("Failed to initialize GStreamer");
        });
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

    /// Run a gst-launch description to EOS, failing on any bus ERROR. Reusable
    /// integration harness: a negotiation, import, or encode failure surfaces as a
    /// bus ERROR, so treating ERROR as fatal makes any pipeline a real regression
    /// guard. Reaching EOS means every `num-buffers` frame negotiated and encoded.
    fn run_pipeline_to_eos(desc: &str) {
        use gst::prelude::*;
        // Make `waylanddisplaysrc` resolvable by parse::launch in the test process.
        crate::plugin_register_static().ok();
        let pipeline = gst::parse::launch_full(desc, None, gst::ParseFlags::empty())
            .expect("parse pipeline")
            .downcast::<gst::Pipeline>()
            .expect("not a pipeline");
        pipeline
            .set_state(gst::State::Playing)
            .expect("set state Playing");
        let bus = pipeline.bus().expect("pipeline bus");
        let mut saw_eos = false;
        for msg in bus.iter_timed(gst::ClockTime::from_seconds(30)) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    saw_eos = true;
                    break;
                }
                gst::MessageView::Error(err) => {
                    let _ = pipeline.set_state(gst::State::Null);
                    panic!(
                        "pipeline error from {:?}: {} ({:?})",
                        err.src().map(|s| s.path_string()),
                        err.error(),
                        err.debug()
                    );
                }
                _ => {}
            }
        }
        pipeline
            .set_state(gst::State::Null)
            .expect("set state Null");
        assert!(saw_eos, "pipeline timed out before EOS (no frames encoded)");
    }

    /// Tier-1 Vulkan encode path: the compositor's RGBA dmabuf is imported into
    /// Vulkan, color-converted to NV12, and encoded by the Vulkan video encoder.
    /// The NV12 surface stays inside Vulkan -- no cross-API dmabuf round-trip and no
    /// modifier negotiation, since the Vulkan driver owns the encode-source layout.
    /// `#[ignore]` + env-gated; run locally once a gst build provides `vulkanh265enc`
    /// (Debian's gst-plugins-bad does not yet build the Vulkan video encoders):
    ///   VULKAN_ENC_NODE=/dev/dri/renderDNNN \
    ///     cargo test -p gst-plugin-wayland-display -- --ignored test_vulkan_encode_pipeline
    #[test]
    #[ignore = "needs a GPU + Vulkan video encoder (vulkanh265enc); set VULKAN_ENC_NODE and run with --ignored"]
    fn test_vulkan_encode_pipeline() {
        test_init();
        let Ok(node) = std::env::var("VULKAN_ENC_NODE") else {
            eprintln!("skip: set VULKAN_ENC_NODE=/dev/dri/renderDNNN to run this");
            return;
        };
        // gst's Vulkan video encoders aren't built in many distros yet; skip cleanly.
        let Some(enc) = ["vulkanh265enc", "vulkanh264enc"]
            .into_iter()
            .find(|e| gst::ElementFactory::find(e).is_some())
        else {
            eprintln!(
                "skip: no Vulkan video encoder (vulkanh265enc/vulkanh264enc) in this gst build"
            );
            return;
        };
        let parse = if enc.contains("265") {
            "h265parse"
        } else {
            "h264parse"
        };
        run_pipeline_to_eos(&format!(
            "waylanddisplaysrc render-node={node} num-buffers=30 \
             ! vulkanupload ! vulkancolorconvert ! {enc} ! {parse} ! fakesink sync=false"
        ));
    }
}
