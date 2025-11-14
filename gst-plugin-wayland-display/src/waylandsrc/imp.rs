use std::fmt::Debug;
use std::ops::DerefMut;
use std::sync::Mutex;

use gst::message::Application;
use gst_video::{VideoCapsBuilder, VideoFormat, VideoInfo, VideoInfoDmaDrm};

use crate::utils::{CAT, GstLayer};
use gst::query::Allocation;
use gst::subclass::prelude::*;
use gst::{Context, Event, Fraction, glib};
use gst::{LibraryError, LoggableError};
use gst::{Structure, prelude::*};
use gst_base::prelude::BaseSrcExt;
use gst_base::subclass::base_src::CreateSuccess;
use gst_base::subclass::prelude::*;
use once_cell::sync::Lazy;
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
    #[cfg(feature = "cuda")]
    cuda_context: Option<cuda::CUDAContext>,
    #[cfg(feature = "cuda")]
    cudabuffer_pool: Option<CUDABufferPool>,
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
        if event.type_() == gst::EventType::CustomUpstream {
            let structure = event.structure().expect("Unable to get message structure");
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
            } else if structure.has_name("SetClipboard") {
                let content = structure
                    .get::<String>("content")
                    .expect("Should contain text content");
                let _ = self.command_tx.send(Command::SetClipboard(content));
                return true;
            }
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
                let cuda_context = {
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
                settings.cuda_context = cuda_context;
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

    #[cfg(feature = "cuda")]
    fn set_context(&self, context: &Context) {
        let elem = self.obj().upcast_ref::<gst::Element>().to_owned();
        let cuda_context = {
            let settings = self.settings.lock().unwrap();
            settings.cuda_context.clone()
        };
        match CUDAContext::new_from_set_context(&elem, &context, -1, cuda_context) {
            Ok(ctx) => {
                let mut settings = self.settings.lock().unwrap();
                if settings.cuda_context.is_none() {
                    settings.cuda_context = Some(ctx);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to create CUDA context: {}", e);
            }
        }
        self.parent_set_context(context)
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
                    cuda::gst_cuda_handle_context_query_wrapped(
                        self.obj().as_ref().as_ref(),
                        query,
                        cuda_context,
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
        let mut settings = self.settings.lock().unwrap();
        if settings.cuda_context.is_none() || !is_cuda {
            return self.parent_decide_allocation(query);
        }
        let cuda_ctx = settings.cuda_context.as_ref().unwrap();

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
                pool.configure(&caps, cuda_ctx.stream().unwrap(), size, min, max)
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
                    .send(Command::UpdateCUDABufferPool(pool.clone()));

                settings.cudabuffer_pool = Some(pool);
            }
            Err(err) => {
                tracing::warn!("Failed to create CUDA buffer pool: {}", err);
            }
        }

        self.parent_decide_allocation(query)
    }

    fn set_caps(&self, caps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let video_info = match VideoInfoDmaDrm::from_caps(caps) {
            Ok(dma_video_info) => GstVideoInfo::DMA(dma_video_info),
            #[cfg(feature = "cuda")]
            Err(_) => {
                let base_video_info =
                    gst_video::VideoInfo::from_caps(caps).expect("failed to get video info");
                let is_cuda = caps
                    .features(0)
                    .expect("Failed to get features")
                    .contains(cuda::CAPS_FEATURE_MEMORY_CUDA_MEMORY);
                let settings = self.settings.lock().unwrap();
                if is_cuda && settings.cuda_context.is_some() {
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
                        cuda_context: settings.cuda_context.clone().unwrap(),
                        buffer_pool: settings.cudabuffer_pool.clone(),
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
        let (render_node, input_devices, cuda_context) = {
            let settings = self.settings.lock().unwrap();
            (
                settings.render_node.clone(),
                settings.input_devices.clone(),
                settings.cuda_context.clone(),
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

        #[cfg(feature = "cuda")]
        match display.get_render_device() {
            Some(render_device) => {
                if *render_device.pci_vendor() == PCIVendor::NVIDIA && cuda_context.is_none() {
                    tracing::info!(
                        "Acquiring a CudaContext from the pipeline, you can manually set the `cuda-device-id` property to override this behavior"
                    );
                    let cuda_context = {
                        let settings = self.settings.lock().unwrap();
                        settings.cuda_context.clone()
                    };
                    // TODO: here we should share the raw C pointer between
                    //       CUDAContext::new_from_set_context and CUDAContext::new_from_gstreamer
                    match CUDAContext::new_from_gstreamer(&elem, -1, cuda_context) {
                        Ok(cuda_context) => {
                            let mut settings = self.settings.lock().unwrap();
                            if settings.cuda_context.is_none() {
                                tracing::info!("Acquired a CudaContext via new_from_gstreamer");
                                settings.cuda_context = Some(cuda_context);
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
            tracing::subscriber::with_default(subscriber, || std::mem::drop(state.display));
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
}
