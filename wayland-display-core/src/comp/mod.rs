use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    os::fd::AsFd,
    sync::{mpsc::Sender, Arc, Mutex, Weak},
    time::{Duration, Instant},
};
use std::os::fd::AsRawFd;

use super::Command;
use gst_video::VideoInfo;
use once_cell::sync::Lazy;
use smithay::{
    backend::{
        allocator::dmabuf::Dmabuf,
        drm::{DrmNode, NodeType},
        egl::{EGLContext, EGLDevice, EGLDisplay},
        libinput::LibinputInputBackend,
        renderer::{
            element::memory::MemoryRenderBuffer,
            damage::{OutputDamageTracker, Error as DTRError},
            gles::{GlesRenderbuffer, GlesRenderer},
            Bind, Offscreen,
        },
    },
    desktop::{
        utils::{
            send_frames_surface_tree, surface_presentation_feedback_flags_from_states,
            surface_primary_scanout_output, update_surface_primary_scanout_output,
            OutputPresentationFeedback,
        },
        PopupManager, Space, Window,
    },
    input::{keyboard::XkbConfig, pointer::CursorImageStatus, Seat, SeatState},
    output::{Mode as OutputMode, Output, PhysicalProperties, Subpixel},
    reexports::{
        calloop::{
            channel::{Channel, Event},
            generic::Generic,
            timer::{TimeoutAction, Timer},
            EventLoop, Interest, LoopHandle, Mode, PostAction,
        },
        input::Libinput,
        wayland_protocols::wp::presentation_time::server::wp_presentation_feedback,
        wayland_server::{
            backend::{ClientData, ClientId, DisconnectReason},
            Display, DisplayHandle,
        },
    },
    utils::{Clock, Logical, Monotonic, Physical, Point, Rectangle, Size, Transform},
    wayland::{
        compositor::{with_states, CompositorState, CompositorClientState},
        dmabuf::{DmabufGlobal, DmabufState},
        output::OutputManagerState,
        presentation::PresentationState,
        shell::xdg::{XdgShellState, XdgToplevelSurfaceData},
        shm::ShmState,
        socket::ListeningSocketSource,
        viewporter::ViewporterState,
        relative_pointer::RelativePointerManagerState,
    },
};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::memory::MemoryBuffer;
use smithay::reexports::drm::buffer::DrmFourcc;
use smithay::reexports::wayland_server::backend::GlobalId;
use smithay::reexports::wayland_server::Client;
use smithay::wayland::pointer_constraints::PointerConstraintsState;
use smithay::wayland::selection::data_device::DataDeviceState;
use smithay::wayland::shell::xdg::SurfaceCachedState;
use tracing::debug;
use tracing::field::debug;

mod focus;
mod input;
mod rendering;

pub use self::focus::*;
pub use self::input::*;
pub use self::rendering::*;
use crate::{utils::RenderTarget, wayland::protocols::wl_drm::create_drm_global};

static EGL_DISPLAYS: Lazy<Mutex<HashMap<Option<DrmNode>, Weak<EGLDisplay>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Default)]
pub struct ClientState {
    pub compositor_state: CompositorClientState,
}

impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {}
    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
}

#[allow(dead_code)]
pub(crate) struct State {
    handle: LoopHandle<'static, State>,
    should_quit: bool,
    clock: Clock<Monotonic>,

    // render
    dtr: Option<OutputDamageTracker>,
    renderbuffer: Option<GlesRenderbuffer>,
    pub renderer: GlesRenderer,
    egl_display_ref: Arc<EGLDisplay>,
    dmabuf_global: Option<(DmabufGlobal, GlobalId)>,
    last_render: Option<Instant>,

    // management
    pub output: Option<Output>,
    pub video_info: Option<VideoInfo>,
    pub seat: Seat<Self>,
    pub space: Space<Window>,
    pub popups: PopupManager,
    pointer_location: Point<f64, Logical>,
    last_pointer_movement: Instant,
    cursor_element: MemoryRenderBuffer,
    pub cursor_state: CursorImageStatus,
    surpressed_keys: HashSet<u32>,
    pub pending_windows: Vec<Window>,
    input_context: Libinput,

    // wayland state
    pub dh: DisplayHandle,
    pub compositor_state: CompositorState,
    pub data_device_state: DataDeviceState,
    pub dmabuf_state: DmabufState,
    output_state: OutputManagerState,
    presentation_state: PresentationState,
    relative_ptr_state: RelativePointerManagerState,
    pointer_constraints_state: PointerConstraintsState,
    pub seat_state: SeatState<Self>,
    pub shell_state: XdgShellState,
    pub shm_state: ShmState,
    viewporter_state: ViewporterState,
}

pub fn get_egl_device_for_node(drm_node: &DrmNode) -> EGLDevice {
    let drm_node = drm_node
        .node_with_type(NodeType::Render)
        .and_then(Result::ok)
        .unwrap_or(drm_node.clone());
    EGLDevice::enumerate()
        .expect("Failed to enumerate EGLDevices")
        .find(|d| d.try_get_render_node().unwrap_or_default() == Some(drm_node))
        .expect("Unable to find EGLDevice for drm-node")
}

pub(crate) fn init(
    command_src: Channel<Command>,
    render: impl Into<RenderTarget>,
    devices_tx: Sender<Vec<CString>>,
    envs_tx: Sender<Vec<CString>>,
) {
    let clock = Clock::new();
    let mut display = Display::<State>::new().unwrap();
    let dh = display.handle();

    // init state
    let compositor_state = CompositorState::new::<State>(&dh);
    let data_device_state = DataDeviceState::new::<State>(&dh);
    let mut dmabuf_state = DmabufState::new();
    let output_state = OutputManagerState::new_with_xdg_output::<State>(&dh);
    let presentation_state = PresentationState::new::<State>(&dh, clock.id() as _);
    let relative_ptr_state = RelativePointerManagerState::new::<State>(&dh);
    let pointer_constraints_state = PointerConstraintsState::new::<State>(&dh);
    let mut seat_state = SeatState::new();
    let shell_state = XdgShellState::new::<State>(&dh);
    let viewporter_state = ViewporterState::new::<State>(&dh);

    let render_target = render.into();
    let render_node: Option<DrmNode> = render_target.clone().into();

    // init render backend
    let (egl_display_ref, context) = {
        let mut displays = EGL_DISPLAYS.lock().unwrap();
        let maybe_display = displays
            .get(&render_node)
            .and_then(|weak_display| weak_display.upgrade());

        let egl = match maybe_display {
            Some(display) => display,
            None => {
                let device = match render_node.as_ref() {
                    Some(render_node) => get_egl_device_for_node(render_node),
                    None => EGLDevice::enumerate()
                        .expect("Failed to enumerate EGLDevices")
                        .find(|device| {
                            device
                                .extensions()
                                .iter()
                                .any(|e| e == "EGL_MESA_device_software")
                        })
                        .expect("Failed to find software device"),
                };
                let egl = unsafe { EGLDisplay::new(device).expect("Failed to create EGLDisplay") };
                let display = Arc::new(egl);
                displays.insert(render_node, Arc::downgrade(&display));
                display
            }
        };
        let context = EGLContext::new(&egl).expect("Failed to initialize EGL context");
        (egl, context)
    };
    let renderer = unsafe { GlesRenderer::new(context) }.expect("Failed to initialize renderer");
    let _ = devices_tx.send(render_target.as_devices());

    let shm_state = ShmState::new::<State>(&dh, vec![]);
    let dmabuf_global = if let RenderTarget::Hardware(node) = render_target {
        let formats = Bind::<Dmabuf>::supported_formats(&renderer)
            .expect("Failed to query formats")
            .into_iter()
            .collect::<Vec<_>>();

        // dma buffer
        let dmabuf_global = dmabuf_state.create_global::<State>(&dh, formats.clone());
        // wl_drm (mesa protocol, so we don't need EGL_WL_bind_display)
        let wl_drm_global = create_drm_global::<State>(
            &dh,
            node.dev_path().expect("Failed to determine DrmNode path?"),
            formats.clone(),
            &dmabuf_global,
        );

        Some((dmabuf_global, wl_drm_global))
    } else {
        None
    };

    let cursor_element =
        MemoryRenderBuffer::from_memory(MemoryBuffer::from_slice(
            CURSOR_DATA_BYTES,
            Fourcc::Abgr8888,
            (64, 64)), 1, Transform::Normal, None);

    // init input backend
    let libinput_context = Libinput::new_from_path(NixInterface);
    let input_context = libinput_context.clone();
    let libinput_backend = LibinputInputBackend::new(libinput_context);

    let space = Space::default();

    let mut seat = seat_state.new_wl_seat(&dh, "seat-0");
    seat.add_keyboard(XkbConfig::default(), 200, 25)
        .expect("Failed to add keyboard to seat");
    seat.add_pointer();

    let mut event_loop =
        EventLoop::<State>::try_new().expect("Unable to create event_loop");

    let mut state = State {
        handle: event_loop.handle(),
        should_quit: false,
        clock,

        renderer,
        egl_display_ref,
        dtr: None,
        renderbuffer: None,
        dmabuf_global,
        video_info: None,
        last_render: None,

        space,
        popups: PopupManager::default(),
        seat,
        output: None,
        pointer_location: (0., 0.).into(),
        last_pointer_movement: Instant::now(),
        cursor_element,
        cursor_state: CursorImageStatus::default_named(),
        surpressed_keys: HashSet::new(),
        pending_windows: Vec::new(),
        input_context,

        dh: display.handle(),
        compositor_state,
        data_device_state,
        dmabuf_state,
        output_state,
        presentation_state,
        relative_ptr_state,
        pointer_constraints_state,
        seat_state,
        shell_state,
        shm_state,
        viewporter_state,
    };

    // init event loop
    event_loop
        .handle()
        .insert_source(libinput_backend, move |event, _, state| {
            state.process_input_event(event)
        })
        .unwrap();

    event_loop
        .handle()
        .insert_source(command_src, move |event, _, state| {
            match event {
                Event::Msg(Command::VideoInfo(info)) => {
                    debug!("Requested video format: {} .to_fourcc() = {}", info.format(), info.format().to_fourcc());
                    let size: Size<i32, Physical> =
                        (info.width() as i32, info.height() as i32).into();
                    let framerate = info.fps();
                    let duration = Duration::from_secs_f64(
                        framerate.numer() as f64 / framerate.denom() as f64,
                    );

                    // init wayland objects
                    let output = state.output.get_or_insert_with(|| {
                        let output = Output::new(
                            "HEADLESS-1".into(),
                            PhysicalProperties {
                                make: "Virtual".into(),
                                model: "Wolf".into(),
                                size: (0, 0).into(),
                                subpixel: Subpixel::Unknown,
                            },
                        );
                        output.create_global::<State>(&state.dh);
                        output
                    });
                    let mode = OutputMode {
                        size: size.into(),
                        refresh: (duration.as_secs_f64() * 1000.0).round() as i32,
                    };
                    output.change_current_state(Some(mode), None, None, None);
                    output.set_preferred(mode);
                    let dtr = OutputDamageTracker::from_output(&output);

                    state.space.map_output(&output, (0, 0));
                    state.dtr = Some(dtr);
                    state.pointer_location = (size.w as f64 / 2.0, size.h as f64 / 2.0).into();
                    state.renderbuffer = Some(
                        state
                            .renderer
                            .create_buffer(Fourcc::try_from(info.format().to_fourcc()).unwrap_or(Fourcc::Abgr8888),
                                           (info.width() as i32, info.height() as i32).into())
                            .expect("Failed to create renderbuffer"),
                    );
                    state.video_info = Some(info);

                    let new_size = size
                        .to_f64()
                        .to_logical(output.current_scale().fractional_scale())
                        .to_i32_round();
                    for window in state.space.elements() {
                        let toplevel = window.toplevel().unwrap();
                        let max_size = Rectangle::from_loc_and_size(
                            (0, 0),
                            with_states(toplevel.wl_surface(), |states| {
                                states
                                    .data_map
                                    .get::<XdgToplevelSurfaceData>()
                                    .map(|attrs| states.cached_state.current::<SurfaceCachedState>().max_size)
                            })
                                .unwrap_or(new_size),
                        );

                        let new_size = max_size
                            .intersection(Rectangle::from_loc_and_size((0, 0), new_size))
                            .map(|rect| rect.size);
                        toplevel.with_pending_state(|state| state.size = new_size);
                        toplevel.send_configure();
                    }
                }
                Event::Msg(Command::InputDevice(path)) => {
                    tracing::info!(path, "Adding input device.");
                    state.input_context.path_add_device(&path);
                }
                Event::Msg(Command::Buffer(buffer_sender)) => {
                    let wait = if let Some(last_render) = state.last_render {
                        let framerate = state.video_info.as_ref().unwrap().fps();
                        let duration = Duration::from_secs_f64(
                            framerate.denom() as f64 / framerate.numer() as f64,
                        );
                        let time_passed = Instant::now().duration_since(last_render);
                        if time_passed < duration {
                            Some(duration - time_passed)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let render = move |state: &mut State, now: Instant| {
                        if let Err(_) = match state.create_frame() {
                            Ok((buf, render_result)) => {
                                state.last_render = Some(now);
                                render_result.sync.wait(); // we need to wait before giving a hardware buffer to gstreamer or we might not be done writing to it
                                let res = buffer_sender.send(Ok(buf));

                                if let Some(output) = state.output.as_ref() {
                                    let mut output_presentation_feedback =
                                        OutputPresentationFeedback::new(output);
                                    for window in state.space.elements() {
                                        window.with_surfaces(|surface, states| {
                                            update_surface_primary_scanout_output(
                                                surface,
                                                output,
                                                states,
                                                &render_result.states,
                                                |next_output, _, _, _| next_output,
                                            );
                                        });
                                        window.send_frame(
                                            output,
                                            state.clock.now(),
                                            Some(Duration::ZERO),
                                            |_, _| Some(output.clone()),
                                        );
                                        window.take_presentation_feedback(
                                            &mut output_presentation_feedback,
                                            surface_primary_scanout_output,
                                            |surface, _| {
                                                surface_presentation_feedback_flags_from_states(
                                                    surface,
                                                    &render_result.states,
                                                )
                                            },
                                        );
                                    }
                                    if render_result.damage.is_some() {
                                        output_presentation_feedback.presented(
                                            state.clock.now(),
                                            Duration::from_millis(output
                                                .current_mode()
                                                .map(|mode| mode.refresh)
                                                .unwrap_or_default() as u64),
                                            0,
                                            wp_presentation_feedback::Kind::Vsync,
                                        );
                                    }
                                    if let CursorImageStatus::Surface(wl_surface) =
                                        &state.cursor_state
                                    {
                                        send_frames_surface_tree(
                                            wl_surface,
                                            output,
                                            state.clock.now(),
                                            None,
                                            |_, _| Some(output.clone()),
                                        )
                                    }
                                }

                                res
                            }
                            Err(err) => {
                                tracing::error!(?err, "Rendering failed.");
                                buffer_sender.send(Err(match err {
                                    DTRError::OutputNoMode(_) => unreachable!(),
                                    DTRError::Rendering(err) => err.into(),
                                }))
                            }
                        } {
                            state.should_quit = true;
                        }
                    };

                    match wait {
                        Some(duration) => {
                            if let Err(err) = state.handle.insert_source(
                                Timer::from_duration(duration),
                                move |now, _, data| {
                                    render(data, now);
                                    TimeoutAction::Drop
                                },
                            ) {
                                tracing::error!(?err, "Event loop error.");
                                state.should_quit = true;
                            };
                        }
                        None => render(state, Instant::now()),
                    };
                }
                Event::Msg(Command::Quit) | Event::Closed => {
                    state.should_quit = true;
                }
            };
        })
        .unwrap();

    let source = ListeningSocketSource::new_auto().unwrap();
    let socket_name = source.socket_name().to_string_lossy().into_owned();
    tracing::info!(?socket_name, "Listening on wayland socket.");
    event_loop
        .handle()
        .insert_source(source, |client_stream, _, state| {
            if let Err(err) = state
                .dh
                .insert_client(client_stream, Arc::new(ClientState::default()))
            {
                tracing::error!(?err, "Error adding wayland client.");
            };
        })
        .expect("Failed to init wayland socket source");

    event_loop.
        handle()
        .insert_source(
            Generic::new(display, Interest::READ, Mode::Level),
            |_, display, state| {
                // Safety: we don't drop the display
                unsafe {
                    display.get_mut().dispatch_clients(state).unwrap();
                }
                Ok(PostAction::Continue)
            },
        )
        .unwrap();

    let env_vars = vec![CString::new(format!("WAYLAND_DISPLAY={}", socket_name)).unwrap()];
    if let Err(err) = envs_tx.send(env_vars) {
        tracing::warn!(?err, "Failed to post environment to application.");
    }

    let signal = event_loop.get_signal();
    if let Err(err) = event_loop.run(None, &mut state, |state| {
        state.dh
            .flush_clients()
            .expect("Failed to flush clients");
        state.space.refresh();
        state.popups.cleanup();

        if state.should_quit {
            signal.stop();
        }
    }) {
        tracing::error!(?err, "Event loop broke.");
    }
}
