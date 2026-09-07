use super::{Command, DrmFormat, GstVideoInfo};
use gst_video::VideoInfo;
use smithay::backend::SwapBuffersError;
use smithay::backend::allocator::format::FormatSet;
use smithay::backend::input::AxisSource;
use smithay::backend::input::TouchSlot;
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::reexports::gbm::BufferObjectFlags;
use smithay::wayland::dmabuf::DmabufFeedbackBuilder;
use smithay::wayland::presentation::Refresh;
use smithay::wayland::seat::WaylandFocus;
use smithay::wayland::single_pixel_buffer::SinglePixelBufferState;
use smithay::{
    backend::{
        allocator::{Fourcc, dmabuf::Dmabuf},
        drm::{DrmDeviceFd, DrmNode},
        libinput::LibinputInputBackend,
        renderer::{
            Bind,
            damage::{Error as DTRError, OutputDamageTracker},
            element::memory::{MemoryBuffer, MemoryRenderBuffer},
        },
    },
    desktop::{
        PopupManager, Space, Window,
        utils::{
            OutputPresentationFeedback, send_frames_surface_tree,
            surface_presentation_feedback_flags_from_states, surface_primary_scanout_output,
            update_surface_primary_scanout_output,
        },
    },
    input::{Seat, SeatState, keyboard::XkbConfig, pointer::CursorImageStatus},
    output::{Mode as OutputMode, Output, PhysicalProperties, Subpixel},
    reexports::{
        calloop::{
            EventLoop, Interest, LoopHandle, Mode, PostAction,
            channel::{Channel, Event},
            generic::Generic,
            timer::{TimeoutAction, Timer},
        },
        input::Libinput,
        wayland_protocols::wp::color_management::v1::server::wp_color_manager_v1::WpColorManagerV1,
        wayland_protocols::wp::presentation_time::server::wp_presentation_feedback,
        wayland_protocols::xdg::shell::server::xdg_toplevel::State as XdgState,
        wayland_server::{
            Display, DisplayHandle,
            backend::{ClientData, ClientId, DisconnectReason, GlobalId},
        },
    },
    utils::{Clock, DeviceFd, Logical, Monotonic, Physical, Point, Rectangle, Size, Transform},
    wayland::{
        compositor::{CompositorClientState, CompositorState, with_states},
        dmabuf::{DmabufGlobal, DmabufState},
        drm_syncobj::{DrmSyncobjState, supports_syncobj_eventfd},
        output::OutputManagerState,
        pointer_constraints::PointerConstraintsState,
        presentation::PresentationState,
        relative_pointer::RelativePointerManagerState,
        selection::data_device::DataDeviceState,
        shell::xdg::{SurfaceCachedState, XdgShellState, XdgToplevelSurfaceData},
        shm::ShmState,
        socket::ListeningSocketSource,
        viewporter::ViewporterState,
    },
};
use std::os::fd::OwnedFd;
use std::sync::Mutex;
use std::{
    collections::HashSet,
    ffi::CString,
    sync::{Arc, mpsc::Sender},
    time::{Duration, Instant},
};
use tracing::debug;

mod focus;
mod input;
mod rendering;

pub use self::focus::*;
pub use self::input::*;
pub use self::rendering::*;
#[cfg(feature = "cuda")]
use crate::utils::allocator::GsCUDABuf;
use crate::utils::allocator::{
    GsBuffer, GsBufferType, GsDmaBuf, GsGlesbuffer, GsNv12Buf, GsVulkanBuf, VideoInfoTypes,
    gst_video_format_to_drm_fourcc, gst_video_format_to_drm_modifier, new_gbm_device,
};
use crate::utils::device::gpu::GPUDevice;
use crate::utils::renderer::setup_renderer;
use crate::utils::vulkan_share::VulkanShare;
use crate::{
    utils::RenderTarget,
    wayland::protocols::{
        frog_color_management::create_frog_color_management_global, wl_drm::create_drm_global,
    },
};

#[derive(Debug, Default)]
pub struct ClientState {
    pub compositor_state: CompositorClientState,
}

impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {}
    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
}

#[allow(dead_code)]
pub struct State {
    pub handle: LoopHandle<'static, State>,
    should_quit: bool,
    pub(crate) clock: Clock<Monotonic>,

    // render
    pub(crate) dtr: Option<OutputDamageTracker>,
    pub(crate) output_buffer: Option<GsBufferType>,
    render_node: Option<DrmNode>,
    pub renderer: GlesRenderer,
    dmabuf_global: Option<(DmabufGlobal, GlobalId)>,
    last_render: Option<Instant>,
    /// WOLF_HDR_CM per-frame PQ-passthrough selector: true when the active fullscreen surface's
    /// most-recent committed buffer is a 10-bit fourcc (gamescope's already-PQ BT.2020 HDR
    /// output, XB30/AB30/XR30/AR30). Threaded into the Vulkan converter's `convert()` so a
    /// 10-bit frame uses the matrix-only passthrough shader instead of re-applying PQ. Always
    /// false unless WOLF_HDR_CM is set (set only in the compositor commit handler).
    pub(crate) current_input_is_pq: bool,

    // management
    pub output: Option<Output>,
    pub video_info: Option<VideoInfo>,
    pub seat: Seat<Self>,
    pub space: Space<Window>,
    pub popups: PopupManager,
    pub(crate) pointer_location: Point<f64, Logical>,
    pub(crate) pointer_absolute_location: Point<f64, Logical>,
    last_pointer_movement: Instant,
    cursor_element: MemoryRenderBuffer,
    pub cursor_state: CursorImageStatus,
    surpressed_keys: HashSet<u32>,
    pub pending_windows: Vec<Window>,
    input_context: Libinput,

    // wayland state
    pub dh: DisplayHandle,
    pub compositor_state: CompositorState,
    pub drm_syncobj_state: Option<DrmSyncobjState>,
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
    cursor_event_count: i32,
    pub single_pixel_buffer_state: SinglePixelBufferState,
    /// `wp_color_manager_v1` global id, present only when `WOLF_HDR_CM` is set. Gated so
    /// that advertising color-management (which changes HDR clients' behaviour) stays
    /// opt-in until the buffer-import side is ready.
    color_mgmt_global: Option<GlobalId>,
    /// `frog_color_management_v1` factory global id, present only when `WOLF_HDR_CM` is set.
    /// gamescope's HDR path uses frog instead of `wp_color_management_v1`; both feed the same
    /// shared per-surface `SurfaceHdrColor`.
    frog_color_mgmt_global: Option<GlobalId>,
    /// Reverse channel (compositor -> element) used to signal OUTPUT HDR-state changes.
    /// `Some` only when `WOLF_HDR_CM` is set; `None` keeps the per-frame check a no-op so
    /// behaviour is exactly as before. See [`State::update_hdr_state`].
    hdr_state_tx: Option<Sender<Command>>,
    /// Last OUTPUT HDR state signalled. The stored-bool compare is the debounce: we only
    /// log + signal on an actual change. Defaults to `false` (SDR).
    last_hdr_state: bool,
    /// When the current candidate HDR<->SDR flip was first observed; the flip is only
    /// committed (TV switched) once it has held for [`HDR_DEBOUNCE`]. `None` = no pending
    /// flip. See [`State::update_hdr_state`].
    hdr_candidate_since: Option<Instant>,
    /// This element's Vulkan-encode device share. A clone of the gst element's own
    /// `Arc<VulkanShare>`, so the compositor thread reads the SAME per-element device the
    /// element mints -- not a process-global singleton. Read in `apply_video_info` when
    /// building the `memory:VulkanImage` output ring. Replaced by the real share in [`init`];
    /// the `State::new` default is an empty placeholder.
    pub(crate) vulkan_share: Arc<VulkanShare>,
}

/// HDR-capable dmabuf fourccs advertised to clients under WOLF_HDR_CM (when the GLES
/// renderer can import them): fp16 scRGB-linear (`Abgr16161616f`) and 10-bit (`Abgr2101010`
/// / `Argb2101010`). These let HDR clients submit real HDR buffers instead of 8-bit sRGB.
/// How long a candidate HDR<->SDR output-state change must hold before it's committed
/// (and the TV is told to switch mode). Filters the rapid flicker from stray 8-bit frames
/// between 10-bit game frames; each real switch blanks the TV ~1-2s, so brief flips must not
/// trigger it.
const HDR_DEBOUNCE: Duration = Duration::from_millis(600);

const HDR_IMPORT_FOURCCS: [Fourcc; 6] = [
    Fourcc::Abgr16161616f,
    Fourcc::Xbgr16161616f,
    Fourcc::Abgr2101010,
    Fourcc::Xbgr2101010,
    Fourcc::Argb2101010,
    Fourcc::Xrgb2101010,
];

/// Add the HDR-capable dmabuf formats (fp16 / 10-bit) the GLES renderer can actually
/// *import* (queried from `ImportDma::dmabuf_formats`, i.e. the EGL texture-import set) to
/// `formats`, so HDR clients submit HDR buffers. Only the HDR fourccs the renderer supports
/// are added (never widening the SDR advertisement), skipping any already present. Logs the
/// formats advertised. Called only when WOLF_HDR_CM is set.
fn advertise_hdr_dmabuf_formats(renderer: &GlesRenderer, formats: &mut Vec<DrmFormat>) {
    use smithay::backend::renderer::ImportDma;
    let importable = renderer.dmabuf_formats();
    // Diagnostic: dump the distinct importable fourccs so we can see what the EGL actually
    // reports (and whether HDR formats appear under an unexpected fourcc).
    let mut codes: Vec<_> = importable.iter().map(|f| f.code).collect();
    codes.sort_by_key(|c| *c as u32);
    codes.dedup();
    tracing::info!(
        "WOLF_HDR_CM: renderer importable dmabuf fourccs ({}): {:?}",
        codes.len(),
        codes
    );
    // All importable HDR formats (regardless of whether they're already in `formats` from the
    // render set). Warn only if NONE are importable; otherwise ensure each is advertised.
    let hdr_importable: Vec<DrmFormat> = importable
        .iter()
        .filter(|f| HDR_IMPORT_FOURCCS.contains(&f.code))
        .copied()
        .collect();
    if hdr_importable.is_empty() {
        tracing::warn!(
            "WOLF_HDR_CM: GLES renderer imports no fp16/10-bit dmabuf formats; HDR clients \
             will fall back to 8-bit"
        );
        return;
    }
    let mut newly = 0usize;
    for f in &hdr_importable {
        if !formats.contains(f) {
            formats.push(*f);
            newly += 1;
        }
    }
    let mut hdr_codes: Vec<_> = hdr_importable.iter().map(|f| f.code).collect();
    hdr_codes.sort_by_key(|c| *c as u32);
    hdr_codes.dedup();
    tracing::info!(
        "WOLF_HDR_CM: {} HDR-capable dmabuf format(s) importable ({} newly advertised): {:?}",
        hdr_importable.len(),
        newly,
        hdr_codes
    );
}

impl State {
    pub fn new(
        render_target: &RenderTarget,
        dh: &DisplayHandle,
        input_context: &Libinput,
        event_loop_handle: LoopHandle<'static, State>,
    ) -> Self {
        let clock = Clock::new();

        // init state
        let compositor_state = CompositorState::new_v6::<State>(&dh);
        let data_device_state = DataDeviceState::new::<State>(&dh);
        let mut dmabuf_state = DmabufState::new();
        let output_state = OutputManagerState::new_with_xdg_output::<State>(&dh);
        let presentation_state = PresentationState::new::<State>(&dh, clock.id() as _);
        let relative_ptr_state = RelativePointerManagerState::new::<State>(&dh);
        let pointer_constraints_state = PointerConstraintsState::new::<State>(&dh);
        let mut seat_state = SeatState::new();
        let shell_state = XdgShellState::new::<State>(&dh);
        let viewporter_state = ViewporterState::new::<State>(&dh);
        let single_pixel_buffer_state = SinglePixelBufferState::new::<Self>(&dh);

        // Color management (staging wp_color_manager_v1). Gated behind WOLF_HDR_CM:
        // advertising it makes HDR clients enable their HDR path and tags HDR surfaces,
        // but the buffer-import side isn't ready yet, so it must be opt-in. When unset the
        // global is never created and behaviour is exactly as before.
        let color_mgmt_global = if std::env::var("WOLF_HDR_CM").is_ok() {
            tracing::info!(
                "WOLF_HDR_CM set: advertising wp_color_manager_v1 (HDR-capable PQ/BT2020 output)"
            );
            Some(dh.create_global::<State, WpColorManagerV1, _>(1, ()))
        } else {
            None
        };

        // frog_color_management_v1 (gamescope's HDR path). Same WOLF_HDR_CM gate; gamescope
        // does NOT speak wp_color_management_v1, so without this its real PQ signal + mastering
        // metadata never reach us. Writes the same shared SurfaceHdrColor as wp above.
        let frog_color_mgmt_global = if std::env::var("WOLF_HDR_CM").is_ok() {
            tracing::info!(
                "WOLF_HDR_CM set: advertising frog_color_management_v1 (gamescope HDR path)"
            );
            Some(create_frog_color_management_global::<State>(&dh))
        } else {
            None
        };

        let render_node: Option<DrmNode> = render_target.clone().into();

        // No `mut`: with the `bind_wl_display` call gone, nothing in this scope takes
        // `renderer` mutably before it moves into `State`.
        let renderer = setup_renderer(render_node);

        let shm_state = ShmState::new::<State>(&dh, vec![]);
        let dmabuf_global = if let RenderTarget::Hardware(node) = render_target {
            let mut formats = Bind::<Dmabuf>::supported_formats(&renderer)
                .expect("Failed to query formats")
                .into_iter()
                .collect::<Vec<_>>();

            // WOLF_HDR_CM: additionally advertise the fp16 / 10-bit dmabuf formats the GLES
            // renderer can *import*, so HDR clients submit HDR (scRGB-fp16 / 10-bit PQ)
            // buffers instead of 8-bit sRGB. Only HDR-capable fourccs the renderer actually
            // imports are added; unset = exactly the render-target format set as before.
            if std::env::var("WOLF_HDR_CM").is_ok() {
                advertise_hdr_dmabuf_formats(&renderer, &mut formats);
            }

            let dmabuf_default_feedback =
                DmabufFeedbackBuilder::new(node.dev_id(), formats.clone()).build();

            let dmabuf_global = if let Ok(default_feedback) = dmabuf_default_feedback {
                dmabuf_state.create_global_with_default_feedback::<State>(&dh, &default_feedback)
            } else {
                tracing::warn!("Failed to create default feedback for dmabuf, falling back to v3");
                dmabuf_state.create_global::<State>(&dh, formats.clone())
            };

            // No `bind_wl_display` here. Its only product is the `EGLBufferReader` behind
            // `BufferType::Egl`, and nothing in this compositor can produce such a buffer:
            // every buffer-carrying global we advertise resolves earlier in smithay's
            // `buffer_type()` dispatch. `ShmState` gives `Shm`, `DmabufState` gives `Dma`,
            // `SinglePixelBufferState` gives `SinglePixel`, and our own `wl_drm` below hands
            // the `WlBuffer` a `Dmabuf` as user data, so it resolves as `Dma` too. The bind
            // therefore enabled an import path with no possible client, while logging
            // "Failed to initialize EGL hardware-acceleration" on every start where the
            // extension is missing -- which reads as a fallback to software rendering that
            // never happened.

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

        let drm_syncobj_state = if let RenderTarget::Hardware(node) = render_target {
            match node.dev_path() {
                Some(path) => match std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&path)
                {
                    Ok(file) => {
                        let device_fd = DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file)));
                        if supports_syncobj_eventfd(&device_fd) {
                            tracing::info!("Enabling explicit sync (linux-drm-syncobj-v1)");
                            Some(DrmSyncobjState::new::<State>(&dh, device_fd))
                        } else {
                            tracing::warn!(
                                "DRM device does not support syncobj eventfd; explicit sync disabled"
                            );
                            None
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            ?err,
                            "Failed to open render node for syncobj; explicit sync disabled"
                        );
                        None
                    }
                },
                None => {
                    tracing::warn!("Render node has no device path; explicit sync disabled");
                    None
                }
            }
        } else {
            None
        };

        let cursor_element = MemoryRenderBuffer::from_memory(
            MemoryBuffer::from_slice(CURSOR_DATA_BYTES, Fourcc::Abgr8888, (64, 64)),
            1,
            Transform::Normal,
            None,
        );

        let space = Space::default();

        let mut seat = seat_state.new_wl_seat(&dh, "seat-0");
        seat.add_keyboard(XkbConfig::default(), 200, 25)
            .expect("Failed to add keyboard to seat");
        seat.add_pointer();
        seat.add_touch();

        State {
            handle: event_loop_handle,
            should_quit: false,
            clock,

            renderer,
            dtr: None,
            output_buffer: None,
            render_node,
            dmabuf_global,
            video_info: None,
            last_render: None,
            current_input_is_pq: false,

            space,
            popups: PopupManager::default(),
            seat,
            output: None,
            pointer_location: (0., 0.).into(),
            pointer_absolute_location: (0., 0.).into(),
            last_pointer_movement: Instant::now(),
            cursor_element,
            cursor_state: CursorImageStatus::default_named(),
            cursor_event_count: 0,
            surpressed_keys: HashSet::new(),
            pending_windows: Vec::new(),
            input_context: input_context.clone(),

            dh: dh.clone(),
            compositor_state,
            drm_syncobj_state,
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
            single_pixel_buffer_state,
            color_mgmt_global,
            frog_color_mgmt_global,
            hdr_state_tx: None,
            last_hdr_state: false,
            vulkan_share: VulkanShare::new(),
            hdr_candidate_since: None,
        }
    }

    /// Whether the active fullscreen surface is HDR (BT.2100 PQ / BT.2020, per
    /// `wp_color_management_v1`). The compositor forces one fullscreen toplevel at a time,
    /// so the first mapped window in the space is the active one. `false` when no window is
    /// mapped or it carries no (or a non-HDR) image description.
    pub fn output_hdr_state(&self) -> bool {
        // HDR when EITHER the active surface declares HDR via wp_color_management
        // (surface_is_hdr) OR the current composited content is a 10-bit already-PQ buffer
        // (current_input_is_pq). gamescope -- the real Steam/HDR path -- does NOT use the
        // color-management protocol; it just submits 10-bit PQ buffers, so the fourcc-based
        // current_input_is_pq is the signal that actually flips for it. Without this the
        // producer colorimetry never flips to bt2100-pq for a gamescope HDR game.
        if self.current_input_is_pq {
            return true;
        }
        self.space
            .elements()
            .next()
            .and_then(|window| window.wl_surface())
            .map(|surface| crate::wayland::handlers::color_management::surface_is_hdr(&surface))
            .unwrap_or(false)
    }

    /// The active fullscreen surface's HDR mastering / content-light-level gst caps strings,
    /// from whichever color-management protocol provided them (frog for gamescope,
    /// `wp_color_management_v1` for sway). `None` when there is no surface or it carries no
    /// mastering metadata -- the producer then keeps its hardcoded HDR defaults.
    fn active_surface_mastering_caps(&self) -> Option<(String, String)> {
        self.space
            .elements()
            .next()
            .and_then(|window| window.wl_surface())
            .and_then(|surface| {
                crate::wayland::handlers::color_management::surface_mastering_caps(&surface)
            })
    }

    /// Recompute the OUTPUT HDR state and, on an actual change, log it and signal the
    /// element over the reverse channel (so it can post a `wolf-hdr-state` application
    /// message on the GStreamer bus). The stored-bool compare debounces repeats. No-op
    /// unless `WOLF_HDR_CM` wired `hdr_state_tx`, so unset = behaviour as before. Does NOT
    /// touch the producer caps/shader -- this only derives + signals the trigger.
    pub(crate) fn update_hdr_state(&mut self) {
        if self.hdr_state_tx.is_none() {
            return;
        }
        let hdr = self.output_hdr_state();
        if hdr == self.last_hdr_state {
            // Settled back to the current committed state -> cancel any pending flip.
            // This is what filters the rapid HDR<->SDR flicker: a stray 8-bit UI frame
            // between 10-bit game frames flips output_hdr_state for a few ms, but it
            // returns to HDR before the debounce elapses, so the candidate is cancelled
            // and the TV never switches mode.
            self.hdr_candidate_since = None;
            return;
        }
        // `hdr` differs from the committed state -> a candidate flip. Only commit it once
        // it has held continuously for HDR_DEBOUNCE; each real change blanks the TV ~1-2s,
        // so brief transitions (loading screens, menu overlays) must NOT switch it.
        match self.hdr_candidate_since {
            Some(since) if since.elapsed() >= HDR_DEBOUNCE => {
                self.last_hdr_state = hdr;
                self.hdr_candidate_since = None;
                tracing::info!(
                    "output HDR state -> {} (debounced)",
                    if hdr { "HDR" } else { "SDR" }
                );
                // When going HDR, carry the active surface's REAL mastering / CLL metadata
                // (from whichever color-management protocol the nested compositor speaks) so
                // the encoder's SEI reflects the game's actual luminance; `None` => the
                // producer keeps its hardcoded HDR defaults. SDR carries no metadata.
                let (mastering, cll) = if hdr {
                    match self.active_surface_mastering_caps() {
                        Some((m, c)) => (Some(m), Some(c)),
                        None => (None, None),
                    }
                } else {
                    (None, None)
                };
                if let Some(tx) = &self.hdr_state_tx {
                    let _ = tx.send(Command::HdrState {
                        hdr,
                        mastering,
                        cll,
                    });
                }
            }
            Some(_) => {} // candidate still maturing
            None => self.hdr_candidate_since = Some(Instant::now()),
        }
    }
}

/// True when `new` differs from `prev` ONLY in colorimetry -- same pixel format, width, height,
/// and frame rate, but a different colorimetry (matrix/transfer/primaries/range, e.g. a dynamic
/// HDR bt709<->bt2100-pq flip). Used under `WOLF_HDR_CM` to skip the Vulkan converter rebuild
/// for such a re-negotiation: the converter produces correct pixels per frame regardless of the
/// caps colorimetry, so only the downstream caps tag needs to change.
fn colorimetry_only_change(prev: &VideoInfo, new: &VideoInfo) -> bool {
    prev.format() == new.format()
        && prev.width() == new.width()
        && prev.height() == new.height()
        && prev.fps() == new.fps()
        && prev.colorimetry() != new.colorimetry()
}

/// Apply a newly-negotiated `GstVideoInfo` to the compositor state: create or update
/// the (single) Output's mode, rebuild the damage tracker + allocator, recenter the
/// pointer, and re-send configure to every mapped toplevel clamped to the new size.
///
/// Called from the `Command::VideoInfo` handler and from the test suite. The
/// `output_already_running` path is what closes the resolution-switching gap --
/// `space.map_output` must stay one-shot, but everything else is safe and desirable
/// to re-run on every VideoInfo so connected clients observe the new state.
pub(crate) fn apply_video_info(
    state: &mut State,
    video_info: GstVideoInfo,
    render_target: &RenderTarget,
    render_node: Option<DrmNode>,
) {
    let output_already_running = state.output.is_some();
    if output_already_running {
        tracing::info!("Output already running, updating with newly negotiated video info");
    }
    let base_info: VideoInfo = video_info.clone().into();
    debug!(
        "Requested video format: {} .to_fourcc() = {}",
        base_info.format(),
        base_info.format().to_fourcc()
    );
    let size: Size<i32, Physical> = (base_info.width() as i32, base_info.height() as i32).into();
    let framerate = base_info.fps();
    let duration = Duration::from_secs_f64(framerate.numer() as f64 / framerate.denom() as f64);

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

    if !output_already_running {
        state.space.map_output(&output, (0, 0));
    }
    state.dtr = Some(dtr);
    let position = (size.w as f64 / 2.0, size.h as f64 / 2.0).into();
    state.pointer_location = position;
    state.pointer_absolute_location = position;
    let prev_video_info = state.video_info.clone();
    state.video_info = Some(video_info.clone().into());

    // WOLF_HDR_CM (dynamic HDR): the producer flips its output caps colorimetry mid-stream
    // (bt709 SDR <-> bt2100-pq HDR) on the SAME format/resolution/fps. Tearing down and
    // rebuilding the Vulkan converter (GsNv12Buf / VulkanNv12) on the compositor thread for
    // that starves frame production and crashes the live stream. The converter produces correct
    // pixels per frame from current_input_is_pq regardless of the caps colorimetry, so a
    // colorimetry-only re-negotiation can keep the existing converter. The caps tag still
    // propagates to the encoder via the producer's caps event independently of this.
    let keep_converter = std::env::var("WOLF_HDR_CM").is_ok()
        && state.output_buffer.is_some()
        && prev_video_info
            .as_ref()
            .is_some_and(|prev| colorimetry_only_change(prev, &base_info));
    if keep_converter {
        tracing::info!("apply_video_info: colorimetry-only change, keeping converter");
    } else {
        match render_target {
            RenderTarget::Hardware(_) => match video_info {
                GstVideoInfo::RAW(base_info) => {
                    let allocator = GsGlesbuffer::new(&mut state.renderer, base_info)
                        .expect("Failed to create GsGlesbuffer");
                    state.output_buffer = Some(GsBufferType::RAW(allocator));
                }
                GstVideoInfo::DMA(base_info) => {
                    let node = render_node.unwrap();
                    // NV12/P010 output goes through the Vulkan converter (render RGBA -> Vulkan
                    // RGBA->NV12/P010 -> exported dmabuf); any other DMA format is the existing
                    // direct path.
                    let fourcc = gst_video_format_to_drm_fourcc(&base_info);
                    let conv_fmt = match fourcc {
                        Some(smithay::reexports::drm::buffer::DrmFourcc::Nv12) => {
                            Some(crate::utils::vulkan_nv12::PixFmt::Nv12)
                        }
                        Some(smithay::reexports::drm::buffer::DrmFourcc::P010) => {
                            Some(crate::utils::vulkan_nv12::PixFmt::P010)
                        }
                        _ => None,
                    };
                    if let Some(conv_fmt) = conv_fmt {
                        let allocator =
                            GsNv12Buf::new(&mut state.renderer, node, base_info, conv_fmt)
                                .expect("Failed to create GsNv12Buf");
                        state.output_buffer = Some(GsBufferType::NV12(allocator));
                    } else {
                        let allocator =
                            GsDmaBuf::new(node, base_info).expect("Failed to create GsDmaBuf");
                        state.output_buffer = Some(GsBufferType::DMA(allocator));
                    }
                }
                GstVideoInfo::VULKAN(params) => {
                    let node = render_node.unwrap();
                    // The downstream encoder shares its GstVulkanDevice via a GstContext
                    // absorbed in set_context on the *streaming* thread, which races this
                    // (compositor-thread) allocation. Wait for the device to arrive instead
                    // of panicking when it merely hasn't been shared yet. If it never comes,
                    // leave output_buffer unset -- the render loop turns that into a clean
                    // FlowError rather than aborting the process.
                    // Clone the Arc so it can be passed to GsVulkanBuf::new while
                    // `state.renderer` is borrowed mutably below.
                    let vulkan_share = Arc::clone(&state.vulkan_share);
                    if vulkan_share
                        .wait_for_shared_device(Duration::from_secs(5))
                        .is_some()
                    {
                        match GsVulkanBuf::new(
                            &mut state.renderer,
                            node,
                            params.video_info,
                            params.profile,
                            &vulkan_share,
                        ) {
                            Some(allocator) => {
                                state.output_buffer = Some(GsBufferType::VULKAN(allocator))
                            }
                            None => tracing::error!(
                                "Failed to create Vulkan output buffer despite a shared GstVulkanDevice"
                            ),
                        }
                    } else {
                        tracing::error!(
                            "No shared GstVulkanDevice within 5s: the downstream Vulkan encoder \
                         never shared its device. Cannot produce memory:VulkanImage output."
                        );
                    }
                }
                #[cfg(feature = "cuda")]
                GstVideoInfo::CUDA(base_info) => {
                    let egl_display = state
                        .renderer
                        .egl_context()
                        .display()
                        .get_display_handle()
                        .handle;
                    let allocator = GsCUDABuf::new(
                        render_node.unwrap(),
                        base_info.cuda_context,
                        base_info.video_info,
                        Arc::new(Mutex::new(None)),
                        &egl_display,
                    )
                    .expect("Failed to create GsCUDABuf");
                    state.output_buffer = Some(GsBufferType::CUDA(allocator));
                }
            },
            RenderTarget::Software => {
                let allocator = GsGlesbuffer::new(&mut state.renderer, base_info.clone())
                    .expect("Failed to create GsGlesbuffer");
                state.output_buffer = Some(GsBufferType::RAW(allocator));
            }
        }
    }

    let new_size = size
        .to_f64()
        .to_logical(output.current_scale().fractional_scale())
        .to_i32_round();
    for window in state.space.elements() {
        let toplevel = window.toplevel().unwrap();
        let max_size = Rectangle::from_size(
            with_states(toplevel.wl_surface(), |states| {
                states
                    .data_map
                    .get::<XdgToplevelSurfaceData>()
                    .map(|_attrs| {
                        states
                            .cached_state
                            .get::<SurfaceCachedState>()
                            .current()
                            .max_size
                    })
            })
            .unwrap_or(new_size),
        );

        let new_size = max_size
            .intersection(Rectangle::from_size(new_size))
            .map(|rect| rect.size);
        toplevel.with_pending_state(|state| {
            state.size = new_size;
            state.states.set(XdgState::Fullscreen);
            state.states.set(XdgState::Activated);
        });
        toplevel.send_configure();
    }
}

pub(crate) fn init(
    command_src: Channel<Command>,
    render: impl Into<RenderTarget>,
    devices_tx: Sender<Vec<CString>>,
    envs_tx: Sender<Vec<CString>>,
    hdr_state_tx: Sender<Command>,
    vulkan_share: Arc<VulkanShare>,
) {
    let render_target = render.into();
    let _ = devices_tx.send(render_target.clone().as_devices());
    let render_node: Option<DrmNode> = render_target.clone().into();

    let mut event_loop = EventLoop::<State>::try_new().expect("Unable to create event_loop");

    let display = Display::<State>::new().unwrap();
    let dh = display.handle();
    dh.set_default_max_buffer_size(10 * 1024 * 1024);
    // init input backend
    let libinput_context = Libinput::new_from_path(NixInterface);
    let input_context = libinput_context.clone();
    let libinput_backend = LibinputInputBackend::new(libinput_context);

    let mut state = State::new(&render_target, &dh, &input_context, event_loop.handle());
    state.vulkan_share = vulkan_share;

    // Wire the compositor -> element HDR-state reverse channel only under WOLF_HDR_CM;
    // unset leaves `hdr_state_tx` as `None`, making the per-frame HDR check a no-op.
    if std::env::var("WOLF_HDR_CM").is_ok() {
        state.hdr_state_tx = Some(hdr_state_tx);
    }

    // init event loop
    state
        .handle
        .insert_source(libinput_backend, move |event, _, state| {
            state.process_input_event(event)
        })
        .unwrap();

    state
        .handle
        .insert_source(command_src, move |event, _, state| {
            match event {
                Event::Msg(Command::VideoInfo(video_info)) => {
                    apply_video_info(state, video_info, &render_target, render_node.clone());
                }
                Event::Msg(Command::InputDevice(path)) => {
                    tracing::info!(path, "Adding input device.");
                    state.input_context.path_add_device(&path);
                }
                Event::Msg(Command::Buffer(buffer_sender, tracer)) => {
                    let wait = if let Some(last_render) = state.last_render {
                        let base_info = state.video_info.as_ref().unwrap().clone();
                        let framerate = base_info.fps();
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
                        let _span = match tracer {
                            Some(ref tracer) => Some(tracer.trace("render")),
                            None => None,
                        };
                        // Derive + signal the OUTPUT HDR state every frame (no-op unless
                        // WOLF_HDR_CM is set). Runs before the buffer check so transitions
                        // are observed even on frames that fail to produce a buffer.
                        state.update_hdr_state();
                        // apply_video_info may have been unable to set up the output buffer
                        // (e.g. a downstream Vulkan encoder that never shared its
                        // GstVulkanDevice). Fail the frame cleanly instead of letting
                        // create_frame() panic on the missing buffer.
                        if state.output_buffer.is_none() {
                            let _ =
                                buffer_sender.send(Err(SwapBuffersError::TemporaryFailure(Box::<
                                    dyn std::error::Error + Send + Sync,
                                >::from(
                                    "no output buffer: downstream did not share a GstVulkanDevice",
                                ))));
                            state.should_quit = true;
                            return;
                        }
                        if let Err(_) = match state.create_frame() {
                            Ok((buf, render_result)) => {
                                let res = buffer_sender.send(Ok(buf));
                                let rendered_states = &render_result.states;
                                let rendered_damage = render_result.damage.is_some();

                                if let Some(output) = state.output.as_ref() {
                                    let mut output_presentation_feedback =
                                        OutputPresentationFeedback::new(output);
                                    for window in state.space.elements() {
                                        window.with_surfaces(|surface, states| {
                                            update_surface_primary_scanout_output(
                                                surface,
                                                output,
                                                states,
                                                rendered_states,
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
                                                    rendered_states,
                                                )
                                            },
                                        );
                                    }
                                    if rendered_damage {
                                        output_presentation_feedback.presented(
                                            state.clock.now(),
                                            output
                                                .current_mode()
                                                .map(|mode| {
                                                    Refresh::fixed(Duration::from_secs_f64(
                                                        1_000f64 / mode.refresh as f64,
                                                    ))
                                                })
                                                .unwrap_or(Refresh::Unknown),
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

                                state.last_render = Some(now);
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
                #[cfg(feature = "cuda")]
                Event::Msg(Command::UpdateCUDABufferPool(pool)) => {
                    tracing::info!("Updating CUDA buffer pool");
                    if let Some(GsBufferType::CUDA(ref mut cuda_buf)) = state.output_buffer {
                        cuda_buf.buffer_pool = pool;
                    }
                }
                Event::Msg(Command::Quit) | Event::Closed => {
                    state.should_quit = true;
                }
                Event::Msg(Command::KeyboardInput(scancode, key_state)) => {
                    let time: Duration = state.clock.now().into();
                    let keycode = state.scancode_to_keycode(scancode);
                    state.keyboard_input(time.as_millis() as u32, keycode, key_state);
                }
                Event::Msg(Command::PointerMotion(position)) => {
                    let time: Duration = state.clock.now().into();
                    state.pointer_motion(
                        time.as_millis() as u32,
                        time.as_nanos() as u64,
                        position,
                        position,
                    );
                }
                Event::Msg(Command::PointerMotionAbsolute(position)) => {
                    let time: Duration = state.clock.now().into();
                    state.pointer_motion_absolute(time.as_millis() as u32, position);
                }
                Event::Msg(Command::PointerButton(btn_code, btn_state)) => {
                    let time: Duration = state.clock.now().into();
                    state.pointer_button(time.as_millis() as u32, btn_code, btn_state);
                }
                Event::Msg(Command::PointerAxis(horizontal_amount, vertical_amount)) => {
                    let time: Duration = state.clock.now().into();
                    state.pointer_axis(
                        time.as_millis() as u32,
                        AxisSource::Wheel,
                        horizontal_amount * 3.0 / 120.0,
                        vertical_amount * 3.0 / 120.0,
                        Some(horizontal_amount),
                        Some(vertical_amount),
                    );
                }
                Event::Msg(Command::GetSupportedDmaFormats(sender)) => {
                    let formats = Bind::<Dmabuf>::supported_formats(&state.renderer);
                    let supported_formats = match &state.output_buffer {
                        None => match state.render_node {
                            // If there's no output_buffer, we'll return all supported DMA formats
                            Some(node) => {
                                let gbm_dev =
                                    new_gbm_device(node).expect("Failed to create gbm device");
                                formats
                                    .unwrap_or_default()
                                    .iter()
                                    .filter(|f| {
                                        gbm_dev.is_format_supported(
                                            f.code,
                                            BufferObjectFlags::RENDERING,
                                        )
                                    })
                                    .map(|f| *f)
                                    .collect()
                            }
                            None => FormatSet::default(),
                        },
                        Some(output_buffer) => {
                            // If we already have negotiated an output buffer,
                            // that's the only format that we are going to support
                            match output_buffer.get_video_info() {
                                VideoInfoTypes::VideoInfo(_) => FormatSet::default(),
                                VideoInfoTypes::VideoInfoDmaDrm(video_info) => {
                                    let fourcc = gst_video_format_to_drm_fourcc(&video_info);
                                    let modifier = gst_video_format_to_drm_modifier(&video_info);
                                    let drm_format = DrmFormat {
                                        code: fourcc.expect(
                                            "Failed to convert gst_video_format to drm_fourcc",
                                        ),
                                        modifier: modifier.expect(
                                            "Failed to convert gst_video_format to drm_modifier",
                                        ),
                                    };
                                    FormatSet::from_iter([drm_format])
                                }
                            }
                        }
                    };
                    debug!("Supported dma formats: {:?}", supported_formats);
                    let _ = sender.send(supported_formats);
                }
                Event::Msg(Command::GetRenderDevice(sender)) => {
                    let render_device: Option<GPUDevice> = match &state.render_node {
                        Some(node) => {
                            let result = GPUDevice::try_from(*node);
                            match result {
                                Ok(device) => Some(device),
                                Err(err) => {
                                    tracing::warn!("Error during GetRenderDevice: {}", err);
                                    None
                                }
                            }
                        }
                        None => None,
                    };
                    debug!("Render device requested: {:?}", render_device);
                    if let Err(err) = sender.send(render_device) {
                        tracing::warn!(?err, "Failed to send render device.");
                    }
                }
                Event::Msg(Command::TouchDown(id, rel_position)) => {
                    let time: Duration = state.clock.now().into();
                    let logical_position = state
                        .relative_touch_to_logical(rel_position)
                        .expect("Failed to convert relative touch position to logical coordinates");
                    state.touch_down(
                        time.as_millis() as u32,
                        TouchSlot::from(Some(id)),
                        logical_position,
                    );
                }
                Event::Msg(Command::TouchUp(id)) => {
                    let time: Duration = state.clock.now().into();
                    state.touch_up(time.as_millis() as u32, TouchSlot::from(Some(id)));
                }
                Event::Msg(Command::TouchMotion(id, rel_position)) => {
                    let time: Duration = state.clock.now().into();
                    let logical_position = state
                        .relative_touch_to_logical(rel_position)
                        .expect("Failed to convert relative touch position to logical coordinates");
                    state.touch_motion(
                        time.as_millis() as u32,
                        TouchSlot::from(Some(id)),
                        logical_position,
                    );
                }
                Event::Msg(Command::TouchCancel) => {
                    state.touch_cancel();
                }
                Event::Msg(Command::TouchFrame) => {
                    state.touch_frame();
                }
                // Reverse-direction signal: only ever sent compositor -> element over the
                // dedicated `hdr_state_tx` channel, never received on this command channel.
                Event::Msg(Command::HdrState { .. }) => {}
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

    event_loop
        .handle()
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
        state.dh.flush_clients().expect("Failed to flush clients");
        state.space.refresh();
        state.popups.cleanup();

        if state.should_quit {
            signal.stop();
        }
    }) {
        tracing::error!(?err, "Event loop broke.");
    }
}
