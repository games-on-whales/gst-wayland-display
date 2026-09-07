use smithay::backend::SwapBuffersError;
use smithay::backend::drm::CreateDrmNodeError;
pub use smithay::reexports::calloop::channel::{Channel, Sender, channel};

#[cfg(feature = "cuda")]
use crate::utils::allocator::cuda::CUDABufferPool;
use crate::utils::device::gpu::GPUDevice;
use crate::utils::vulkan_share::VulkanShare;
pub use smithay::backend::allocator::{
    Format as DrmFormat, Fourcc, Modifier as DrmModifier, Vendor as DrmVendor, format::FormatSet,
};
pub use smithay::backend::input::{ButtonState, KeyState};
use smithay::utils::{Logical, Point};
use std::ffi::{CString, c_char, c_void};
use std::str::FromStr;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use utils::RenderTarget;

pub(crate) mod comp;
#[cfg(test)]
mod tests;
pub mod utils;
pub(crate) mod wayland;

pub use crate::utils::video_info::GstVideoInfo;

pub enum Command {
    InputDevice(String),
    VideoInfo(GstVideoInfo),
    Buffer(
        SyncSender<Result<gst::Buffer, SwapBuffersError>>,
        Option<Tracer>,
    ),
    #[cfg(feature = "cuda")]
    UpdateCUDABufferPool(Arc<Mutex<Option<CUDABufferPool>>>),
    KeyboardInput(u32, KeyState),
    PointerMotion(Point<f64, Logical>),
    PointerMotionAbsolute(Point<f64, Logical>),
    PointerButton(u32, ButtonState),
    PointerAxis(f64, f64),
    GetSupportedDmaFormats(SyncSender<FormatSet>),
    GetRenderDevice(SyncSender<Option<GPUDevice>>),
    TouchDown(u32, Point<f64, Logical>),
    TouchUp(u32),
    TouchMotion(u32, Point<f64, Logical>),
    TouchCancel,
    TouchFrame,
    Quit,
    /// Compositor -> element signal (reverse direction): the OUTPUT HDR state of the
    /// active fullscreen surface changed. Sent only when `WOLF_HDR_CM` is set, over a
    /// dedicated reverse channel (never the element -> compositor command channel), and
    /// drained by the element via [`WaylandDisplay::poll_hdr_state`] so it can surface
    /// the change as a `wolf-hdr-state` application message on the GStreamer bus.
    ///
    /// `mastering` / `cll` carry the active surface's REAL HDR static metadata (the gst
    /// `mastering-display-info` / `content-light-level` caps strings) when `hdr` is true and
    /// either color-management protocol provided it; `None` means the producer keeps its
    /// hardcoded HDR defaults. Both are `None` when going SDR.
    HdrState {
        hdr: bool,
        mastering: Option<String>,
        cll: Option<String>,
    },
}

#[derive(Clone)]
pub struct Tracer {
    start_fn: extern "C" fn(*const c_char) -> *mut c_void,
    end_fn: extern "C" fn(*mut c_void),
}

pub struct Trace {
    ctx: *mut c_void,
    end_fn: extern "C" fn(*mut c_void),
}

impl Tracer {
    pub fn new(
        start_fn: extern "C" fn(*const c_char) -> *mut c_void,
        end_fn: extern "C" fn(*mut c_void),
    ) -> Self {
        Tracer { start_fn, end_fn }
    }

    pub fn trace(&self, name: &str) -> Trace {
        let trace_name = CString::new(name).unwrap();
        let ctx = (self.start_fn)(trace_name.as_ptr());
        Trace::new(ctx, self.end_fn)
    }
}

impl Trace {
    pub fn new(ctx: *mut c_void, end_fn: extern "C" fn(*mut c_void)) -> Self {
        Trace { ctx, end_fn }
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        (self.end_fn)(self.ctx);
    }
}

pub struct WaylandDisplay {
    thread_handle: Option<JoinHandle<()>>,
    command_tx: Sender<Command>,
    /// Reverse channel (compositor -> element) carrying `Command::HdrState` whenever the
    /// OUTPUT HDR state changes. Empty unless `WOLF_HDR_CM` is set; drained by
    /// [`WaylandDisplay::poll_hdr_state`].
    hdr_state_rx: Receiver<Command>,

    pub tracer: Option<Tracer>,
    pub devices: MaybeRecv<Vec<CString>>,
    pub envs: MaybeRecv<Vec<CString>>,
}

pub enum MaybeRecv<T: Clone> {
    Rx(Receiver<T>),
    Value(T),
}

impl<T: Clone> MaybeRecv<T> {
    pub fn get(&mut self) -> &T {
        match self {
            MaybeRecv::Rx(recv) => {
                let value = recv.recv().unwrap();
                *self = MaybeRecv::Value(value.clone());
                self.get()
            }
            MaybeRecv::Value(val) => val,
        }
    }
}

impl WaylandDisplay {
    pub fn new(render_node: Option<String>) -> Result<WaylandDisplay, CreateDrmNodeError> {
        let (channel_tx, channel_rx) = std::sync::mpsc::sync_channel(0);
        let (devices_tx, devices_rx) = std::sync::mpsc::channel();
        let (envs_tx, envs_rx) = std::sync::mpsc::channel();
        // Reverse channel (compositor -> element) for HDR-state notifications.
        let (hdr_state_tx, hdr_state_rx) = std::sync::mpsc::channel();
        // This constructor has no gst element to answer context queries, so nothing outside
        // the compositor thread mints on it -- but comp::init needs one, and making it
        // per-instance keeps the process-global slots gone on this path too.
        let vulkan_share = VulkanShare::new();
        let compositor_vulkan_share = Arc::clone(&vulkan_share);
        let render_target = RenderTarget::from_str(
            &render_node.unwrap_or_else(|| String::from("/dev/dri/renderD128")),
        )?;

        let thread_handle = std::thread::spawn(move || {
            if let Err(err) = std::panic::catch_unwind(|| {
                // calloops channel is not "UnwindSafe", but the std channel is... *sigh* lets workaround it creatively
                let (command_tx, command_src) = smithay::reexports::calloop::channel::channel();
                channel_tx.send(command_tx).unwrap();
                comp::init(
                    command_src,
                    render_target,
                    devices_tx,
                    envs_tx,
                    hdr_state_tx,
                    compositor_vulkan_share,
                );
            }) {
                tracing::error!(?err, "Compositor thread panic'ed!");
            }
        });
        let command_tx = channel_rx.recv().unwrap();

        Ok(WaylandDisplay {
            thread_handle: Some(thread_handle),
            command_tx,
            hdr_state_rx,
            tracer: None,
            devices: MaybeRecv::Rx(devices_rx),
            envs: MaybeRecv::Rx(envs_rx),
        })
    }

    pub fn new_with_channel(
        render_node: Option<String>,
        command_tx: Sender<Command>,
        commands_rx: Channel<Command>,
        vulkan_share: Arc<VulkanShare>,
    ) -> Result<WaylandDisplay, CreateDrmNodeError> {
        let (devices_tx, devices_rx) = std::sync::mpsc::channel();
        let (envs_tx, envs_rx) = std::sync::mpsc::channel();
        // Reverse channel (compositor -> element) for HDR-state notifications.
        let (hdr_state_tx, hdr_state_rx) = std::sync::mpsc::channel();
        // Per-element Vulkan share: the gst element owns one for its whole lifetime and hands
        // the compositor thread a clone, so producer + compositor + encoder all resolve THIS
        // element's device instead of a process-global singleton.
        let compositor_vulkan_share = Arc::clone(&vulkan_share);
        let render_target = RenderTarget::from_str(
            &render_node.unwrap_or_else(|| String::from("/dev/dri/renderD128")),
        )?;

        let thread_handle = std::thread::spawn(move || {
            comp::init(
                commands_rx,
                render_target,
                devices_tx,
                envs_tx,
                hdr_state_tx,
                compositor_vulkan_share,
            );
        });

        Ok(WaylandDisplay {
            thread_handle: Some(thread_handle),
            command_tx,
            hdr_state_rx,
            tracer: None,
            devices: MaybeRecv::Rx(devices_rx),
            envs: MaybeRecv::Rx(envs_rx),
        })
    }

    pub fn devices(&mut self) -> impl Iterator<Item = &str> {
        self.devices
            .get()
            .iter()
            .map(|string| string.to_str().unwrap())
    }

    pub fn env_vars(&mut self) -> impl Iterator<Item = &str> {
        self.envs
            .get()
            .iter()
            .map(|string| string.to_str().unwrap())
    }

    pub fn add_input_device(&self, path: impl Into<String>) {
        let _ = self.command_tx.send(Command::InputDevice(path.into()));
    }

    pub fn set_video_info(&self, info: GstVideoInfo) {
        let _ = self.command_tx.send(Command::VideoInfo(info));
    }

    pub fn keyboard_input(&self, key: u32, pressed: bool) {
        let state = if pressed {
            KeyState::Pressed
        } else {
            KeyState::Released
        };
        let _ = self.command_tx.send(Command::KeyboardInput(key, state));
    }

    pub fn pointer_motion(&self, x: f64, y: f64) {
        let _ = self.command_tx.send(Command::PointerMotion((x, y).into()));
    }

    pub fn pointer_motion_absolute(&self, x: f64, y: f64) {
        let _ = self
            .command_tx
            .send(Command::PointerMotionAbsolute((x, y).into()));
    }

    pub fn pointer_button(&self, button: u32, pressed: bool) {
        let state = if pressed {
            ButtonState::Pressed
        } else {
            ButtonState::Released
        };
        let _ = self.command_tx.send(Command::PointerButton(button, state));
    }

    pub fn pointer_axis(&self, x: f64, y: f64) {
        let _ = self.command_tx.send(Command::PointerAxis(x, y));
    }

    pub fn touch_down(&self, id: u32, rel_x: f64, rel_y: f64) {
        let _ = self
            .command_tx
            .send(Command::TouchDown(id, (rel_x, rel_y).into()));
    }

    pub fn touch_up(&self, id: u32) {
        let _ = self.command_tx.send(Command::TouchUp(id));
    }

    pub fn touch_motion(&self, id: u32, rel_x: f64, rel_y: f64) {
        let _ = self
            .command_tx
            .send(Command::TouchMotion(id, (rel_x, rel_y).into()));
    }

    pub fn touch_cancel(&self) {
        let _ = self.command_tx.send(Command::TouchCancel);
    }

    pub fn touch_frame(&self) {
        let _ = self.command_tx.send(Command::TouchFrame);
    }

    pub fn frame(&self) -> Result<gst::Buffer, gst::FlowError> {
        let (buffer_tx, buffer_rx) = mpsc::sync_channel(0);
        if let Err(err) = self
            .command_tx
            .send(Command::Buffer(buffer_tx, self.tracer.clone()))
        {
            tracing::warn!(?err, "Failed to send buffer command.");
            return Err(gst::FlowError::Eos);
        }

        match buffer_rx.recv() {
            Ok(Ok(buffer)) => Ok(buffer),
            Ok(Err(err)) => match err {
                SwapBuffersError::AlreadySwapped => unreachable!(),
                SwapBuffersError::ContextLost(_) => Err(gst::FlowError::Eos),
                SwapBuffersError::TemporaryFailure(_) => Err(gst::FlowError::Error),
            },
            Err(err) => {
                tracing::warn!(?err, "Failed to recv buffer ack.");
                Err(gst::FlowError::Error)
            }
        }
    }

    pub fn get_supported_dma_formats(&self) -> FormatSet {
        let (buffer_tx, buffer_rx) = mpsc::sync_channel(0);
        let _ = self
            .command_tx
            .send(Command::GetSupportedDmaFormats(buffer_tx));
        buffer_rx.recv().unwrap()
    }

    pub fn get_render_device(&self) -> Option<GPUDevice> {
        let (buffer_tx, buffer_rx) = mpsc::sync_channel(0);
        let _ = self.command_tx.send(Command::GetRenderDevice(buffer_tx));
        buffer_rx.recv().unwrap()
    }

    /// Drain any pending compositor -> element HDR-state notifications, returning the most
    /// recent state if it changed, or `None` if nothing was signalled. The tuple is
    /// `(hdr, mastering, cll)`: `hdr` is the new output HDR state, and `mastering` / `cll`
    /// are the active surface's real HDR static metadata caps strings (or `None` to fall back
    /// to the producer's hardcoded defaults). The compositor only sends on this channel when
    /// `WOLF_HDR_CM` is set, so unset = always `None`.
    pub fn poll_hdr_state(&self) -> Option<(bool, Option<String>, Option<String>)> {
        let mut latest = None;
        while let Ok(cmd) = self.hdr_state_rx.try_recv() {
            if let Command::HdrState {
                hdr,
                mastering,
                cll,
            } = cmd
            {
                latest = Some((hdr, mastering, cll));
            }
        }
        latest
    }
}

impl Drop for WaylandDisplay {
    fn drop(&mut self) {
        if let Err(err) = self.command_tx.send(Command::Quit) {
            tracing::warn!("Failed to send stop command: {}", err);
            return;
        };
        if self.thread_handle.take().unwrap().join().is_err() {
            tracing::warn!("Failed to join compositor thread");
        };
    }
}
