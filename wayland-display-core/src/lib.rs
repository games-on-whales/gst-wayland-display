use smithay::backend::SwapBuffersError;
use smithay::backend::drm::{CreateDrmNodeError, DrmNode};
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
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
    TouchDown(u32, Point<f64, Logical>),
    TouchUp(u32),
    TouchMotion(u32, Point<f64, Logical>),
    TouchCancel,
    TouchFrame,
    Quit,
    /// Compositor -> element signal (reverse direction): the OUTPUT HDR state of the
    /// active fullscreen surface changed. Sent only when dynamic HDR is enabled, over a
    /// dedicated reverse channel (never the element -> compositor command channel), and
    /// drained by the element via [`WaylandDisplay::poll_hdr_state`] so it can surface
    /// the change as a `wolf-hdr-state` application message on the GStreamer bus
    /// and retag its caps. Also sent while HDR when the active surface's metadata changes.
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
    /// Downstream refused the HDR caps: keep the output SDR for the rest of the stream.
    RefuseHdr,
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

/// How long a caller waits for the compositor thread to answer a command before giving up.
/// The streaming thread waits under its pad stream lock, so an unanswered command (a wedged or
/// dead compositor event loop) must not block it forever. Generous on purpose: the first reply
/// can queue behind compositor start-up and first-frame (e.g. CUDA) initialisation, and a tight
/// bound would kill healthy streams.
const COMPOSITOR_REPLY_TIMEOUT: Duration = Duration::from_secs(30);

/// How often [`PendingFrame::wait_unless`] checks whether its wait was cancelled.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(10);

pub struct WaylandDisplay {
    thread_handle: Option<JoinHandle<()>>,
    command_tx: Sender<Command>,
    /// Reverse channel (compositor -> element) carrying `Command::HdrState` whenever the
    /// OUTPUT HDR state changes. Empty unless dynamic HDR is enabled; drained by
    /// [`WaylandDisplay::poll_hdr_state`].
    hdr_state_rx: Receiver<Command>,
    reply_timeout: Duration,
    // Resolved once at construction: the render node never changes, and asking the compositor
    // thread would block callers (the streaming thread, under its pad stream lock) forever if
    // the compositor stopped answering.
    render_device: Option<GPUDevice>,

    pub tracer: Option<Tracer>,
    pub devices: MaybeRecv<Vec<CString>>,
    pub envs: MaybeRecv<Vec<CString>>,
}

pub enum MaybeRecv<T: Clone> {
    Rx(Receiver<T>),
    /// A wait already ran out: the compositor is wedged, so later calls don't wait again.
    TimedOut(Receiver<T>),
    Value(T),
}

impl<T: Clone> MaybeRecv<T> {
    /// Waits up to `timeout` for the compositor to publish the value. Errs if the compositor
    /// thread exited before sending it (e.g. its renderer failed to initialize) or is wedged and
    /// never sends it. After one timeout, later calls only take a value that has arrived since
    /// and never wait again, so callers asking several times don't stack full timeouts.
    pub fn get(&mut self, timeout: Duration) -> Result<&T, RecvTimeoutError> {
        let received = match self {
            MaybeRecv::Rx(recv) => Some(recv.recv_timeout(timeout)),
            MaybeRecv::TimedOut(recv) => Some(recv.try_recv().map_err(|err| match err {
                TryRecvError::Empty => RecvTimeoutError::Timeout,
                TryRecvError::Disconnected => RecvTimeoutError::Disconnected,
            })),
            MaybeRecv::Value(_) => None,
        };
        match received {
            Some(Ok(val)) => *self = MaybeRecv::Value(val),
            Some(Err(err)) => {
                if err == RecvTimeoutError::Timeout {
                    if let MaybeRecv::Rx(_) = self {
                        let MaybeRecv::Rx(recv) =
                            std::mem::replace(self, MaybeRecv::TimedOut(mpsc::channel().1))
                        else {
                            unreachable!()
                        };
                        *self = MaybeRecv::TimedOut(recv);
                    }
                }
                return Err(err);
            }
            None => {}
        }
        match self {
            MaybeRecv::Value(val) => Ok(val),
            MaybeRecv::Rx(_) | MaybeRecv::TimedOut(_) => unreachable!(),
        }
    }

    fn timed_out(&self) -> bool {
        matches!(self, MaybeRecv::TimedOut(_))
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
        let render_device = render_device(&render_target);

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
                    Arc::new(Default::default()),
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
            reply_timeout: COMPOSITOR_REPLY_TIMEOUT,
            render_device,
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
        va_share: Arc<crate::utils::va_share::VaShare>,
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
        let render_device = render_device(&render_target);

        let thread_handle = std::thread::spawn(move || {
            comp::init(
                commands_rx,
                render_target,
                devices_tx,
                envs_tx,
                hdr_state_tx,
                compositor_vulkan_share,
                va_share,
            );
        });

        Ok(WaylandDisplay {
            thread_handle: Some(thread_handle),
            command_tx,
            hdr_state_rx,
            reply_timeout: COMPOSITOR_REPLY_TIMEOUT,
            render_device,
            tracer: None,
            devices: MaybeRecv::Rx(devices_rx),
            envs: MaybeRecv::Rx(envs_rx),
        })
    }

    pub fn devices(&mut self) -> Result<impl Iterator<Item = &str>, RecvTimeoutError> {
        Ok(self
            .devices
            .get(self.reply_timeout)?
            .iter()
            .map(|string| string.to_str().unwrap()))
    }

    pub fn env_vars(&mut self) -> Result<impl Iterator<Item = &str>, RecvTimeoutError> {
        Ok(self
            .envs
            .get(self.reply_timeout)?
            .iter()
            .map(|string| string.to_str().unwrap()))
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
        self.request_frame()?.wait()
    }

    /// Asks the compositor for the next frame without waiting for it, so a caller holding a lock
    /// around the display can release it before blocking in [`PendingFrame::wait`].
    pub fn request_frame(&self) -> Result<PendingFrame, gst::FlowError> {
        // Room for the one reply: the compositor must not block handing over a frame whose
        // waiter was cancelled (PendingFrame::wait_unless) and will only pick it up later.
        let (buffer_tx, buffer_rx) = mpsc::sync_channel(1);
        if let Err(err) = self
            .command_tx
            .send(Command::Buffer(buffer_tx, self.tracer.clone()))
        {
            tracing::warn!(?err, "Failed to send buffer command.");
            return Err(gst::FlowError::Eos);
        }
        Ok(PendingFrame {
            buffer_rx,
            timeout: self.reply_timeout,
        })
    }

    /// Downstream refused the HDR colorimetry: make the compositor stay SDR, starting with
    /// the next frame.
    pub fn refuse_hdr(&self) {
        let _ = self.command_tx.send(Command::RefuseHdr);
    }

    /// Fails when the compositor does not answer within the reply timeout, or is gone. Callers
    /// must not treat that as "no formats": that would silently renegotiate to the wrong caps.
    pub fn get_supported_dma_formats(&self) -> Result<FormatSet, RecvTimeoutError> {
        let (buffer_tx, buffer_rx) = mpsc::sync_channel(0);
        let _ = self
            .command_tx
            .send(Command::GetSupportedDmaFormats(buffer_tx));
        buffer_rx.recv_timeout(self.reply_timeout)
    }

    /// How long calls that wait on the compositor give up after.
    pub fn reply_timeout(&self) -> Duration {
        self.reply_timeout
    }

    pub fn get_render_device(&self) -> Option<GPUDevice> {
        self.render_device.clone()
    }

    /// Drain any pending compositor -> element HDR-state notifications, returning the most
    /// recent state if it changed, or `None` if nothing was signalled. The tuple is
    /// `(hdr, mastering, cll)`: `hdr` is the new output HDR state, and `mastering` / `cll`
    /// are the active surface's real HDR static metadata caps strings (or `None` to fall back
    /// to the producer's hardcoded defaults). The compositor only sends on this channel when
    /// dynamic HDR is enabled, so off = always `None`.
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

/// A frame requested from the compositor. Independent of the display: if the display is dropped
/// meanwhile, the compositor quits and [`PendingFrame::wait`] fails instead of blocking.
pub struct PendingFrame {
    buffer_rx: Receiver<Result<gst::Buffer, SwapBuffersError>>,
    timeout: Duration,
}

impl PendingFrame {
    pub fn wait(self) -> Result<gst::Buffer, gst::FlowError> {
        self.wait_unless(&AtomicBool::new(false))
    }

    /// Like [`PendingFrame::wait`], but gives up with `FlowError::Flushing` soon after `cancel`
    /// is set, so a streaming thread can be unblocked (GstBaseSrc `unlock()`) without waiting
    /// out the reply timeout. Keep the frame after a cancel and wait on it again later: dropping
    /// it makes the compositor fail to deliver the frame, and it shuts down when that happens.
    pub fn wait_unless(&self, cancel: &AtomicBool) -> Result<gst::Buffer, gst::FlowError> {
        let deadline = Instant::now() + self.timeout;
        let result = loop {
            if cancel.load(Ordering::Acquire) {
                return Err(gst::FlowError::Flushing);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self
                .buffer_rx
                .recv_timeout(remaining.min(CANCEL_POLL_INTERVAL))
            {
                Err(RecvTimeoutError::Timeout) if remaining > CANCEL_POLL_INTERVAL => continue,
                result => break result,
            }
        };
        match result {
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
}

fn render_device(render_target: &RenderTarget) -> Option<GPUDevice> {
    let node: Option<DrmNode> = render_target.clone().into();
    let render_device = match GPUDevice::try_from(node?) {
        Ok(device) => Some(device),
        Err(err) => {
            tracing::warn!("Failed to identify the render device: {}", err);
            None
        }
    };
    tracing::debug!("Render device: {:?}", render_device);
    render_device
}

impl Drop for WaylandDisplay {
    fn drop(&mut self) {
        if let Err(err) = self.command_tx.send(Command::Quit) {
            tracing::warn!("Failed to send stop command: {}", err);
            return;
        };
        // A wedged compositor never acts on Quit; joining it unbounded would move the hang into
        // pipeline teardown (stop() → drop). Give up after the reply timeout and detach it.
        let thread_handle = self.thread_handle.take().unwrap();
        // Init already timed out: don't stack a second full timeout on top of the first.
        if self.envs.timed_out() || self.devices.timed_out() {
            tracing::error!("Compositor never finished initializing; detaching it");
            return;
        }
        let deadline = Instant::now() + self.reply_timeout;
        while !thread_handle.is_finished() {
            if Instant::now() >= deadline {
                tracing::error!("Compositor thread did not quit in time; detaching it");
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        if thread_handle.join().is_err() {
            tracing::warn!("Failed to join compositor thread");
        };
    }
}
