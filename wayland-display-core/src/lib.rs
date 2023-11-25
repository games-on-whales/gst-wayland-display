use gst_video::VideoInfo;

use smithay::backend::drm::CreateDrmNodeError;
use smithay::backend::SwapBuffersError;
use smithay::reexports::calloop::channel::Sender;

use std::ffi::{CString};
use std::str::FromStr;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::JoinHandle;

use utils::RenderTarget;

pub(crate) mod comp;
pub(crate) mod utils;
pub(crate) mod wayland;

pub(crate) enum Command {
    InputDevice(String),
    VideoInfo(VideoInfo),
    Buffer(SyncSender<Result<gst::Buffer, SwapBuffersError>>),
    Quit,
}

pub struct WaylandDisplay {
    thread_handle: Option<JoinHandle<()>>,
    command_tx: Sender<Command>,

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
        let render_target = RenderTarget::from_str(
            &render_node.unwrap_or_else(|| String::from("/dev/dri/renderD128")),
        )?;

        let thread_handle = std::thread::spawn(move || {
            if let Err(err) = std::panic::catch_unwind(|| {
                // calloops channel is not "UnwindSafe", but the std channel is... *sigh* lets workaround it creatively
                let (command_tx, command_src) = smithay::reexports::calloop::channel::channel();
                channel_tx.send(command_tx).unwrap();
                comp::init(command_src, render_target, devices_tx, envs_tx);
            }) {
                tracing::error!(?err, "Compositor thread panic'ed!");
            }
        });
        let command_tx = channel_rx.recv().unwrap();

        Ok(WaylandDisplay {
            thread_handle: Some(thread_handle),
            command_tx,
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

    pub fn set_video_info(&self, info: VideoInfo) {
        let _ = self.command_tx.send(Command::VideoInfo(info));
    }

    pub fn frame(&self) -> Result<gst::Buffer, gst::FlowError> {
        let (buffer_tx, buffer_rx) = mpsc::sync_channel(0);
        if let Err(err) = self.command_tx.send(Command::Buffer(buffer_tx)) {
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