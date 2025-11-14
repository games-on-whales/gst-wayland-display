use crate::comp::State;
use smithay::input::Seat;
use smithay::wayland::selection::{SelectionHandler, SelectionTarget};
use smithay::{
    delegate_data_device,
    wayland::selection::data_device::{
        ClientDndGrabHandler, DataDeviceHandler, DataDeviceState, ServerDndGrabHandler,
    },
};
use std::io::Write;
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

impl ServerDndGrabHandler for State {}

impl ClientDndGrabHandler for State {}

impl SelectionHandler for State {
    type SelectionUserData = String;

    fn send_selection(
        &mut self,
        _ty: SelectionTarget,
        _mime_type: String,
        fd: OwnedFd,
        _seat: Seat<Self>,
        user_data: &Self::SelectionUserData,
    ) {
        let mut f = unsafe { std::fs::File::from_raw_fd(fd.into_raw_fd()) };
        write!(&mut f, "{}", user_data).unwrap();
    }
}

impl DataDeviceHandler for State {
    fn data_device_state(&self) -> &DataDeviceState {
        &self.data_device_state
    }
}

delegate_data_device!(State);
