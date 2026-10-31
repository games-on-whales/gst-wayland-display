use smithay::input::Seat;
use smithay::input::dnd::{DnDGrab, DndGrabHandler, GrabType, Source};
use smithay::input::pointer::Focus;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::Serial;
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::selection::data_device::{
    DataDeviceHandler, DataDeviceState, WaylandDndGrabHandler,
};

use crate::comp::State;

// A client's drag starts a drag-and-drop grab on the pointer or touch that started it, as
// smithay did on its own before the generic DnD rework (its default now cancels the drag).
impl WaylandDndGrabHandler for State {
    fn dnd_requested<S: Source>(
        &mut self,
        source: S,
        _icon: Option<WlSurface>,
        seat: Seat<Self>,
        serial: Serial,
        type_: GrabType,
    ) {
        match type_ {
            GrabType::Pointer => {
                let Some(pointer) = seat.get_pointer() else {
                    return source.cancel();
                };
                let Some(start_data) = pointer.grab_start_data() else {
                    return source.cancel();
                };
                let grab = DnDGrab::new_pointer(&self.dh, start_data, source, seat);
                pointer.set_grab(self, grab, serial, Focus::Keep);
            }
            GrabType::Touch => {
                let Some(touch) = seat.get_touch() else {
                    return source.cancel();
                };
                let Some(start_data) = touch.grab_start_data() else {
                    return source.cancel();
                };
                let grab = DnDGrab::new_touch(&self.dh, start_data, source, seat);
                touch.set_grab(self, grab, serial);
            }
        }
    }
}

impl DndGrabHandler for State {}

impl SelectionHandler for State {
    type SelectionUserData = ();
}

impl DataDeviceHandler for State {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device_state
    }
}
