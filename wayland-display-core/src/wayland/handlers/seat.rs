use smithay::{
    backend::input::TabletToolDescriptor,
    delegate_seat, delegate_tablet_manager,
    input::{Seat, SeatHandler, SeatState, pointer::CursorImageStatus},
    reexports::wayland_server::Resource,
    wayland::{selection::data_device::set_data_device_focus, tablet_manager::TabletSeatHandler},
};

use crate::comp::{FocusTarget, State};

impl SeatHandler for State {
    type KeyboardFocus = FocusTarget;
    type PointerFocus = FocusTarget;
    type TouchFocus = FocusTarget;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, seat: &Seat<Self>, focus: Option<&Self::KeyboardFocus>) {
        if let Some(surface) = focus {
            let client = match surface {
                FocusTarget::Wayland(w) => w.toplevel().unwrap().wl_surface().client(),
                FocusTarget::Popup(p) => p.wl_surface().client(),
            };
            set_data_device_focus(&self.dh, seat, client);
        } else {
            set_data_device_focus(&self.dh, seat, None);
        }
    }

    fn cursor_image(&mut self, _seat: &Seat<Self>, image: CursorImageStatus) {
        self.cursor_state = image;
    }
}

delegate_seat!(State);

impl TabletSeatHandler for State {
    fn tablet_tool_image(&mut self, _tool: &TabletToolDescriptor, image: CursorImageStatus) {
        // The pen drives the same cursor as the pointer
        self.cursor_state = image;
    }
}

delegate_tablet_manager!(State);
