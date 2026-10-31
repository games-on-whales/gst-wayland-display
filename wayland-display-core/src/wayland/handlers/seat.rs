use smithay::{
    backend::input::TabletToolDescriptor,
    input::tablet::TabletSeatHandler,
    input::{Seat, SeatHandler, SeatState, pointer::CursorImageStatus},
    reexports::wayland_server::Resource,
    reexports::wayland_server::protocol::wl_surface::WlSurface,
    wayland::selection::data_device::set_data_device_focus,
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
        self.set_pointer_cursor_image(image);
    }
}

impl TabletSeatHandler for State {
    // The tool's focus is the surface under it; which surfaces get tablet-v2 events at all
    // (and which a pointer, emulated from the pen) is decided in `comp::input`.
    type ToolFocus = WlSurface;

    fn tablet_tool_image(&mut self, _tool: &TabletToolDescriptor, image: CursorImageStatus) {
        // The pen's image replaces the composited cursor until the mouse moves again.
        self.set_tablet_cursor_image(image);
    }
}
