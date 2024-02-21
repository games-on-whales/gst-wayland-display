use smithay::delegate_relative_pointer;
use smithay::delegate_pointer_constraints;
use smithay::input::pointer::PointerHandle;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::wayland::output::OutputHandler;
use smithay::wayland::pointer_constraints::PointerConstraintsHandler;

use crate::comp::State;

impl OutputHandler for State {}
impl PointerConstraintsHandler for State{
    fn new_constraint(&mut self, surface: &WlSurface, pointer: &PointerHandle<Self>) {
        todo!()
    }
}

delegate_relative_pointer!(State);
delegate_pointer_constraints!(State); // Needed by SDL in order to lock the pointer to the window