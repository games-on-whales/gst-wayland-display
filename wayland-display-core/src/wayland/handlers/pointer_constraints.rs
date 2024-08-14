use crate::comp::State;
use smithay::delegate_pointer_constraints;
use smithay::input::pointer::PointerHandle;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::wayland::pointer_constraints::{with_pointer_constraint, PointerConstraintsHandler};
use smithay::wayland::seat::WaylandFocus;

impl PointerConstraintsHandler for State {
    fn new_constraint(&mut self, surface: &WlSurface, pointer: &PointerHandle<Self>) {
        if pointer
            .current_focus()
            .map(|x| &*(x.wl_surface().unwrap()) == surface)
            .unwrap_or(false)
        {
            with_pointer_constraint(surface, pointer, |constraint| {
                constraint.unwrap().activate();
            });
        }
    }
}

delegate_pointer_constraints!(State); // Needed by SDL in order to lock the pointer to the window
