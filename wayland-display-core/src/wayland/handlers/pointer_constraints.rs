use smithay::delegate_pointer_constraints;
use smithay::input::pointer::PointerHandle;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::wayland::pointer_constraints::{
    PointerConstraintsHandler, with_pointer_constraint,
};
use smithay::wayland::seat::WaylandFocus;
use crate::comp::{State};

impl PointerConstraintsHandler for State {
    fn new_constraint(&mut self, surface: &WlSurface, pointer: &PointerHandle<Self>) {
        if pointer.current_focus().and_then(|x| x.wl_surface()).as_ref() == Some(surface) {
            with_pointer_constraint(surface, pointer, |constraint| {
                constraint.unwrap().activate();
            });
        }
    }
}

delegate_pointer_constraints!(State); // Needed by SDL in order to lock the pointer to the window