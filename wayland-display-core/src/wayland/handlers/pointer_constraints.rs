use crate::comp::State;
use smithay::input::pointer::PointerHandle;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point};
use smithay::wayland::pointer_constraints::{PointerConstraintsHandler, with_pointer_constraint};
use smithay::wayland::seat::WaylandFocus;

impl PointerConstraintsHandler for State {
    fn new_constraint(&mut self, surface: &WlSurface, pointer: &PointerHandle<Self>) {
        if pointer
            .current_focus()
            .map(|x| &*(x.wl_surface().unwrap()) == surface)
            .unwrap_or(false)
        {
            // Resolved through `pointer_focus`, not `Space::element_under`: with a
            // fullscreen fit active the raw (render-space) pointer position misses the
            // window entirely outside its unscaled bbox, and the constraint's region would
            // be evaluated off by the fit scale. This is the lock-before-map path (SDL,
            // nested gamescope with --force-grab-cursor), so it must agree with the motion
            // path that activates constraints later.
            let (focus, position) = self.pointer_focus(self.pointer_location);
            let under = focus.map(|(window, origin)| (window.into(), origin));
            self.maybe_activate_pointer_constraint(&under, position);
        }
    }

    fn cursor_position_hint(
        &mut self,
        surface: &WlSurface,
        pointer: &PointerHandle<Self>,
        location: Point<f64, Logical>,
    ) {
        if with_pointer_constraint(surface, pointer, |constraint| {
            constraint.is_some_and(|c| c.is_active())
        }) {
            let hinted = self
                .space
                .elements()
                .find(|window| window.wl_surface().as_deref() == Some(surface))
                .map(|window| {
                    // `origin` matches `pointer_focus`'s: the window's location in its own
                    // (unfit) coordinate space.
                    let origin = (self.space.element_location(window).unwrap_or_default()
                        - window.geometry().loc)
                        .to_f64();
                    let (scale, offset) = self.window_fit(window);
                    // `location` is surface-local, i.e. already in the CLIENT's space -- the
                    // space smithay's pointer location lives in since the fit landed, so the
                    // handle is set without any transform. `pointer_location` is the
                    // compositor's own RENDER-space position (where the cursor is drawn and
                    // what `clamp_coords` bounds), so it takes the FORWARD fit -- the exact
                    // inverse of `unfit`. Without this the hint silently half-lands under a
                    // 2x fit, and the next relative motion accumulates from the stale spot:
                    // the unlock position for exactly the SDL-style games that lock the
                    // pointer in the first place.
                    (origin + location, origin + offset + location.upscale(scale))
                });

            if let Some((client_pos, render_pos)) = hinted {
                pointer.set_location(client_pos);
                self.set_pointer_location(self.clamp_coords(render_pos));
            }
        }
    }
}
