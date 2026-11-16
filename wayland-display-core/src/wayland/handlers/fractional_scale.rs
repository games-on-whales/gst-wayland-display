use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::wayland::fractional_scale::FractionalScaleHandler;

use crate::comp::{State, set_preferred_ui_scale};

impl FractionalScaleHandler for State {
    /// A client just created a `wp_fractional_scale_v1` for one of its surfaces: answer
    /// immediately with the currently requested UI scale, so a surface that appears after a
    /// `Command::UiScale` still learns about it. Later changes are re-announced from
    /// `comp::announce_ui_scale`.
    ///
    /// The debounce inside `set_preferred_scale` (it only emits when the value *changes*,
    /// `wayland/fractional_scale/mod.rs:238-245`) is correct here, and both orderings are
    /// covered: smithay **replays** an already-stored `preferred_scale` to the new object at
    /// bind time, immediately before calling this handler (`:163-171`). So a value that
    /// `announce_ui_scale` stored against the surface before the client created its object
    /// (which it can: a toplevel is in `pending_windows` from `get_toplevel()` onward, and
    /// `with_fractional_scale` creates the per-surface state on demand) has already been sent
    /// by the time we get here — and this call is then correctly a no-op.
    ///
    /// **Verify this on any smithay re-pin:** if that replay ever goes away, the debounce
    /// turns into a silent "client never receives an initial preferred_scale" bug. The
    /// `previous`/`sent` fields logged below are what would show it.
    fn new_fractional_scale(&mut self, surface: WlSurface) {
        let scale = self.ui_scale;
        let previous = set_preferred_ui_scale(&surface, scale);
        tracing::info!(
            scale,
            ?previous,
            sent = previous != Some(scale),
            "Client created wp_fractional_scale_v1",
        );
    }
}
