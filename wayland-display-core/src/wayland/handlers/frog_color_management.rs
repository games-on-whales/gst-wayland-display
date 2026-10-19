use crate::{
    comp::State, wayland::protocols::frog_color_management::delegate_frog_color_management,
};

// frog writes the shared `SurfaceHdrColor` (in `handlers::color_management`) via
// `set_surface_hdr_color`, so there is no per-protocol handler trait to implement here --
// only the dispatch delegation, exactly as `wl_drm` does.
delegate_frog_color_management!(State);
