// SPDX-License-Identifier: GPL-3.0-only

// Server side of gamescope's `frog_color_management_v1` protocol -- the Steam Deck / Steam
// HDR path. gamescope does NOT speak the standard `wp_color_management_v1` (that's what
// sway uses); it ships this cut-down extension instead, so without it we can only detect
// HDR via a 10-bit-fourcc heuristic and have to hardcode the mastering metadata. Handling
// frog gives us the game's REAL PQ signal + mastering / content-light-level values.
//
// The bindings are hand-rolled from the vendored XML exactly like `wl_drm` (the
// `wayland-protocols` crate does not generate frog). The per-surface colour state this
// writes is the SAME shared `SurfaceHdrColor` the `wp_color_management_v1` handler writes,
// so the consumer (`output_hdr_state` + the producer's mastering metadata) is
// protocol-agnostic -- see `wayland::handlers::color_management`.
//
// Everything is gated behind `WOLF_HDR_CM` at the global-creation site in `State::new`;
// when unset the global is never advertised and none of this runs.

// Re-export only the actual code; the `generated` module is boilerplate to isolate the
// scanner output (same pattern as `wl_drm`).
pub use generated::{frog_color_managed_surface, frog_color_management_factory_v1};

mod generated {
    use smithay::reexports::wayland_server::{self, protocol::*};

    pub mod __interfaces {
        use smithay::reexports::wayland_server::protocol::__interfaces::*;
        use wayland_backend;
        wayland_scanner::generate_interfaces!("resources/protocols/frog-color-management-v1.xml");
    }

    use self::__interfaces::*;
    wayland_scanner::generate_server_code!("resources/protocols/frog-color-management-v1.xml");
}

use std::sync::Mutex;

use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, WEnum,
    Weak as WlWeak, backend::GlobalId, protocol::wl_surface::WlSurface,
};

use frog_color_managed_surface::{FrogColorManagedSurface, Primaries, TransferFunction};
use frog_color_management_factory_v1::FrogColorManagementFactoryV1;

use crate::wayland::handlers::color_management::{
    MasteringDisplayInfo, SurfaceHdrColor, set_surface_hdr_color,
};

/// ZST the `GlobalDispatch`/`Dispatch` impls live on; delegated to from `State` by the
/// [`delegate_frog_color_management`] macro (same indirection as `wl_drm`).
pub struct FrogColorManagementState;

/// Accumulated frog colour state for one `frog_color_managed_surface`. The client sets each
/// piece in a separate request, so we accumulate here and mirror the whole thing into the
/// target surface's shared [`SurfaceHdrColor`] after every request.
#[derive(Default)]
struct FrogColorAccum {
    is_pq: bool,
    is_bt2020: bool,
    mastering: Option<MasteringDisplayInfo>,
    max_cll: Option<u32>,
    max_fall: Option<u32>,
}

impl FrogColorAccum {
    fn to_hdr_color(&self) -> SurfaceHdrColor {
        SurfaceHdrColor {
            is_pq: self.is_pq,
            is_bt2020: self.is_bt2020,
            mastering: self.mastering,
            max_cll: self.max_cll,
            max_fall: self.max_fall,
        }
    }
}

/// User data of a `frog_color_managed_surface`: the (weak) target surface so it goes inert
/// once the surface is destroyed, plus the per-surface accumulator.
pub struct FrogSurfaceData {
    surface: WlWeak<WlSurface>,
    accum: Mutex<FrogColorAccum>,
}

// --- frog_color_management_factory_v1 ----------------------------------------------

impl<D> GlobalDispatch<FrogColorManagementFactoryV1, (), D> for FrogColorManagementState
where
    D: GlobalDispatch<FrogColorManagementFactoryV1, ()>
        + Dispatch<FrogColorManagementFactoryV1, ()>
        + Dispatch<FrogColorManagedSurface, FrogSurfaceData>
        + 'static,
{
    fn bind(
        _state: &mut D,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<FrogColorManagementFactoryV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, D>,
    ) {
        // The factory singleton has no bind-time events.
        data_init.init(resource, ());
    }
}

impl<D> Dispatch<FrogColorManagementFactoryV1, (), D> for FrogColorManagementState
where
    D: Dispatch<FrogColorManagementFactoryV1, ()>
        + Dispatch<FrogColorManagedSurface, FrogSurfaceData>
        + 'static,
{
    fn request(
        _state: &mut D,
        _client: &Client,
        _factory: &FrogColorManagementFactoryV1,
        request: frog_color_management_factory_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            frog_color_management_factory_v1::Request::GetColorManagedSurface {
                surface,
                callback,
            } => {
                let cms = data_init.init(
                    callback,
                    FrogSurfaceData {
                        surface: surface.downgrade(),
                        accum: Mutex::new(FrogColorAccum::default()),
                    },
                );
                // Tell gamescope (our nested frog CLIENT) that the output it targets is
                // HDR: ST2084 PQ / BT.2020. gamescope gates nested HDR exposure on this
                // event's transfer_function == ST2084_PQ (WaylandBackend.cpp
                // `bExposeHDRSupport = cv_hdr_enabled && tf == ST2084_PQ`); without it,
                // `gamescope --hdr-enabled` silently falls back to SDR and the game's
                // in-game HDR (and its luminance slider) never engages. Primaries/white
                // are BT.2020 + D65 in frog's 0.00002 units (== gst ×50000); luminance
                // is the display peak gamescope tone-maps the game to. Only runs under
                // WOLF_HDR_CM (the global is gated there).
                // Display peak luminance advertised to gamescope (what the game calibrates
                // its HDR to). Tunable via WOLF_HDR_PEAK_NITS (nits, default 1000) so it can
                // be matched to the client TV without a rebuild; clamped to frog's u16 range.
                // max-full-frame ~= 40% of peak (typical sustained-fullscreen ceiling).
                let peak_nits: u32 = std::env::var("WOLF_HDR_PEAK_NITS")
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .filter(|&n| n > 0)
                    .unwrap_or(1000)
                    .min(65535);
                let max_full_frame = (peak_nits * 2 / 5).max(100);
                // preferred_metadata args: transfer function, then BT.2020 R/G/B + D65 white
                // chromaticities in frog's 0.00002 units (R 0.708,0.292  G 0.170,0.797
                // B 0.131,0.046  W 0.3127,0.3290), then max / min / max-full-frame luminance.
                cms.preferred_metadata(
                    TransferFunction::St2084Pq,
                    35400,
                    14600,
                    8500,
                    39850,
                    6550,
                    2300,
                    15635,
                    16450,
                    peak_nits,
                    1,
                    max_full_frame,
                );
                tracing::info!(
                    peak_nits,
                    "frog: sent preferred_metadata ST2084_PQ/BT2020 (HDR output) to client"
                );
            }
            frog_color_management_factory_v1::Request::Destroy => {}
        }
    }
}

// --- frog_color_managed_surface ----------------------------------------------------

impl<D> Dispatch<FrogColorManagedSurface, FrogSurfaceData, D> for FrogColorManagementState
where
    D: Dispatch<FrogColorManagedSurface, FrogSurfaceData> + 'static,
{
    fn request(
        _state: &mut D,
        _client: &Client,
        _obj: &FrogColorManagedSurface,
        request: frog_color_managed_surface::Request,
        data: &FrogSurfaceData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        use frog_color_managed_surface::Request;

        // Surface gone -> object is inert; drop the request.
        let Ok(surface) = data.surface.upgrade() else {
            return;
        };
        let mut accum = data.accum.lock().unwrap();

        match request {
            Request::SetKnownTransferFunction { transfer_function } => {
                let pq = matches!(transfer_function, WEnum::Value(TransferFunction::St2084Pq));
                if pq && !accum.is_pq {
                    tracing::info!(surface = ?surface.id(), "frog: surface -> PQ (ST2084)");
                }
                accum.is_pq = pq;
            }
            Request::SetKnownContainerColorVolume { primaries } => {
                accum.is_bt2020 = matches!(primaries, WEnum::Value(Primaries::Rec2020));
            }
            Request::SetRenderIntent { .. } => {
                // Only `perceptual` is defined in v1; nothing to store.
            }
            Request::SetHdrMetadata {
                mastering_display_primary_red_x,
                mastering_display_primary_red_y,
                mastering_display_primary_green_x,
                mastering_display_primary_green_y,
                mastering_display_primary_blue_x,
                mastering_display_primary_blue_y,
                mastering_white_point_x,
                mastering_white_point_y,
                max_display_mastering_luminance,
                min_display_mastering_luminance,
                max_cll,
                max_fall,
            } => {
                // frog chromaticity unit is 0.00002, which is exactly gst's 1/50000 unit
                // (0.00002 * 50000 == 1.0), so the raw frog values ARE the gst ×50000 values
                // 1:1. frog max-lum is in cd/m² -> gst wants 0.0001 cd/m² (×10000); frog
                // min-lum is already in 0.0001 cd/m² (1:1).
                accum.mastering = Some(MasteringDisplayInfo {
                    primaries: [
                        mastering_display_primary_red_x,
                        mastering_display_primary_red_y,
                        mastering_display_primary_green_x,
                        mastering_display_primary_green_y,
                        mastering_display_primary_blue_x,
                        mastering_display_primary_blue_y,
                        mastering_white_point_x,
                        mastering_white_point_y,
                    ],
                    max_luminance: max_display_mastering_luminance.saturating_mul(10000),
                    min_luminance: min_display_mastering_luminance,
                });
                // frog max_cll / max_fall are in cd/m² (nits) == gst content-light-level units.
                accum.max_cll = Some(max_cll);
                accum.max_fall = Some(max_fall);
                tracing::info!(
                    surface = ?surface.id(),
                    max_cll,
                    max_fall,
                    "frog: surface hdr_metadata set (mastering + CLL)"
                );
            }
            Request::Destroy => {
                // Destroying resets the surface's colour state back to undefined.
                *accum = FrogColorAccum::default();
                set_surface_hdr_color(&surface, None);
                return;
            }
        }

        let hdr = accum.to_hdr_color();
        drop(accum);
        set_surface_hdr_color(&surface, Some(hdr));
    }
}

/// Create the `frog_color_management_factory_v1` global (version 1). Called only under
/// `WOLF_HDR_CM`.
pub fn create_frog_color_management_global<D>(display: &DisplayHandle) -> GlobalId
where
    D: GlobalDispatch<FrogColorManagementFactoryV1, ()>
        + Dispatch<FrogColorManagementFactoryV1, ()>
        + Dispatch<FrogColorManagedSurface, FrogSurfaceData>
        + 'static,
{
    display.create_global::<D, FrogColorManagementFactoryV1, _>(1, ())
}

macro_rules! delegate_frog_color_management {
    ($(@<$( $lt:tt $( : $clt:tt $(+ $dlt:tt )* )? ),+>)? $ty: ty) => {
        smithay::reexports::wayland_server::delegate_global_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::wayland::protocols::frog_color_management::frog_color_management_factory_v1::FrogColorManagementFactoryV1: ()
        ] => $crate::wayland::protocols::frog_color_management::FrogColorManagementState);
        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::wayland::protocols::frog_color_management::frog_color_management_factory_v1::FrogColorManagementFactoryV1: ()
        ] => $crate::wayland::protocols::frog_color_management::FrogColorManagementState);
        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::wayland::protocols::frog_color_management::frog_color_managed_surface::FrogColorManagedSurface: $crate::wayland::protocols::frog_color_management::FrogSurfaceData
        ] => $crate::wayland::protocols::frog_color_management::FrogColorManagementState);
    };
}
pub(crate) use delegate_frog_color_management;
