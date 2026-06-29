use smithay::{
    backend::allocator::{Buffer as _, Fourcc},
    backend::renderer::utils::on_commit_buffer_handler,
    delegate_compositor, delegate_single_pixel_buffer,
    desktop::PopupKind,
    reexports::{
        calloop::Interest,
        wayland_protocols::xdg::shell::server::xdg_toplevel::State as XdgState,
        wayland_server::{
            Client, Resource,
            protocol::{wl_buffer::WlBuffer, wl_surface::WlSurface},
        },
    },
    utils::SERIAL_COUNTER,
    wayland::{
        buffer::BufferHandler,
        compositor::{
            BufferAssignment, CompositorClientState, CompositorHandler, CompositorState,
            SurfaceAttributes, add_blocker, add_pre_commit_hook, with_states,
        },
        dmabuf::get_dmabuf,
        drm_syncobj::DrmSyncobjCachedState,
        seat::WaylandFocus,
        shell::xdg::{SurfaceCachedState, XdgPopupSurfaceData, XdgToplevelSurfaceData},
    },
};

use crate::comp::{ClientState, FocusTarget, State};

/// Whether `WOLF_HDR_CM` is set (read once). Gates the per-surface client-buffer-format
/// logging below, which would otherwise be hot in the commit path.
fn hdr_cm_enabled() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("WOLF_HDR_CM").is_ok())
}

/// WOLF_HDR_CM diagnostic: log the fourcc (and modifier) of the dmabuf a client just
/// committed to `surface`, so we can see what pixel format an HDR game actually submits
/// (e.g. `Abgr16161616f` for scRGB-fp16, `Abgr2101010` for 10-bit). Logged only when the
/// fourcc changes per surface (avoids per-frame spam); SHM / non-dmabuf buffers are skipped.
fn log_client_buffer_fourcc(surface: &WlSurface) {
    use std::cell::Cell;
    with_states(surface, |states| {
        // BufferAssignment isn't Clone, so match the committed buffer by reference and pull
        // out just the (Copy) fourcc + modifier; the cached_state guard stays alive for the
        // borrow.
        let attrs = states.cached_state.get::<SurfaceAttributes>();
        let (fourcc, modifier) = match &attrs.current().buffer {
            Some(BufferAssignment::NewBuffer(buffer)) => match get_dmabuf(buffer) {
                Ok(dmabuf) => (dmabuf.format().code, dmabuf.format().modifier),
                Err(_) => return, // not a dmabuf (e.g. SHM); nothing to report
            },
            _ => return,
        };
        let last = states
            .data_map
            .get_or_insert::<Cell<Option<Fourcc>>, _>(|| Cell::new(None));
        if last.get() != Some(fourcc) {
            last.set(Some(fourcc));
            tracing::info!(
                surface = ?surface.id(),
                "client_buffer fourcc={fourcc:?} modifier={modifier:?}"
            );
        }
    });
}

impl BufferHandler for State {
    fn buffer_destroyed(&mut self, _buffer: &WlBuffer) {}
}

impl CompositorHandler for State {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    fn new_surface(&mut self, surface: &WlSurface) {
        add_pre_commit_hook::<Self, _>(surface, move |state, _dh, surface| {
            let mut acquire_point = None;
            let maybe_dmabuf = with_states(surface, |surface_data| {
                acquire_point.clone_from(
                    &surface_data
                        .cached_state
                        .get::<DrmSyncobjCachedState>()
                        .pending()
                        .acquire_point,
                );
                surface_data
                    .cached_state
                    .get::<SurfaceAttributes>()
                    .pending()
                    .buffer
                    .as_ref()
                    .and_then(|assignment| match assignment {
                        BufferAssignment::NewBuffer(buffer) => get_dmabuf(buffer).cloned().ok(),
                        _ => None,
                    })
            });
            if let Some(dmabuf) = maybe_dmabuf {
                // Explicit sync: block the commit on the client's acquire timeline point.
                if let Some(acquire_point) = acquire_point {
                    if let Ok((blocker, source)) = acquire_point.generate_blocker() {
                        if let Some(client) = surface.client() {
                            let res = state.handle.insert_source(source, move |_, _, data| {
                                let dh = data.dh.clone();
                                data.client_compositor_state(&client)
                                    .blocker_cleared(data, &dh);
                                Ok(())
                            });
                            if res.is_ok() {
                                add_blocker(surface, blocker);
                                return;
                            }
                        }
                    }
                }
                // Implicit sync fallback: the client isn't using linux-drm-syncobj-v1,
                // so block on the dmabuf's implicit read-fence instead.
                if let Ok((blocker, source)) = dmabuf.generate_blocker(Interest::READ) {
                    if let Some(client) = surface.client() {
                        let res = state.handle.insert_source(source, move |_, _, data| {
                            let dh = data.dh.clone();
                            data.client_compositor_state(&client)
                                .blocker_cleared(data, &dh);
                            Ok(())
                        });
                        if res.is_ok() {
                            add_blocker(surface, blocker);
                        }
                    }
                }
            }
        });
    }

    fn commit(&mut self, surface: &WlSurface) {
        on_commit_buffer_handler::<Self>(surface);

        // WOLF_HDR_CM: report what pixel format the client committed (so we can confirm an
        // HDR game submits fp16 / 10-bit buffers). Off (no-op) unless WOLF_HDR_CM is set.
        if hdr_cm_enabled() {
            log_client_buffer_fourcc(surface);
        }

        if let Some(window) = self
            .space
            .elements()
            .find(|w| w.wl_surface().map(|s| &*s == surface).unwrap_or(false))
        {
            window.on_commit();
        }
        self.popups.commit(surface);

        // send the initial configure if relevant
        if let Some(idx) = self
            .pending_windows
            .iter_mut()
            .position(|w| w.wl_surface().map(|s| &*s == surface).unwrap_or(false))
        {
            let window = self.pending_windows.swap_remove(idx);

            let toplevel = window.toplevel().unwrap();
            let (initial_configure_sent, max_size) = with_states(surface, |states| {
                let attributes = states.data_map.get::<XdgToplevelSurfaceData>().unwrap();
                let attributes_guard = attributes.lock().unwrap();

                (
                    attributes_guard.initial_configure_sent,
                    states
                        .cached_state
                        .get::<SurfaceCachedState>()
                        .current()
                        .max_size,
                )
            });

            if self.output.is_none() {
                return;
            }

            if !initial_configure_sent {
                if max_size.w == 0 && max_size.h == 0 {
                    toplevel.with_pending_state(|state| {
                        state.size = Some(
                            self.output
                                .as_ref()
                                .unwrap()
                                .current_mode()
                                .unwrap()
                                .size
                                .to_f64()
                                .to_logical(
                                    self.output
                                        .as_ref()
                                        .unwrap()
                                        .current_scale()
                                        .fractional_scale(),
                                )
                                .to_i32_round(),
                        );
                        state.states.set(XdgState::Fullscreen);
                    });
                }
                toplevel.with_pending_state(|state| {
                    state.states.set(XdgState::Activated);
                });
                toplevel.send_configure();
                self.pending_windows.push(window);
            } else {
                let loc = (0, 0);
                self.space.map_element(window.clone(), loc, true);
                self.seat.get_keyboard().unwrap().set_focus(
                    self,
                    Some(FocusTarget::from(window)),
                    SERIAL_COUNTER.next_serial(),
                );
            }

            return;
        }

        if let Some(popup) = self.popups.find_popup(surface) {
            let PopupKind::Xdg(ref popup) = popup else {
                // Our compositor doesn't do input handling in the popup code
                unreachable!()
            };
            let initial_configure_sent = with_states(surface, |states| {
                states
                    .data_map
                    .get::<XdgPopupSurfaceData>()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .initial_configure_sent
            });
            if !initial_configure_sent {
                // NOTE: This should never fail as the initial configure is always
                // allowed.
                popup.send_configure().expect("initial configure failed");
            }

            return;
        };
    }
}

delegate_compositor!(State);
delegate_single_pixel_buffer!(State);
