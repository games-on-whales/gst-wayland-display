use smithay::{
    backend::renderer::{Renderer, sync::SyncPoint, utils::on_commit_buffer_handler},
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
                // Ensure cached EGLImages respect the acquire fence; keep the commit blocker as fallback.
                if let Some(acquire_point) = acquire_point {
                    let sync = SyncPoint::from(acquire_point.clone());
                    if state.renderer.wait(&sync).is_ok() {
                        return;
                    }
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
                // Window::bbox() stays (0,0) until on_commit() recomputes it from the
                // surface tree, and the per-commit on_commit() above only runs for
                // surfaces already in the space. A client that never re-commits its
                // root toplevel after this mapping commit (an idle wev, a launcher
                // rendering via subsurfaces) would keep an empty bbox forever, so
                // Space::element_under() never resolves it and wl_pointer focus is
                // never assigned (keyboard focus, set directly below, is unaffected).
                // Refresh the bbox from the buffer committed just now -- the same fix
                // tests/fixture.rs applies manually for the pointer tests to work.
                window.on_commit();
                self.seat.get_keyboard().unwrap().set_focus(
                    self,
                    Some(FocusTarget::from(window)),
                    SERIAL_COUNTER.next_serial(),
                );
                // Synthetic zero-delta motion: delivers wl_pointer.enter to the newly
                // mapped (and just-raised) toplevel immediately -- without waiting for
                // the next physical motion event -- and runs
                // maybe_activate_pointer_constraint(), so a client that requested a
                // pointer lock/confine before mapping (nested gamescope with
                // --force-grab-cursor) gets its constraint activated the moment its
                // surface is focusable. Pointer focus thereby follows the newest
                // toplevel exactly like keyboard focus above.
                let time: std::time::Duration = self.clock.now().into();
                self.pointer_motion(
                    time.as_millis() as u32,
                    time.as_micros() as u64,
                    (0., 0.).into(),
                    (0., 0.).into(),
                );
                // Arm the edge-triggered pointer refocus for the NEXT motion: the enter
                // emitted by the synthetic motion above reaches only the wl_pointer
                // resources that exist at this instant, and a client that calls
                // wl_seat.get_pointer afterwards (rootful Xwayland always; gamescope
                // intermittently) would never see one, because smithay records focus
                // regardless and every later motion then takes its same-target arm.
                //
                // ORDER IS LOAD-BEARING: this must be set AFTER the synthetic motion.
                // That motion is itself a pointer_motion() call, so arming the flag first
                // would let it consume itself inside the exact race window this fix
                // exists to escape.
                self.pending_pointer_refocus = true;
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
