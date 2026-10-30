use smithay::{
    backend::allocator::{Buffer as _, Fourcc},
    backend::egl::fence::EGLFence,
    backend::renderer::{gles::GlesRenderer, utils::on_commit_buffer_handler},
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
        drm_syncobj::{DrmSyncPoint, DrmSyncobjCachedState},
        seat::WaylandFocus,
        shell::xdg::{SurfaceCachedState, XdgPopupSurfaceData, XdgToplevelSurfaceData},
    },
};

use crate::comp::{ClientState, FocusTarget, State};

/// Order the renderer's GL context after the client's explicit-sync acquire point by
/// inserting its fence into the context's command stream (`eglWaitSync`), so every later
/// read of the buffer -- including through a cached EGLImage, which on some drivers (NVIDIA)
/// is otherwise not ordered after the client's rendering -- waits for it on the GPU.
///
/// Called once the commit blocker saw the point signal, so the inserted wait is already
/// satisfied: frames never wait on a client's in-flight rendering (the render is CPU-waited
/// in `create_frame`, which would otherwise couple the compositor's pacing and input to the
/// slowest client). Never waits on the CPU itself, unlike `Renderer::wait`, which falls back
/// to an unbounded CPU wait. Best effort: `false` when the fence cannot be imported.
fn wait_on_gpu(renderer: &GlesRenderer, acquire_point: &DrmSyncPoint) -> bool {
    let display = renderer.egl_context().display();
    if !EGLFence::supports_importing(display) {
        return false;
    }
    let Ok(sync_file) = acquire_point.export_sync_file() else {
        return false;
    };
    let Ok(fence) = EGLFence::import(display, sync_file) else {
        return false;
    };
    // SAFETY: the compositor thread owns this renderer's context; this only binds it.
    if unsafe { renderer.egl_context().make_current() }.is_err() {
        return false;
    }
    fence.wait(display).is_ok()
}

/// Dynamic-HDR diagnostic: log the fourcc (and modifier) of the dmabuf a client just
/// committed to `surface`, so we can see what pixel format an HDR game actually submits
/// (e.g. `Abgr16161616f` for scRGB-fp16, `Abgr2101010` for 10-bit). Logged only when the
/// fourcc changes per surface (avoids per-frame spam); SHM / non-dmabuf buffers are skipped.
fn log_client_buffer_fourcc(surface: &WlSurface) {
    use std::cell::Cell;
    with_states(surface, |states| {
        // BufferAssignment isn't Clone, so match the committed buffer by reference and pull
        // out just the (Copy) fourcc + modifier; the cached_state guard stays alive for the
        // borrow.
        let mut attrs = states.cached_state.get::<SurfaceAttributes>();
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

impl State {
    /// Size a toplevel to the output (fullscreen, unless it declared a max size), activate
    /// it and configure it. Requires an output.
    ///
    /// The configure goes out through Smithay's `send_pending_configure`, which sends only
    /// when the pending state differs from the last state sent to the client (the newest
    /// unacked configure, else the last acked one). The initial configure always goes out,
    /// since Smithay counts a toplevel that hasn't had one as having pending changes; a
    /// re-send to a window that was already configured with this size and these states is
    /// dropped instead of reaching the client as a redundant configure.
    pub(crate) fn send_initial_configure(
        &self,
        window: &smithay::desktop::Window,
        max_size: smithay::utils::Size<i32, smithay::utils::Logical>,
    ) {
        let (Some(toplevel), Some(output)) = (window.toplevel(), self.output.as_ref()) else {
            return;
        };
        if max_size.w == 0 && max_size.h == 0 {
            toplevel.with_pending_state(|state| {
                state.size = Some(
                    output
                        .current_mode()
                        .unwrap()
                        .size
                        .to_f64()
                        .to_logical(output.current_scale().fractional_scale())
                        .to_i32_round(),
                );
                state.states.set(XdgState::Fullscreen);
            });
        }
        toplevel.with_pending_state(|state| {
            state.states.set(XdgState::Activated);
        });
        toplevel.send_pending_configure();
    }
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
            if state.hdr_cm {
                crate::wayland::handlers::color_management::stage_legacy_hdr_buffer(surface);
            }
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
                // Explicit sync: block the commit until the client's acquire timeline point
                // signals, then order the GL context after it as well (see `wait_on_gpu`).
                if let Some(acquire_point) = acquire_point {
                    if let Ok((blocker, source)) = acquire_point.generate_blocker() {
                        if let Some(client) = surface.client() {
                            let res = state.handle.insert_source(source, move |_, _, data| {
                                wait_on_gpu(&data.renderer, &acquire_point);
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

        // Color state was applied with the buffer by Smithay. Log before the window
        // and popup handlers consume current().buffer.
        if self.hdr_cm {
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
                // No output to size it against yet: hold it apart from the toplevels that
                // haven't committed. apply_video_info sends its initial configure.
                self.windows_awaiting_output.push(window);
                return;
            }

            if !initial_configure_sent {
                self.send_initial_configure(&window, max_size);
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
