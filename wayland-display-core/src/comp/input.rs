use super::{State, focus::FocusTarget};
use smithay::backend::input::Keycode;
use smithay::backend::libinput::LibinputInputBackend;
use smithay::input::keyboard::Keysym;
use smithay::reexports::input::event::pointer::PointerEventTrait;
use smithay::wayland::seat::WaylandFocus;
use smithay::wayland::tablet_manager::{TabletDescriptor, TabletSeatTrait, TabletToolHandle};
use smithay::{
    backend::input::{
        AbsolutePositionEvent, Axis, AxisSource, ButtonState, Device, DeviceCapability, Event,
        InputEvent, KeyState, KeyboardKeyEvent, PointerAxisEvent, PointerButtonEvent,
        PointerMotionEvent, ProximityState, TabletToolButtonEvent, TabletToolDescriptor,
        TabletToolEvent, TabletToolProximityEvent, TabletToolTipEvent, TabletToolTipState,
        TouchEvent, TouchSlot,
    },
    input::{
        keyboard::{FilterResult, keysyms},
        pointer::{AxisFrame, ButtonEvent, MotionEvent, RelativeMotionEvent},
        touch::{DownEvent, MotionEvent as TouchMotionEvent, UpEvent},
    },
    reexports::wayland_server::{Resource, protocol::wl_surface::WlSurface},
    reexports::{
        input::LibinputInterface,
        rustix::fs::{Mode, OFlags, open},
    },
    utils::{Logical, Point, SERIAL_COUNTER, Serial},
    wayland::pointer_constraints::{PointerConstraint, with_pointer_constraint},
};
use std::{os::unix::io::OwnedFd, path::Path, time::Instant};

// From linux/input-event-codes.h
const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;
const BTN_STYLUS: u32 = 0x14b;

/// The tablet tool axes that changed in a given event (None = unchanged)
#[derive(Debug, Default, Clone, Copy)]
pub struct TabletToolAxes {
    pub pressure: Option<f64>,
    pub distance: Option<f64>,
    pub tilt: Option<(f64, f64)>,
    pub rotation: Option<f64>,
    pub slider: Option<f64>,
    pub wheel: Option<(f64, i32)>,
}

impl TabletToolAxes {
    /// Every tablet tool event (axis, tip and proximity alike) carries the axes that changed with it;
    /// dropping them on tip/proximity events would leave the client with a stale pressure value.
    pub fn from_event<B: smithay::backend::input::InputBackend, E: TabletToolEvent<B>>(
        event: &E,
    ) -> Self {
        TabletToolAxes {
            pressure: event.pressure_has_changed().then(|| event.pressure()),
            distance: event.distance_has_changed().then(|| event.distance()),
            tilt: event.tilt_has_changed().then(|| event.tilt()),
            rotation: event.rotation_has_changed().then(|| event.rotation()),
            slider: event.slider_has_changed().then(|| event.slider_position()),
            wheel: event
                .wheel_has_changed()
                .then(|| (event.wheel_delta(), event.wheel_delta_discrete())),
        }
    }
}

pub struct NixInterface;

impl LibinputInterface for NixInterface {
    fn open_restricted(&mut self, path: &Path, flags: i32) -> Result<OwnedFd, i32> {
        open(
            path,
            OFlags::from_bits_truncate(flags as u32),
            Mode::empty(),
        )
        .map_err(|err| err.raw_os_error())
    }
    fn close_restricted(&mut self, fd: OwnedFd) {
        let _ = fd;
    }
}

impl State {
    pub fn scancode_to_keycode(&self, scancode: u32) -> Keycode {
        // see: https://github.com/rust-x-bindings/xkbcommon-rs/blob/cb449998d8a3de375d492fb7ec015b2925f38ddb/src/xkb/mod.rs#L50-L58
        Keycode::new(scancode + 8)
    }

    pub fn keyboard_input(&mut self, event_time_msec: u32, keycode: Keycode, state: KeyState) {
        let serial = SERIAL_COUNTER.next_serial();
        let keyboard = self.seat.get_keyboard().unwrap();

        keyboard.input::<(), _>(
            self,
            keycode,
            state,
            serial,
            event_time_msec,
            |data, modifiers, handle| {
                if state == KeyState::Pressed {
                    if modifiers.ctrl && modifiers.shift && !modifiers.alt && !modifiers.logo {
                        match handle.modified_sym() {
                            Keysym::Tab => {
                                if let Some(element) = data.space.elements().last().cloned() {
                                    data.surpressed_keys.insert(keysyms::KEY_Tab);
                                    let location = data.space.element_location(&element).unwrap();
                                    data.space.map_element(element.clone(), location, true);
                                    data.seat.get_keyboard().unwrap().set_focus(
                                        data,
                                        Some(FocusTarget::from(element)),
                                        serial,
                                    );
                                    return FilterResult::Intercept(());
                                }
                            }
                            Keysym::Q => {
                                if let Some(target) =
                                    data.seat.get_keyboard().unwrap().current_focus()
                                {
                                    match target {
                                        FocusTarget::Wayland(window) => {
                                            window.toplevel().unwrap().send_close();
                                        }
                                        _ => return FilterResult::Forward,
                                    };
                                    data.surpressed_keys.insert(keysyms::KEY_Q);
                                    return FilterResult::Intercept(());
                                }
                            }
                            _ => {}
                        }
                    }
                } else {
                    if data.surpressed_keys.remove(&handle.modified_sym().raw()) {
                        return FilterResult::Intercept(());
                    }
                }

                FilterResult::Forward
            },
        );
    }

    pub(crate) fn maybe_activate_pointer_constraint(
        &self,
        new_under: &Option<(FocusTarget, Point<f64, Logical>)>,
        new_location: Point<f64, Logical>,
    ) {
        let pointer = self.seat.get_pointer().unwrap();

        if let Some((under, surface_location)) = new_under
            .as_ref()
            .and_then(|(target, loc)| Some((target.wl_surface()?, loc)))
        {
            with_pointer_constraint(&under, &pointer, |constraint| match constraint {
                Some(constraint) if !constraint.is_active() => {
                    let point = new_location - *surface_location;
                    if constraint
                        .region()
                        .map_or(true, |region| region.contains(point.to_i32_round()))
                    {
                        constraint.activate();
                    }
                }
                _ => {}
            });
        }
    }

    fn can_pointer_move(
        &self,
        under: &Option<(FocusTarget, Point<f64, Logical>)>,
        target_position: Point<f64, Logical>,
    ) -> bool {
        let pointer = self.seat.get_pointer().unwrap();
        let mut should_motion = true;

        let new_under = self
            .space
            .element_under(target_position)
            .map(|(w, pos)| (w.clone().into(), pos.to_f64()));

        if let Some((surface, surface_loc)) =
            under
                .as_ref()
                .and_then(|(target, l): &(FocusTarget, Point<f64, Logical>)| {
                    Some((target.wl_surface()?, l))
                })
        {
            with_pointer_constraint(&surface, &pointer, |constraint| match constraint {
                Some(constraint) if constraint.is_active() => {
                    // Constraint does not apply if not within region
                    if !constraint.region().map_or(true, |x| {
                        x.contains((pointer.current_location() - *surface_loc).to_i32_round())
                    }) {
                        return;
                    }
                    match &*constraint {
                        PointerConstraint::Locked(_locked) => {
                            should_motion = false;
                        }
                        PointerConstraint::Confined(confine) => {
                            // If confined, don't move pointer if it would go outside surface or region
                            if let Some((surface, surface_loc)) = &under {
                                if new_under.as_ref().and_then(
                                    |(under, _): &(FocusTarget, Point<f64, Logical>)| {
                                        under.wl_surface()
                                    },
                                ) != surface.wl_surface()
                                {
                                    should_motion = false;
                                }
                                if let Some(region) = confine.region() {
                                    if !region
                                        .contains((target_position - *surface_loc).to_i32_round())
                                    {
                                        should_motion = false;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            });
        }

        // If pointer is now in a constraint region, activate it
        self.maybe_activate_pointer_constraint(&new_under, target_position);

        should_motion
    }

    pub fn pointer_motion(
        &mut self,
        event_time_msec: u32,
        event_time_usec: u64,
        delta: Point<f64, Logical>,
        delta_unaccelerated: Point<f64, Logical>,
    ) {
        self.last_pointer_movement = Instant::now();
        let pointer = self.seat.get_pointer().unwrap();
        let under: Option<(FocusTarget, Point<f64, Logical>)> = self
            .space
            .element_under(self.pointer_location)
            .map(|(w, pos)| (w.clone().into(), pos.to_f64()));

        // Edge-triggered pointer refocus.
        //
        // smithay only emits wl_pointer.enter on a focus TRANSITION, and broadcasts it to
        // the wl_pointer resources that exist at that instant while recording the new focus
        // unconditionally. This compositor resolves pointer focus once, via a synthetic
        // zero-delta motion in the toplevel map handler. A client that calls
        // wl_seat.get_pointer AFTER that instant loses the race: the enter reached zero
        // resources, smithay's focus is already Some(surface), and every later motion takes
        // smithay's same-target arm (wl_pointer.motion only) -- so the client never learns
        // it has pointer focus and ignores all pointer input, forever. Rootful Xwayland
        // (maps exactly once) fails deterministically; gamescope fails intermittently.
        //
        // Fix: on the first real motion after a map, force smithay's focus to None so the
        // motion below takes the (focus, None) => enter arm and re-emits the enter.
        //
        // The flag is RETAINED (not consumed) when there is no surface under the pointer --
        // there is nothing to refocus onto yet -- and while the pointer is grabbed, since
        // smithay's default click grab is active for as long as a button is held and a
        // forced leave mid-click-drag would break the drag.
        let refocus_target = under
            .as_ref()
            .filter(|_| self.pending_pointer_refocus && !pointer.is_grabbed());
        if let Some((target, _)) = refocus_target {
            // An active pointer constraint proves the client already received its enter
            // (constraints only activate on an entered surface), so the refocus is
            // unnecessary rather than deferred -- clear the flag, don't retry.
            let constrained = target
                .wl_surface()
                .map(|surface| {
                    with_pointer_constraint(
                        &surface,
                        &pointer,
                        |constraint| matches!(constraint, Some(c) if c.is_active()),
                    )
                })
                .unwrap_or(false);

            self.pending_pointer_refocus = false;
            if !constrained {
                pointer.motion(
                    self,
                    None,
                    &MotionEvent {
                        location: self.pointer_location,
                        serial: SERIAL_COUNTER.next_serial(),
                        time: event_time_msec,
                    },
                );
            }
        }

        // Allocated AFTER the refocus block: the re-emitted wl_pointer.enter below must carry
        // a serial NEWER than the forced leave's, since clients echo enter serials back.
        let serial = SERIAL_COUNTER.next_serial();

        let possible_pos = self.clamp_coords(self.pointer_location + delta);

        // Pointer should only move if it's not locked or confined (and going out of bounds)
        if self.can_pointer_move(&under, possible_pos) {
            self.set_pointer_location(possible_pos);

            // Not sure why, but order here matters (at least when using Sway nested)!!!
            // If we send the motion event after the relative_motion it'll behave oddly
            pointer.motion(
                self,
                under.clone(),
                &MotionEvent {
                    location: self.pointer_location,
                    serial,
                    time: event_time_msec,
                },
            );
        }

        // Relative motion is always applied
        pointer.relative_motion(
            self,
            under.map(|(w, pos)| (w, pos.to_f64())),
            &RelativeMotionEvent {
                delta,
                delta_unaccel: delta_unaccelerated,
                utime: event_time_usec,
            },
        );

        pointer.frame(self);
    }

    pub fn set_pointer_location(&mut self, location: Point<f64, Logical>) {
        self.pointer_location = location;
        self.pointer_absolute_location = location;
    }

    pub fn pointer_motion_absolute(&mut self, event_time_msec: u32, position: Point<f64, Logical>) {
        let relative_movement = (
            position.x - self.pointer_absolute_location.x,
            position.y - self.pointer_absolute_location.y,
        )
            .into();

        self.pointer_motion(
            event_time_msec,
            event_time_msec as u64 * 1000,
            relative_movement,
            relative_movement,
        );
        //  pointer_absolute_location should always point to the unclamped position sent by Moonlight
        self.pointer_absolute_location = position;
    }

    pub fn pointer_button(&mut self, event_time_msec: u32, button_code: u32, state: ButtonState) {
        self.last_pointer_movement = Instant::now();
        let serial = SERIAL_COUNTER.next_serial();

        if ButtonState::Pressed == state {
            self.update_keyboard_focus(serial);
        };
        let pointer = self.seat.get_pointer().unwrap();
        pointer.button(
            self,
            &ButtonEvent {
                button: button_code,
                state: state.try_into().unwrap(),
                serial,
                time: event_time_msec,
            },
        );
        pointer.frame(self);
    }

    pub fn pointer_axis(
        &mut self,
        event_time_msec: u32,
        source: AxisSource,
        horizontal_amount: f64,
        vertical_amount: f64,
        horizontal_amount_discrete: Option<f64>,
        vertical_amount_discrete: Option<f64>,
    ) {
        let mut frame = AxisFrame::new(event_time_msec).source(source);
        if horizontal_amount != 0.0 {
            frame = frame.value(Axis::Horizontal, horizontal_amount);
            if let Some(discrete) = horizontal_amount_discrete {
                frame = frame.v120(Axis::Horizontal, discrete as i32);
            }
        } else if source == AxisSource::Finger {
            frame = frame.stop(Axis::Horizontal);
        }
        if vertical_amount != 0.0 {
            frame = frame.value(Axis::Vertical, vertical_amount);
            if let Some(discrete) = vertical_amount_discrete {
                frame = frame.v120(Axis::Vertical, discrete as i32);
            }
        } else if source == AxisSource::Finger {
            frame = frame.stop(Axis::Vertical);
        }
        let pointer = self.seat.get_pointer().unwrap();
        pointer.axis(self, frame);
        pointer.frame(self);
    }

    fn touch_location_transformed<
        B: smithay::backend::input::InputBackend,
        E: AbsolutePositionEvent<B>,
    >(
        &self,
        evt: &E,
    ) -> Option<Point<f64, Logical>> {
        let output = self
            .space
            .outputs()
            .find(|output| output.name().starts_with("eDP"))
            .or_else(|| self.space.outputs().next())?;

        let output_geometry = self.space.output_geometry(output)?;
        let transform = output.current_transform();
        let size = transform.invert().transform_size(output_geometry.size);

        Some(
            transform.transform_point_in(evt.position_transformed(size), &size.to_f64())
                + output_geometry.loc.to_f64(),
        )
    }

    pub fn relative_touch_to_logical(
        &mut self,
        relative_pos: Point<f64, Logical>, // 0.0 to 1.0
    ) -> Option<Point<f64, Logical>> {
        let output = self
            .space
            .outputs()
            .find(|output| output.name().starts_with("eDP"))
            .or_else(|| self.space.outputs().next())?;

        let output_geometry = self.space.output_geometry(output)?;
        let transform = output.current_transform();

        // Size before transform
        let untransformed_size = transform.invert().transform_size(output_geometry.size);
        let size_f64 = untransformed_size.to_f64();

        // Scaled raw position in untransformed space
        let pos = Point::from((relative_pos.x * size_f64.w, relative_pos.y * size_f64.h));

        // Now apply the output transform
        let transformed_pos = transform.transform_point_in(pos, &size_f64);

        // Map to global logical coordinates
        Some(transformed_pos + output_geometry.loc.to_f64())
    }

    pub fn touch_down(
        &mut self,
        event_time_msec: u32,
        slot: TouchSlot,
        location: Point<f64, Logical>,
    ) {
        let serial = SERIAL_COUNTER.next_serial();
        let touch = self.seat.get_touch().unwrap();
        let under = self
            .space
            .element_under(location)
            .map(|(w, pos)| (w.clone().into(), pos.to_f64()));

        touch.down(
            self,
            under,
            &DownEvent {
                slot: slot,
                location: location,
                serial,
                time: event_time_msec,
            },
        );
        touch.frame(self);
    }

    pub fn touch_up(&mut self, event_time_msec: u32, slot: TouchSlot) {
        let serial = SERIAL_COUNTER.next_serial();
        let touch = self.seat.get_touch().unwrap();

        touch.up(
            self,
            &UpEvent {
                slot: slot,
                serial,
                time: event_time_msec,
            },
        );
        touch.frame(self);
    }

    pub fn touch_motion(
        &mut self,
        event_time_msec: u32,
        slot: TouchSlot,
        location: Point<f64, Logical>,
    ) {
        let touch = self.seat.get_touch().unwrap();
        let under = self
            .space
            .element_under(location)
            .map(|(w, pos)| (w.clone().into(), pos.to_f64()));

        touch.motion(
            self,
            under,
            &TouchMotionEvent {
                slot,
                location,
                time: event_time_msec,
            },
        );
        touch.frame(self);
    }

    pub fn touch_cancel(&mut self) {
        let touch = self.seat.get_touch().unwrap();
        touch.cancel(self);
    }

    pub fn touch_frame(&mut self) {
        let touch = self.seat.get_touch().unwrap();
        touch.frame(self);
    }

    pub fn process_input_event(&mut self, event: InputEvent<LibinputInputBackend>) {
        match event {
            InputEvent::Keyboard { event, .. } => {
                self.keyboard_input(event.time_msec(), event.key_code(), event.state());
            }
            InputEvent::PointerMotion { event, .. } => {
                self.pointer_motion(
                    event.time_msec(),
                    event.time_usec(),
                    event.delta(),
                    event.delta_unaccel(),
                );
            }
            InputEvent::PointerMotionAbsolute { event } => {
                if let Some(output) = self.output.as_ref() {
                    let output_size = output
                        .current_mode()
                        .unwrap()
                        .size
                        .to_f64()
                        .to_logical(output.current_scale().fractional_scale())
                        .to_i32_round();

                    let new_x = event.x_transformed(output_size.w);
                    let new_y = event.y_transformed(output_size.h);

                    self.pointer_motion_absolute(event.time_msec(), (new_x, new_y).into());
                }
            }
            InputEvent::PointerButton { event, .. } => {
                self.pointer_button(event.time_msec(), event.button(), event.state());
            }
            InputEvent::PointerAxis { event, .. } => {
                self.last_pointer_movement = Instant::now();
                let horizontal_amount = event
                    .amount(Axis::Horizontal)
                    .or_else(|| event.amount_v120(Axis::Horizontal).map(|x| x * 3.0 / 120.0))
                    .unwrap_or(0.0);
                let vertical_amount = event
                    .amount(Axis::Vertical)
                    .or_else(|| event.amount_v120(Axis::Vertical).map(|y| y * 3.0 / 120.0))
                    .unwrap_or(0.0);
                let horizontal_amount_discrete = event.amount_v120(Axis::Horizontal);
                let vertical_amount_discrete = event.amount_v120(Axis::Vertical);

                self.pointer_axis(
                    event.time_msec(),
                    event.source(),
                    horizontal_amount,
                    vertical_amount,
                    horizontal_amount_discrete,
                    vertical_amount_discrete,
                );
            }
            InputEvent::TouchDown { event, .. } => {
                if let Some(location) = self.touch_location_transformed(&event) {
                    self.touch_down(event.time_msec(), event.slot(), location);
                }
            }
            InputEvent::TouchUp { event, .. } => {
                self.touch_up(event.time_msec(), event.slot());
            }
            InputEvent::TouchMotion { event, .. } => {
                if let Some(location) = self.touch_location_transformed(&event) {
                    self.touch_motion(event.time_msec(), event.slot(), location);
                }
            }
            InputEvent::TouchCancel { .. } => {
                self.touch_cancel();
            }
            InputEvent::TouchFrame { .. } => {
                self.touch_frame();
            }
            InputEvent::DeviceAdded { device } => {
                if Device::has_capability(&device, DeviceCapability::TabletTool) {
                    self.tablet_added(&TabletDescriptor::from(&device));
                }
            }
            InputEvent::DeviceRemoved { device } => {
                if Device::has_capability(&device, DeviceCapability::TabletTool) {
                    self.tablet_removed(&TabletDescriptor::from(&device));
                }
            }
            InputEvent::TabletToolProximity { event, .. } => {
                if let Some(location) = self.touch_location_transformed(&event) {
                    let axes = TabletToolAxes::from_event(&event);
                    self.tablet_tool_proximity(
                        event.time_msec(),
                        &TabletDescriptor::from(&event.device()),
                        &event.tool(),
                        location,
                        axes,
                        event.state() == ProximityState::In,
                    );
                }
            }
            InputEvent::TabletToolAxis { event, .. } => {
                if let Some(location) = self.touch_location_transformed(&event) {
                    let axes = TabletToolAxes::from_event(&event);
                    self.tablet_tool_axis(
                        event.time_msec(),
                        &TabletDescriptor::from(&event.device()),
                        &event.tool(),
                        location,
                        axes,
                    );
                }
            }
            InputEvent::TabletToolTip { event, .. } => {
                // A tip event carries the final axis values with it (libinput does not send a separate
                // axis event for them), so the pressure going back to 0 on tip up is only visible here.
                if let Some(location) = self.touch_location_transformed(&event) {
                    let axes = TabletToolAxes::from_event(&event);
                    self.tablet_tool_tip(
                        event.time_msec(),
                        &TabletDescriptor::from(&event.device()),
                        &event.tool(),
                        location,
                        axes,
                        TabletToolTipEvent::tip_state(&event) == TabletToolTipState::Down,
                    );
                }
            }
            InputEvent::TabletToolButton { event, .. } => {
                self.tablet_tool_button(
                    event.time_msec(),
                    &event.tool(),
                    TabletToolButtonEvent::button(&event),
                    TabletToolButtonEvent::button_state(&event),
                );
            }
            _ => {}
        }
    }

    pub fn tablet_added(&mut self, tablet: &TabletDescriptor) {
        tracing::info!(name = %tablet.name, "Tablet added");
        self.seat.tablet_seat().add_tablet::<Self>(&self.dh, tablet);
    }

    pub fn tablet_removed(&mut self, tablet: &TabletDescriptor) {
        tracing::info!(name = %tablet.name, "Tablet removed");
        let tablet_seat = self.seat.tablet_seat();
        tablet_seat.remove_tablet(tablet);
        // If there are no tablets left, the tools are gone too
        if tablet_seat.count_tablets() == 0 {
            tablet_seat.clear_tools();
        }
        self.release_tablet_pointer_emulation(0);
    }

    /// True when the client owning `surface` has bound the tablet protocol (zwp_tablet_seat_v2).
    ///
    /// Clients that don't speak tablet-v2 (gamescope, most games, older toolkits) would otherwise
    /// see the cursor follow the pen but never receive a click; for those we emulate a pointer.
    fn surface_accepts_tablet(&self, surface: &WlSurface) -> bool {
        let Some(client) = surface.client() else {
            return false;
        };
        let mut accepts = false;
        let _ = self
            .dh
            .backend_handle()
            .with_all_objects_for(client.id(), |id| {
                accepts |= id.interface().name == "zwp_tablet_seat_v2";
            });
        accepts
    }

    fn tablet_focus(
        &self,
        location: Point<f64, Logical>,
    ) -> Option<(WlSurface, Point<f64, Logical>)> {
        self.space.element_under(location).and_then(|(w, pos)| {
            let target: FocusTarget = w.clone().into();
            target.wl_surface().map(|s| (s.into_owned(), pos.to_f64()))
        })
    }

    /// True when the pen over `location` is being delivered as real tablet events, in which case no
    /// wl_pointer event may be sent for it (the client would see - and act on - every stroke twice).
    fn tablet_aware_at(&self, location: Point<f64, Logical>) -> bool {
        self.tablet_focus(location)
            .map(|(surface, _)| self.surface_accepts_tablet(&surface))
            .unwrap_or(false)
    }

    /// Moves the cursor that is composited into the streamed video so that it keeps following the pen,
    /// *without* delivering a wl_pointer event: the tablet-aware client gets zwp_tablet_tool_v2.motion
    /// instead, and duplicating it as pointer motion breaks drawing applications.
    fn tablet_cursor_only_motion(&mut self, location: Point<f64, Logical>) {
        self.last_pointer_movement = Instant::now();
        let clamped = self.clamp_coords(location);
        self.set_pointer_location(clamped);
        // Keep the unclamped position around so that a later absolute mouse event doesn't jump
        self.pointer_absolute_location = location;
    }

    /// Moves the pointer for the pen: a real wl_pointer motion for clients that don't speak tablet-v2,
    /// a render-only cursor update for those that do.
    fn tablet_pointer_motion(&mut self, event_time_msec: u32, location: Point<f64, Logical>) {
        if self.tablet_aware_at(self.clamp_coords(location)) {
            self.tablet_cursor_only_motion(location);
        } else {
            self.pointer_motion_absolute(event_time_msec, location);
        }
    }

    pub fn tablet_tool_proximity(
        &mut self,
        event_time_msec: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
        location: Point<f64, Logical>,
        axes: TabletToolAxes,
        is_in: bool,
    ) {
        let tablet_seat = self.seat.tablet_seat();
        let dh = self.dh.clone();
        tablet_seat.add_tool::<Self>(self, &dh, tool);

        // The cursor always follows the pen, tablet aware or not
        self.tablet_pointer_motion(event_time_msec, location);

        let tablet_handle = tablet_seat.get_tablet(tablet);
        let tool_handle = tablet_seat.get_tool(tool);
        if let (Some(tablet_handle), Some(tool_handle)) = (tablet_handle, tool_handle) {
            if is_in {
                if let Some(under) = self.tablet_focus(self.pointer_location) {
                    tool_handle.proximity_in(
                        self.pointer_location,
                        under,
                        &tablet_handle,
                        SERIAL_COUNTER.next_serial(),
                        event_time_msec,
                    );
                    // The proximity event carries the axes the tool came in with; smithay keeps them
                    // pending until the next motion event flushes them to the client.
                    Self::send_tool_axes(&tool_handle, axes);
                }
            } else {
                tool_handle.proximity_out(event_time_msec);
            }
        }
        if !is_in {
            self.release_tablet_pointer_emulation(event_time_msec);
        }
    }

    /// Forwards every axis that changed to the client; the caller has to send the frame event.
    fn send_tool_axes(tool_handle: &TabletToolHandle, axes: TabletToolAxes) {
        if let Some(pressure) = axes.pressure {
            tool_handle.pressure(pressure);
        }
        if let Some(distance) = axes.distance {
            tool_handle.distance(distance);
        }
        if let Some(tilt) = axes.tilt {
            tool_handle.tilt(tilt);
        }
        if let Some(rotation) = axes.rotation {
            tool_handle.rotation(rotation);
        }
        if let Some(slider) = axes.slider {
            tool_handle.slider_position(slider);
        }
        if let Some((degrees, clicks)) = axes.wheel {
            tool_handle.wheel(degrees, clicks);
        }
    }

    pub fn tablet_tool_axis(
        &mut self,
        event_time_msec: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
        location: Point<f64, Logical>,
        axes: TabletToolAxes,
    ) {
        let tablet_seat = self.seat.tablet_seat();

        // The cursor always follows the pen, tablet aware or not
        self.tablet_pointer_motion(event_time_msec, location);

        let tablet_handle = tablet_seat.get_tablet(tablet);
        let tool_handle = tablet_seat.get_tool(tool);
        if let (Some(tablet_handle), Some(tool_handle)) = (tablet_handle, tool_handle) {
            Self::send_tool_axes(&tool_handle, axes);

            let under = self.tablet_focus(self.pointer_location);
            tool_handle.motion(
                self.pointer_location,
                under,
                &tablet_handle,
                SERIAL_COUNTER.next_serial(),
                event_time_msec,
            );
        }
    }

    pub fn tablet_tool_tip(
        &mut self,
        event_time_msec: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
        location: Point<f64, Logical>,
        axes: TabletToolAxes,
        is_down: bool,
    ) {
        // A tip event doubles as an axis event: it carries the position and the axis values the tip
        // went down / came up with (in particular pressure back to 0 on tip up).
        self.tablet_tool_axis(event_time_msec, tablet, tool, location, axes);

        if is_down {
            let serial = SERIAL_COUNTER.next_serial();
            self.update_keyboard_focus(serial);

            if self.tablet_aware_at(self.pointer_location) {
                if let Some(tool_handle) = self.seat.tablet_seat().get_tool(tool) {
                    tool_handle.tip_down(serial, event_time_msec);
                }
            } else {
                self.tablet_emulated_buttons.insert(BTN_LEFT);
                self.pointer_button(event_time_msec, BTN_LEFT, ButtonState::Pressed);
            }
        } else if self.tablet_emulated_buttons.remove(&BTN_LEFT) {
            self.pointer_button(event_time_msec, BTN_LEFT, ButtonState::Released);
        } else if let Some(tool_handle) = self.seat.tablet_seat().get_tool(tool) {
            tool_handle.tip_up(event_time_msec);
        }
    }

    pub fn tablet_tool_button(
        &mut self,
        event_time_msec: u32,
        tool: &TabletToolDescriptor,
        button: u32,
        state: ButtonState,
    ) {
        // BTN_STYLUS -> right click, BTN_STYLUS2 / BTN_STYLUS3 -> middle click
        let emulated_button = match button {
            BTN_STYLUS => BTN_RIGHT,
            _ => BTN_MIDDLE,
        };

        if state == ButtonState::Pressed {
            if self.tablet_aware_at(self.pointer_location) {
                if let Some(tool_handle) = self.seat.tablet_seat().get_tool(tool) {
                    tool_handle.button(
                        button,
                        state,
                        SERIAL_COUNTER.next_serial(),
                        event_time_msec,
                    );
                }
            } else {
                self.tablet_emulated_buttons.insert(emulated_button);
                self.pointer_button(event_time_msec, emulated_button, ButtonState::Pressed);
            }
        } else if self.tablet_emulated_buttons.remove(&emulated_button) {
            self.pointer_button(event_time_msec, emulated_button, ButtonState::Released);
        } else if let Some(tool_handle) = self.seat.tablet_seat().get_tool(tool) {
            tool_handle.button(button, state, SERIAL_COUNTER.next_serial(), event_time_msec);
        }
    }

    /// Releases any pointer button still held by the pen -> pointer emulation,
    /// so that a pen leaving proximity (or being unplugged) mid-press can't leave a stuck button.
    fn release_tablet_pointer_emulation(&mut self, event_time_msec: u32) {
        let buttons: Vec<u32> = self.tablet_emulated_buttons.drain().collect();
        for button in buttons {
            self.pointer_button(event_time_msec, button, ButtonState::Released);
        }
    }

    fn clamp_coords(&self, pos: Point<f64, Logical>) -> Point<f64, Logical> {
        if let Some(output) = self.output.as_ref() {
            if let Some(mode) = output.current_mode() {
                return (
                    pos.x.max(0.0).min((mode.size.w - 2) as f64),
                    pos.y.max(0.0).min((mode.size.h - 2) as f64),
                )
                    .into();
            }
        }
        pos
    }

    fn update_keyboard_focus(&mut self, serial: Serial) {
        let pointer = self.seat.get_pointer().unwrap();
        let keyboard = self.seat.get_keyboard().unwrap();
        // change the keyboard focus unless the pointer or keyboard is grabbed
        // We test for any matching surface type here but always use the root
        // (in case of a window the toplevel) surface for the focus.
        // So for example if a user clicks on a subsurface or popup the toplevel
        // will receive the keyboard focus. Directly assigning the focus to the
        // matching surface leads to issues with clients dismissing popups and
        // subsurface menus (for example firefox-wayland).
        // see here for a discussion about that issue:
        // https://gitlab.freedesktop.org/wayland/wayland/-/issues/294
        if !pointer.is_grabbed() && !keyboard.is_grabbed() {
            if let Some((window, _)) = self
                .space
                .element_under(self.pointer_location)
                .map(|(w, p)| (w.clone(), p))
            {
                self.space.raise_element(&window, true);
                keyboard.set_focus(self, Some(FocusTarget::from(window)), serial);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comp::State;
    use crate::utils::RenderTarget;
    use smithay::input::keyboard::xkb;
    use smithay::output::{Output, PhysicalProperties, Subpixel};
    use smithay::reexports::calloop::EventLoop;
    use smithay::reexports::input::Libinput;
    use smithay::{reexports::wayland_server::Display, utils::Point};

    struct TestState {
        state: State,
    }

    impl TestState {
        fn new() -> Self {
            let libinput_context = Libinput::new_from_path(NixInterface);
            let event_loop = EventLoop::<State>::try_new().expect("Unable to create event_loop");
            let display = Display::<State>::new().unwrap();
            let dh = display.handle();

            let state = State::new(
                &RenderTarget::Software,
                &dh,
                &libinput_context,
                event_loop.handle(),
            );

            TestState { state }
        }

        fn state(&mut self) -> &mut State {
            &mut self.state
        }
    }

    #[test]
    fn keyboard_scancode_conversion() {
        let mut harness = TestState::new();
        let state = harness.state();

        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap =
            xkb::Keymap::new_from_names(&context, "", "", "us", "", None, xkb::COMPILE_NO_FLAGS)
                .unwrap();
        let xkb_state = xkb::State::new(&keymap);

        // Evdev keycode, from `input-event-codes.h`
        // Linux evdev keycode (30)
        const KEY_A: u32 = 30;
        //     ↓ +8
        // X11 keycode (38)
        let x11_key_code = state.scancode_to_keycode(KEY_A);
        //     ↓ keymap lookup
        // X11 keysym (0x61 = 'a')
        let x11_keysym = xkb_state.key_get_one_sym(x11_key_code);

        assert_eq!(Keysym::a, x11_keysym);
    }

    #[test]
    fn keyboard_input() {
        let mut harness = TestState::new();
        let state = harness.state();
        let kb = state.seat.get_keyboard().unwrap();

        const KEY_A: u32 = 30;
        let test_key_code = state.scancode_to_keycode(KEY_A);
        state.keyboard_input(0, test_key_code, KeyState::Pressed);
        assert!(kb.pressed_keys().contains(&test_key_code));

        state.keyboard_input(0, test_key_code, KeyState::Released);
        assert!(kb.pressed_keys().is_empty());
    }

    #[test]
    fn pointer_motion_moves_pointer_location() {
        let mut harness = TestState::new();
        let state = harness.state();

        state.pointer_location = Point::from((50.0, 50.0));
        let delta = Point::from((15.5, -5.0));
        let expected_location = Point::from((65.5, 45.0));

        // Call the method to test
        state.pointer_motion(0, 0, delta, delta);

        // Check that the internal pointer location is updated
        assert_eq!(state.pointer_location, expected_location);

        // Check that the pointer's location is also updated
        let pointer = state.seat.get_pointer().unwrap();
        assert_eq!(pointer.current_location(), expected_location);
    }

    #[test]
    fn pointer_motion_absolute_moves_pointer_location() {
        let mut harness = TestState::new();
        let state = harness.state();

        state.set_pointer_location(Point::from((50.0, 50.0)));
        let expected_location = Point::from((65.5, 45.0));

        state.pointer_motion_absolute(0, expected_location);

        assert_eq!(state.pointer_location, expected_location);
        let pointer = state.seat.get_pointer().unwrap();
        assert_eq!(pointer.current_location(), expected_location);
    }

    #[test]
    fn clamp_coords_keeps_within_bounds() {
        let mut harness = TestState::new();
        let state = harness.state();
        let output = Output::new(
            "HEADLESS-1".into(),
            PhysicalProperties {
                make: "Virtual".into(),
                model: "Wolf".into(),
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
            },
        );
        output.create_global::<State>(&state.dh);
        output.change_current_state(
            Some(smithay::output::Mode {
                size: (10, 10).into(),
                refresh: 1000,
            }),
            None,
            None,
            None,
        );
        state.output = Some(output);

        let extreme_pos = Point::from((-100.0, 5000.0));
        let clamped = state.clamp_coords(extreme_pos);

        // Should clamp negative x to 0 and large y to 10
        assert!(clamped.x >= 0.0);
        assert!(clamped.y <= 10.0);
    }

    #[test]
    fn scancode_to_keycode_adds_xkb_offset() {
        let mut harness = TestState::new();
        let state = harness.state();

        // xkb/X11 keycodes are evdev scancodes plus the historical +8 offset.
        for scancode in [0u32, 1, 30, 103, 240] {
            assert_eq!(
                state.scancode_to_keycode(scancode),
                Keycode::new(scancode + 8)
            );
        }
    }

    #[test]
    fn clamp_coords_passes_through_in_bounds() {
        let mut harness = TestState::new();
        let state = harness.state();
        let output = Output::new(
            "HEADLESS-1".into(),
            PhysicalProperties {
                make: "Virtual".into(),
                model: "Wolf".into(),
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
            },
        );
        output.create_global::<State>(&state.dh);
        output.change_current_state(
            Some(smithay::output::Mode {
                size: (100, 100).into(),
                refresh: 1000,
            }),
            None,
            None,
            None,
        );
        state.output = Some(output);

        // A point already inside the output is returned unchanged.
        let inside = Point::from((42.0, 17.0));
        let clamped = state.clamp_coords(inside);
        assert_eq!(clamped, inside);
    }
}
