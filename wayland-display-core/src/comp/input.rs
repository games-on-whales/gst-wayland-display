use super::{focus::FocusTarget, State};
use smithay::{
    backend::{
        input::{
            Axis, AxisSource, Event, InputEvent, KeyState, KeyboardKeyEvent, PointerAxisEvent,
            PointerButtonEvent, PointerMotionEvent, ButtonState,
        },
        libinput::LibinputInputBackend,
    },
    input::{
        keyboard::{keysyms, FilterResult},
        pointer::{AxisFrame, ButtonEvent, MotionEvent, RelativeMotionEvent},
    },
    reexports::{
        input::LibinputInterface,
        rustix::{
            fs::{open, OFlags, Mode}
        },
    },
    utils::{Logical, Point, Serial, SERIAL_COUNTER},
    wayland::pointer_constraints::{with_pointer_constraint, PointerConstraint},
};
use std::{
    os::unix::io::OwnedFd,
    path::Path,
    time::Instant,
};
use smithay::input::keyboard::Keysym;
use smithay::wayland::seat::WaylandFocus;

pub struct NixInterface;

impl LibinputInterface for NixInterface {
    fn open_restricted(&mut self, path: &Path, flags: i32) -> Result<OwnedFd, i32> {
        open(path, OFlags::from_bits_truncate(flags as u32), Mode::empty())
            .map_err(|err| err.raw_os_error())
    }
    fn close_restricted(&mut self, fd: OwnedFd) {
        let _ = fd;
    }
}

impl State {
    pub fn process_input_event(&mut self, event: InputEvent<LibinputInputBackend>) {
        match event {
            InputEvent::Keyboard { event, .. } => {
                let keycode = event.key_code();
                let state = event.state();
                let serial = SERIAL_COUNTER.next_serial();
                let time = event.time_msec();
                let keyboard = self.seat.get_keyboard().unwrap();

                keyboard.input::<(), _>(
                    self,
                    keycode,
                    state,
                    serial,
                    time,
                    |data, modifiers, handle| {
                        if state == KeyState::Pressed {
                            if modifiers.ctrl
                                && modifiers.shift
                                && !modifiers.alt
                                && !modifiers.logo
                            {
                                match handle.modified_sym() {
                                    Keysym::Tab => {
                                        if let Some(element) = data.space.elements().last().cloned()
                                        {
                                            data.surpressed_keys.insert(keysyms::KEY_Tab);
                                            let location =
                                                data.space.element_location(&element).unwrap();
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
            InputEvent::PointerMotion { event, .. } => {
                self.last_pointer_movement = Instant::now();
                let serial = SERIAL_COUNTER.next_serial();
                let delta = event.delta();

                let pointer = self.seat.get_pointer().unwrap();
                let under = self
                    .space
                    .element_under(self.pointer_location)
                    .map(|(w, pos)| (w.clone().into(), pos));

                /* Check if the pointer is locked or confined (pointer constraints protocol) */
                let mut pointer_locked = false;
                let mut pointer_confined = false;
                let mut confine_region = None;
                if let Some((surface, surface_loc)) = under
                    .as_ref()
                    .and_then(|(target, l): &(FocusTarget, Point<i32, Logical>)| Some((target.wl_surface()?, l)))
                {
                    with_pointer_constraint(&surface, &pointer, |constraint| match constraint {
                        Some(constraint) if constraint.is_active() => {
                            // Constraint does not apply if not within region
                            if !constraint.region().map_or(true, |x| {
                                x.contains(pointer.current_location().to_i32_round() - *surface_loc)
                            }) {
                                return;
                            }
                            match &*constraint {
                                PointerConstraint::Locked(_locked) => {
                                    pointer_locked = true;
                                }
                                PointerConstraint::Confined(confine) => {
                                    pointer_confined = true;
                                    confine_region = confine.region().cloned();
                                }
                            }
                        }
                        _ => {}
                    });
                }

                /* Relative motion is always applied */
                pointer.relative_motion(
                    self,
                    under,
                    &RelativeMotionEvent {
                        delta,
                        delta_unaccel: event.delta_unaccel(),
                        utime: event.time(),
                    },
                );

                // If pointer is locked, only emit relative motion
                if pointer_locked {
                    pointer.frame(self);
                    return;
                }

                self.pointer_location += delta;
                self.pointer_location = self.clamp_coords(self.pointer_location);
                let new_under = self
                    .space
                    .element_under(self.pointer_location)
                    .map(|(w, pos)| (w.clone().into(), pos));


                // TODO: If confined, don't move pointer if it would go outside surface or region
                pointer.motion(
                    self,
                    new_under.clone(),
                    &MotionEvent {
                        location: self.pointer_location,
                        serial,
                        time: event.time_msec(),
                    },
                );

                // If pointer is now in a constraint region, activate it
                if let Some((under, surface_location)) =
                    new_under.and_then(|(target, loc)| Some((target.wl_surface()?, loc)))
                {
                    with_pointer_constraint(&under, &pointer, |constraint| match constraint {
                        Some(constraint) if !constraint.is_active() => {
                            let point = self.pointer_location.to_i32_round() - surface_location;
                            if constraint.region().map_or(true, |region| region.contains(point)) {
                                constraint.activate();
                            }
                        }
                        _ => {}
                    });
                }
                pointer.frame(self);
            }
            InputEvent::PointerMotionAbsolute { event } => {
                self.last_pointer_movement = Instant::now();
                let serial = SERIAL_COUNTER.next_serial();
                if let Some(output) = self.output.as_ref() {
                    let output_size = output
                        .current_mode()
                        .unwrap()
                        .size
                        .to_f64()
                        .to_logical(output.current_scale().fractional_scale())
                        .to_i32_round();
                    self.pointer_location = (
                        event.absolute_x_transformed(output_size.w),
                        event.absolute_y_transformed(output_size.h),
                    )
                        .into();

                    let pointer = self.seat.get_pointer().unwrap();
                    let under = self
                        .space
                        .element_under(self.pointer_location)
                        .map(|(w, pos)| (w.clone().into(), pos));
                    pointer.motion(
                        self,
                        under.clone(),
                        &MotionEvent {
                            location: self.pointer_location,
                            serial,
                            time: event.time_msec(),
                        },
                    );
                    pointer.frame(self);
                }
            }
            InputEvent::PointerButton { event, .. } => {
                self.last_pointer_movement = Instant::now();
                let serial = SERIAL_COUNTER.next_serial();
                let button = event.button_code();

                let state = ButtonState::from(event.state());
                if ButtonState::Pressed == state {
                    self.update_keyboard_focus(serial);
                };
                self.seat.get_pointer().unwrap().button(
                    self,
                    &ButtonEvent {
                        button,
                        state: state.try_into().unwrap(),
                        serial,
                        time: event.time_msec(),
                    },
                );
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

                {
                    let mut frame = AxisFrame::new(event.time_msec()).source(event.source());
                    if horizontal_amount != 0.0 {
                        frame = frame.value(Axis::Horizontal, horizontal_amount);
                        if let Some(discrete) = horizontal_amount_discrete {
                            frame = frame.v120(Axis::Horizontal, discrete as i32);
                        }
                    } else if event.source() == AxisSource::Finger {
                        frame = frame.stop(Axis::Horizontal);
                    }
                    if vertical_amount != 0.0 {
                        frame = frame.value(Axis::Vertical, vertical_amount);
                        if let Some(discrete) = vertical_amount_discrete {
                            frame = frame.v120(Axis::Vertical, discrete as i32);
                        }
                    } else if event.source() == AxisSource::Finger {
                        frame = frame.stop(Axis::Vertical);
                    }
                    let pointer = self.seat.get_pointer().unwrap();
                    pointer.axis(self, frame);
                    pointer.frame(self);
                }
            }
            _ => {}
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
