use super::{State, focus::FocusTarget};
use smithay::backend::input::InputTime;
use smithay::backend::input::Keycode;
use smithay::backend::libinput::LibinputInputBackend;
use smithay::input::keyboard::Keysym;
use smithay::input::tablet::tool::{
    AxisFrame as ToolAxisFrame, ButtonEvent as ToolButtonEvent, DownEvent as ToolDownEvent,
    MotionEvent as ToolMotionEvent, ProximityInEvent, ProximityOutEvent, TabletToolHandle,
    UpEvent as ToolUpEvent,
};
use smithay::input::tablet::{TabletDescriptor, TabletSeatTrait};
use smithay::reexports::input::event::pointer::PointerEventTrait;
use smithay::wayland::seat::WaylandFocus;
use smithay::{
    backend::input::{
        AbsolutePositionEvent, Axis, AxisSource, ButtonState, Device, DeviceCapability, Event,
        InputEvent, KeyState, KeyboardKeyEvent, PointerAxisEvent, PointerButtonEvent,
        PointerMotionEvent, ProximityState, TabletToolButtonEvent, TabletToolCapabilities,
        TabletToolDescriptor, TabletToolEvent, TabletToolProximityEvent, TabletToolTipEvent,
        TabletToolTipState, TouchEvent, TouchSlot,
    },
    desktop::WindowSurfaceType,
    input::{
        keyboard::{FilterResult, keysyms},
        pointer::{AxisFrame, ButtonEvent, CursorImageStatus, MotionEvent, RelativeMotionEvent},
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
use std::collections::{HashMap, HashSet};
use std::{os::unix::io::OwnedFd, path::Path, time::Instant};

// From linux/input-event-codes.h
const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;
const BTN_STYLUS: u32 = 0x14b;
/// Stands for the tip in [`TabletInput::pointer_buttons`]: no physical button has this code.
const TIP: u32 = u32::MAX;

/// The tablet tool axes that changed in a given event (`None` = unchanged).
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
    fn frame(&self) -> ToolAxisFrame {
        ToolAxisFrame {
            pressure: self.pressure,
            distance: self.distance,
            tilt: self.tilt,
            rotation: self.rotation,
            slider: self.slider,
            wheel: self.wheel,
        }
    }

    /// Every tablet tool event (axis, tip and proximity alike) carries the axes that changed
    /// with it; dropping them on tip/proximity events would leave the client with a stale value.
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

/// How a tool's tip contact or one of its buttons is being delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Delivery {
    /// As `zwp_tablet_tool_v2` events to a tablet-aware client.
    Native,
    /// As this `wl_pointer` button, for a client that doesn't speak tablet-v2.
    Emulated(u32),
}

/// Per-tool state. Each physical tool (and each end of a stylus: pen and eraser are separate
/// libinput tools) is tracked on its own, so one tool never acts on another's contact,
/// buttons or position -- and none of it is tied to where the mouse is.
#[derive(Debug)]
struct ToolState {
    tablet: TabletDescriptor,
    /// Where the pen is, in compositor coordinates (clamped to the output).
    location: Point<f64, Logical>,
    /// The surface receiving this tool's tablet-v2 events, with its origin; `None` while the
    /// tool is over nothing tablet-aware (or emulating a pointer).
    focus: Option<(WlSurface, Point<f64, Logical>)>,
    /// The tip is down, and how that contact is delivered. A native contact holds an implicit
    /// grab on `focus`: the stroke keeps going to that surface until the tip comes up.
    contact: Option<Delivery>,
    /// Physical button code -> how its press was delivered, so its release goes the same way.
    buttons: HashMap<u32, Delivery>,
}

/// A tool on a particular tablet. Tools without a serial (serial 0: most non-Wacom and virtual
/// tablets) are only told apart by their tablet; see [`smithay_tool`].
type ToolKey = (TabletDescriptor, TabletToolDescriptor);

/// The descriptor smithay knows `tool` on `tablet` by. Smithay keys its tools (their focus,
/// tip and buttons) by descriptor alone, so two serial-less pens on two tablets would be one
/// tool to it; such a tool gets a serial derived from its tablet instead.
fn smithay_tool(tablet: &TabletDescriptor, tool: &TabletToolDescriptor) -> TabletToolDescriptor {
    if tool.hardware_serial != 0 {
        return tool.clone();
    }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tablet.hash(&mut hasher);
    TabletToolDescriptor {
        hardware_serial: hasher.finish() | 1,
        ..tool.clone()
    }
}

/// Who holds a `wl_pointer` button down.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ButtonOwner {
    /// A pen's tip ([`TIP`]) or one of its physical stylus buttons, emulating the button.
    Tool(ToolKey, u32),
    /// The mouse.
    Mouse,
}

/// The seat's tablet input state; see [`ToolState`].
#[derive(Debug, Default)]
pub(crate) struct TabletInput {
    tools: HashMap<ToolKey, ToolState>,
    /// Pointer button -> who holds it: the mouse and any pen emulating it. The pointer button
    /// is pressed when its first owner presses and released only when its last owner lets go,
    /// so two stylus buttons mapped to middle click, two pens, or a pen and the mouse can't
    /// release each other's press.
    pointer_buttons: HashMap<u32, HashSet<ButtonOwner>>,
    /// The pointer's own cursor image while a tablet tool's image is shown instead; restored
    /// by the next mouse motion (and when the pen goes back to emulating the pointer).
    saved_pointer_image: Option<CursorImageStatus>,
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
            InputTime::from_millis(event_time_msec),
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
        let should_motion = self.constraint_allows_move(under, target_position);
        let new_under = self
            .space
            .element_under(target_position)
            .map(|(w, pos)| (w.clone().into(), pos.to_f64()));
        // If pointer is now in a constraint region, activate it
        self.maybe_activate_pointer_constraint(&new_under, target_position);
        should_motion
    }

    /// Whether an active pointer constraint on `under` (the surface under the pointer) lets
    /// the pointer move to `target_position`. Unlike [`Self::can_pointer_move`] this activates
    /// no constraint.
    fn constraint_allows_move(
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

        should_motion
    }

    /// The pending edge-triggered refocus (see `pointer_motion`): force smithay's pointer
    /// focus to None so the next motion re-emits `wl_pointer.enter`.
    fn refocus_pointer_if_pending(
        &mut self,
        under: &Option<(FocusTarget, Point<f64, Logical>)>,
        time: u32,
    ) {
        let pointer = self.seat.get_pointer().unwrap();
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
                        time: InputTime::from_millis(time),
                    },
                );
            }
        }
    }

    pub fn pointer_motion(
        &mut self,
        event_time_msec: u32,
        event_time_usec: u64,
        delta: Point<f64, Logical>,
        delta_unaccelerated: Point<f64, Logical>,
    ) {
        self.last_pointer_movement = Instant::now();
        // The mouse moves: the cursor is the pointer's again, not a pen's.
        self.restore_pointer_cursor_image();
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
        self.refocus_pointer_if_pending(&under, event_time_msec);

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
                    time: InputTime::from_millis(event_time_msec),
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
                time: InputTime::from_micros(event_time_usec),
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

    /// A mouse button. A button a pen already holds (an emulated drag) stays down until both
    /// let go; see [`TabletInput::pointer_buttons`].
    pub fn pointer_button(&mut self, event_time_msec: u32, button_code: u32, state: ButtonState) {
        self.owned_button(event_time_msec, ButtonOwner::Mouse, button_code, state);
    }

    /// Records `owner`'s press or release of pointer `button`, and sends it to the pointer only
    /// on the first press and the last release.
    fn owned_button(&mut self, time: u32, owner: ButtonOwner, button: u32, state: ButtonState) {
        match state {
            ButtonState::Pressed => {
                let owners = self.tablet.pointer_buttons.entry(button).or_default();
                let first = owners.is_empty();
                owners.insert(owner);
                if first {
                    self.send_pointer_button(time, button, state);
                }
            }
            ButtonState::Released => match self.tablet.pointer_buttons.get_mut(&button) {
                Some(owners) => {
                    if owners.remove(&owner) && owners.is_empty() {
                        self.tablet.pointer_buttons.remove(&button);
                        self.send_pointer_button(time, button, state);
                    }
                }
                // Not pressed through here (e.g. pressed before this compositor saw it):
                // forward the mouse's release rather than leave the client's button stuck.
                None if owner == ButtonOwner::Mouse => {
                    self.send_pointer_button(time, button, state)
                }
                None => {}
            },
        }
    }

    fn send_pointer_button(&mut self, event_time_msec: u32, button_code: u32, state: ButtonState) {
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
                time: InputTime::from_millis(event_time_msec),
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
        let mut frame = AxisFrame::new(InputTime::from_millis(event_time_msec)).source(source);
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
                time: InputTime::from_millis(event_time_msec),
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
                time: InputTime::from_millis(event_time_msec),
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
                time: InputTime::from_millis(event_time_msec),
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

    /// Adds the device at `path`, replacing the handle an earlier announcement of the same path
    /// left: libinput doesn't de-duplicate paths, and a second handle doubles every event. A
    /// reconnected device can reuse its path while libinput still holds the old handle.
    pub fn add_input_device(&mut self, path: &str) {
        if let Some(old) = self.input_devices.remove(path) {
            // A no-op if libinput already removed it after ENODEV.
            self.input_context.path_remove_device(old);
        }
        match self.input_context.path_add_device(path) {
            Some(device) => {
                self.input_devices.insert(path.to_owned(), device);
            }
            None => tracing::warn!(path, "libinput could not add the input device."),
        }
    }

    pub(crate) fn input_device_removed(&mut self, device: &smithay::reexports::input::Device) {
        self.input_devices.retain(|_, added| added != device);
    }

    pub fn process_input_event(&mut self, event: InputEvent<LibinputInputBackend>) {
        match event {
            InputEvent::DeviceAdded { device } => {
                if Device::has_capability(&device, DeviceCapability::TabletTool) {
                    self.tablet_added(&TabletDescriptor::from(&device));
                }
            }
            InputEvent::DeviceRemoved { device } => {
                if Device::has_capability(&device, DeviceCapability::TabletTool) {
                    self.tablet_removed(&TabletDescriptor::from(&device));
                }
                self.input_device_removed(&device);
            }
            InputEvent::TabletToolProximity { event, .. } => {
                if let Some(location) = self.touch_location_transformed(&event) {
                    self.tablet_tool_proximity(
                        smithay::backend::input::Event::time(&event).millis(),
                        &TabletDescriptor::from(&event.device()),
                        &event.tool(),
                        location,
                        TabletToolAxes::from_event(&event),
                        event.state() == ProximityState::In,
                    );
                }
            }
            InputEvent::TabletToolAxis { event, .. } => {
                if let Some(location) = self.touch_location_transformed(&event) {
                    self.tablet_tool_axis(
                        smithay::backend::input::Event::time(&event).millis(),
                        &TabletDescriptor::from(&event.device()),
                        &event.tool(),
                        location,
                        TabletToolAxes::from_event(&event),
                    );
                }
            }
            InputEvent::TabletToolTip { event, .. } => {
                // A tip event carries the final axis values with it (libinput sends no separate
                // axis event for them), so pressure returning to 0 on tip-up is only seen here.
                if let Some(location) = self.touch_location_transformed(&event) {
                    self.tablet_tool_tip(
                        smithay::backend::input::Event::time(&event).millis(),
                        &TabletDescriptor::from(&event.device()),
                        &event.tool(),
                        location,
                        TabletToolAxes::from_event(&event),
                        TabletToolTipEvent::tip_state(&event) == TabletToolTipState::Down,
                    );
                }
            }
            InputEvent::TabletToolButton { event, .. } => {
                self.tablet_tool_button(
                    smithay::backend::input::Event::time(&event).millis(),
                    &TabletDescriptor::from(&event.device()),
                    &event.tool(),
                    TabletToolButtonEvent::button(&event),
                    TabletToolButtonEvent::button_state(&event),
                );
            }
            InputEvent::Keyboard { event, .. } => {
                self.keyboard_input(
                    smithay::backend::input::Event::time(&event).millis(),
                    event.key_code(),
                    event.state(),
                );
            }
            InputEvent::PointerMotion { event, .. } => {
                self.pointer_motion(
                    smithay::backend::input::Event::time(&event).millis(),
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

                    self.pointer_motion_absolute(
                        smithay::backend::input::Event::time(&event).millis(),
                        (new_x, new_y).into(),
                    );
                }
            }
            InputEvent::PointerButton { event, .. } => {
                self.pointer_button(
                    smithay::backend::input::Event::time(&event).millis(),
                    event.button(),
                    event.state(),
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

                self.pointer_axis(
                    smithay::backend::input::Event::time(&event).millis(),
                    event.source(),
                    horizontal_amount,
                    vertical_amount,
                    horizontal_amount_discrete,
                    vertical_amount_discrete,
                );
            }
            InputEvent::TouchDown { event, .. } => {
                if let Some(location) = self.touch_location_transformed(&event) {
                    self.touch_down(
                        smithay::backend::input::Event::time(&event).millis(),
                        event.slot(),
                        location,
                    );
                }
            }
            InputEvent::TouchUp { event, .. } => {
                self.touch_up(
                    smithay::backend::input::Event::time(&event).millis(),
                    event.slot(),
                );
            }
            InputEvent::TouchMotion { event, .. } => {
                if let Some(location) = self.touch_location_transformed(&event) {
                    self.touch_motion(
                        smithay::backend::input::Event::time(&event).millis(),
                        event.slot(),
                        location,
                    );
                }
            }
            InputEvent::TouchCancel { .. } => {
                self.touch_cancel();
            }
            InputEvent::TouchFrame { .. } => {
                self.touch_frame();
            }
            _ => {}
        }
    }

    pub fn tablet_added(&mut self, tablet: &TabletDescriptor) {
        tracing::info!(name = %tablet.name, "Tablet added");
        self.seat.tablet_seat().add_wp_tablet(&self.dh, tablet);
    }

    /// Smithay's handle for `tool` on `tablet`, announced to the clients when first used.
    fn tool_handle(
        &mut self,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
    ) -> TabletToolHandle<Self> {
        let descriptor = smithay_tool(tablet, tool);
        let tablet_seat = self.seat.tablet_seat();
        match tablet_seat.get_tool(&descriptor) {
            Some(handle) => handle,
            None => {
                let dh = self.dh.clone();
                tablet_seat.add_wp_tool(self, &dh, &descriptor)
            }
        }
    }

    /// A tablet went away: end every contact and release every button its tools hold, so
    /// nothing stays pressed, and forget the tools.
    pub fn tablet_removed(&mut self, tablet: &TabletDescriptor) {
        tracing::info!(name = %tablet.name, "Tablet removed");
        let tools: Vec<TabletToolDescriptor> = self
            .tablet
            .tools
            .keys()
            .filter(|(t, _)| t == tablet)
            .map(|(_, tool)| tool.clone())
            .collect();
        let time = self.clock.now().as_millis();
        for tool in &tools {
            self.tablet_tool_leave(time, tablet, tool);
            self.tablet.tools.remove(&(tablet.clone(), tool.clone()));
        }
        let tablet_seat = self.seat.tablet_seat();
        tablet_seat.remove_tablet(tablet);
        for tool in &tools {
            // Keep smithay's tool while the same descriptor is still live on another tablet.
            let descriptor = smithay_tool(tablet, tool);
            if !self
                .tablet
                .tools
                .keys()
                .any(|(t, other)| smithay_tool(t, other) == descriptor)
            {
                tablet_seat.remove_tool(&descriptor);
            }
        }
        // With no tablet left no tool can be in use.
        if tablet_seat.count_tablets() == 0 {
            tablet_seat.clear_tools();
        }
    }

    /// Whether the client owning `surface` has bound the tablet protocol (`zwp_tablet_seat_v2`).
    ///
    /// Tablet support is detected per client connection: a nested compositor (gamescope, a
    /// nested sway, Xwayland) is one client here, so its own clients only get tablet events if
    /// it binds tablet-v2 and forwards them. Clients that don't (gamescope, most games, older
    /// toolkits) would see the cursor follow the pen but never a click; for those the pen
    /// emulates a pointer instead.
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

    /// The surface actually under `location` -- a popup or subsurface where there is one, not
    /// just its window's root surface -- with its origin in compositor coordinates.
    fn surface_under(
        &self,
        location: Point<f64, Logical>,
    ) -> Option<(WlSurface, Point<f64, Logical>)> {
        let (window, window_loc) = self.space.element_under(location)?;
        window
            .surface_under(location - window_loc.to_f64(), WindowSurfaceType::ALL)
            .map(|(surface, loc)| (surface, (loc + window_loc).to_f64()))
    }

    /// Moves the cursor composited into the stream so it keeps following the pen, *without*
    /// a wl_pointer event: the tablet-aware client gets zwp_tablet_tool_v2.motion instead, and
    /// duplicating it as pointer motion makes drawing applications see every stroke twice.
    ///
    /// A pointer lock (or confinement) holds the cursor: moving it here would also move the
    /// pointer the next mouse motion starts from, letting the mouse escape the lock.
    fn tablet_cursor_only_motion(&mut self, location: Point<f64, Logical>) {
        let current: Option<(FocusTarget, Point<f64, Logical>)> = self
            .space
            .element_under(self.pointer_location)
            .map(|(w, pos)| (w.clone().into(), pos.to_f64()));
        if !self.constraint_allows_move(&current, location) {
            return;
        }
        self.last_pointer_movement = Instant::now();
        self.set_pointer_location(location);
    }

    /// A tablet tool set its cursor image (smithay's `TabletSeatHandler`): show it in place
    /// of the pointer's, which is kept to be restored.
    pub(crate) fn set_tablet_cursor_image(&mut self, image: CursorImageStatus) {
        let previous = std::mem::replace(&mut self.cursor_state, image);
        self.tablet.saved_pointer_image.get_or_insert(previous);
    }

    /// The pointer's client set its cursor image. While a pen's image is shown, keep it for
    /// when the cursor is the pointer's again.
    pub(crate) fn set_pointer_cursor_image(&mut self, image: CursorImageStatus) {
        match &mut self.tablet.saved_pointer_image {
            Some(saved) => *saved = image,
            None => self.cursor_state = image,
        }
    }

    fn restore_pointer_cursor_image(&mut self) {
        if let Some(saved) = self.tablet.saved_pointer_image.take() {
            self.cursor_state = saved;
        }
    }

    /// Whether `key`'s tool holds any emulated pointer button (its tip or a stylus button).
    fn tool_emulates(&self, key: &ToolKey) -> bool {
        self.tablet.pointer_buttons.values().any(|owners| {
            owners
                .iter()
                .any(|owner| matches!(owner, ButtonOwner::Tool(k, _) if k == key))
        })
    }

    /// Presses emulated pointer `button` on behalf of (`key`, `physical`); the pointer sees
    /// the press only if nobody (no pen, not the mouse) held it yet.
    fn emulated_press(&mut self, time: u32, key: &ToolKey, physical: u32, button: u32) {
        let owner = ButtonOwner::Tool(key.clone(), physical);
        self.owned_button(time, owner, button, ButtonState::Pressed);
    }

    /// Releases (`key`, `physical`)'s hold on emulated pointer `button`; the pointer sees the
    /// release only when no other owner still holds it.
    fn emulated_release(&mut self, time: u32, key: &ToolKey, physical: u32, button: u32) {
        let owner = ButtonOwner::Tool(key.clone(), physical);
        self.owned_button(time, owner, button, ButtonState::Released);
    }

    /// Moves `tool` to `location` and delivers the move: to the surface holding its native
    /// contact (implicit grab), as pointer motion while it emulates a pointer button, else to
    /// whatever is under it. Every axis that changed reaches the client with this move, also
    /// when it changes the tool's focus (smithay would hold them back until the next motion).
    fn tablet_tool_move(
        &mut self,
        time: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
        location: Point<f64, Logical>,
        axes: TabletToolAxes,
    ) {
        let location = self.clamp_coords(location);
        let tool_handle = self.tool_handle(tablet, tool);
        let key: ToolKey = (tablet.clone(), tool.clone());
        let state = self.tablet.tools.entry(key.clone()).or_insert(ToolState {
            tablet: tablet.clone(),
            location,
            focus: None,
            contact: None,
            buttons: HashMap::new(),
        });
        state.tablet = tablet.clone();
        state.location = location;
        let prev_focus = state.focus.clone();
        let contact = state.contact;

        // Where the tablet-v2 events go now.
        let native_focus = match (contact, &prev_focus) {
            // Keep a stroke on the surface it started on, even across windows, as long as
            // that surface is alive (smithay's implicit grab does the same).
            (Some(Delivery::Native), Some((surface, origin))) if surface.is_alive() => {
                Some((surface.clone(), *origin))
            }
            // An emulated press is a pointer drag: keep delivering pointer motion, even over a
            // tablet-aware window (smithay's click grab keeps it on the pressed window).
            _ if self.tool_emulates(&key) => None,
            _ => self
                .surface_under(location)
                .filter(|(surface, _)| self.surface_accepts_tablet(surface)),
        };

        let Some(tablet_handle) = self.seat.tablet_seat().get_tablet(tablet) else {
            return;
        };
        let time = InputTime::from_millis(time);
        let serial = SERIAL_COUNTER.next_serial();
        if native_focus.is_some() {
            self.tablet_cursor_only_motion(location);
        }
        // Smithay moves the tool's tablet-v2 focus: proximity out of the surface it leaves and
        // in to the one it enters (with the tip and held buttons), nothing while `None`.
        if tool_handle.current_tablet().is_none() {
            tool_handle.proximity_in(
                self,
                native_focus.clone(),
                tablet_handle,
                &ProximityInEvent {
                    location,
                    axis: Some(axes.frame()),
                    serial,
                    time,
                },
            );
        } else {
            tool_handle.motion(
                self,
                native_focus.clone(),
                &ToolMotionEvent {
                    location,
                    serial,
                    time,
                },
            );
            // After the motion, so every axis that changed reaches the surface the tool is
            // over now, also when the motion changed it.
            tool_handle.axis(self, axes.frame());
        }
        tool_handle.frame(self, time);

        let lost_native_contact = native_focus.is_none() && contact == Some(Delivery::Native);
        if let Some(state) = self.tablet.tools.get_mut(&key) {
            state.focus = native_focus.clone();
            if lost_native_contact {
                state.contact = None;
            }
        }
        if lost_native_contact {
            // The contact's surface is gone and nothing tablet-aware is under the pen: the
            // stroke is over, so a later surface must not get its tip down either.
            tool_handle.up(self, &ToolUpEvent { serial, time });
        }
        if native_focus.is_none() {
            // The pen drives the pointer now, so the cursor is the pointer's.
            self.restore_pointer_cursor_image();
            self.tablet_pointer_warp(time.millis(), location);
        }
    }

    /// Moves the pointer to the pen at `location`, resolving the pointer focus there (the
    /// relative-motion path resolves it at the previous position, which would hand the pen's
    /// next click to the window it just left).
    fn tablet_pointer_warp(&mut self, time: u32, location: Point<f64, Logical>) {
        self.last_pointer_movement = Instant::now();
        let pointer = self.seat.get_pointer().unwrap();
        let current: Option<(FocusTarget, Point<f64, Logical>)> = self
            .space
            .element_under(self.pointer_location)
            .map(|(w, pos)| (w.clone().into(), pos.to_f64()));
        // A locked (or confining) pointer constraint holds the pointer as it does for the
        // mouse: the pen's emulated input then goes to the constrained surface.
        if !self.can_pointer_move(&current, location) {
            return;
        }
        let under: Option<(FocusTarget, Point<f64, Logical>)> = self
            .space
            .element_under(location)
            .map(|(w, pos)| (w.clone().into(), pos.to_f64()));
        // Same late-`wl_pointer` refocus as mouse motion.
        self.refocus_pointer_if_pending(&under, time);
        self.set_pointer_location(location);
        pointer.motion(
            self,
            under,
            &MotionEvent {
                location,
                serial: SERIAL_COUNTER.next_serial(),
                time: InputTime::from_millis(time),
            },
        );
        pointer.frame(self);
    }

    pub fn tablet_tool_proximity(
        &mut self,
        time: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
        location: Point<f64, Logical>,
        axes: TabletToolAxes,
        is_in: bool,
    ) {
        if is_in {
            self.tablet_tool_move(time, tablet, tool, location, axes);
        } else {
            self.tablet_tool_leave(time, tablet, tool);
        }
    }

    /// The tool left proximity (or its tablet went away): lift its tip, release its buttons
    /// and end its tablet-v2 focus. Only this tool's state is touched.
    fn tablet_tool_leave(
        &mut self,
        time: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
    ) {
        let key: ToolKey = (tablet.clone(), tool.clone());
        let Some(state) = self.tablet.tools.get_mut(&key) else {
            return;
        };
        state.focus = None;
        state.contact = None;
        let buttons: Vec<(u32, Delivery)> = state.buttons.drain().collect();
        let tool_handle = self
            .seat
            .tablet_seat()
            .get_tool(&smithay_tool(tablet, tool));
        for (physical, delivery) in buttons {
            if let Delivery::Emulated(button) = delivery {
                self.emulated_release(time, &key, physical, button);
            }
        }
        self.emulated_release(time, &key, TIP, BTN_LEFT);
        self.restore_pointer_cursor_image();
        if let Some(tool_handle) = tool_handle.filter(|t| t.current_tablet().is_some()) {
            // Sends zero pressure and `up` first if the tip was down, then proximity out.
            let time = InputTime::from_millis(time);
            let serial = SERIAL_COUNTER.next_serial();
            tool_handle.proximity_out(self, &ProximityOutEvent { serial, time });
            tool_handle.frame(self, time);
        }
    }

    pub fn tablet_tool_axis(
        &mut self,
        time: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
        location: Point<f64, Logical>,
        axes: TabletToolAxes,
    ) {
        self.tablet_tool_move(time, tablet, tool, location, axes);
    }

    pub fn tablet_tool_tip(
        &mut self,
        time: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
        location: Point<f64, Logical>,
        mut axes: TabletToolAxes,
        is_down: bool,
    ) {
        if !is_down && tool.capabilities.contains(TabletToolCapabilities::PRESSURE) {
            // No pressure may linger once the tip is up, whatever the device reported (none,
            // or a small residue).
            axes.pressure = Some(0.0);
        }
        // The tip event doubles as an axis event: position and the axes the tip went down /
        // came up with reach the client before the down / up.
        self.tablet_tool_move(time, tablet, tool, location, axes);
        let key: ToolKey = (tablet.clone(), tool.clone());
        let Some(state) = self.tablet.tools.get(&key) else {
            return;
        };
        let native = state.focus.is_some();
        let contact = state.contact;

        if is_down {
            if contact.is_some() {
                return;
            }
            let serial = SERIAL_COUNTER.next_serial();
            self.update_keyboard_focus(serial);
            if native {
                let tool_handle = self.tool_handle(tablet, tool);
                let time = InputTime::from_millis(time);
                tool_handle.down(self, &ToolDownEvent { serial, time });
                tool_handle.frame(self, time);
                self.tablet.tools.get_mut(&key).unwrap().contact = Some(Delivery::Native);
            } else {
                self.emulated_press(time, &key, TIP, BTN_LEFT);
                self.tablet.tools.get_mut(&key).unwrap().contact =
                    Some(Delivery::Emulated(BTN_LEFT));
            }
        } else {
            match contact {
                Some(Delivery::Native) => {
                    let tool_handle = self.tool_handle(tablet, tool);
                    let time = InputTime::from_millis(time);
                    let serial = SERIAL_COUNTER.next_serial();
                    tool_handle.up(self, &ToolUpEvent { serial, time });
                    tool_handle.frame(self, time);
                }
                Some(Delivery::Emulated(button)) => self.emulated_release(time, &key, TIP, button),
                None => {}
            }
            if let Some(state) = self.tablet.tools.get_mut(&key) {
                state.contact = None;
            }
            // Contact ended: settle the focus on whatever is under the pen now.
            if let Some(state) = self.tablet.tools.get(&key) {
                let location = state.location;
                self.tablet_tool_move(time, tablet, tool, location, Default::default());
            }
        }
    }

    pub fn tablet_tool_button(
        &mut self,
        time: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
        button: u32,
        state: ButtonState,
    ) {
        let key: ToolKey = (tablet.clone(), tool.clone());
        let Some(tool_state) = self.tablet.tools.get(&key) else {
            return;
        };
        match state {
            ButtonState::Pressed => {
                if tool_state.buttons.contains_key(&button) {
                    return;
                }
                // Routed by where this pen is, never by the mouse.
                let delivery = if tool_state.focus.is_some() {
                    self.tool_button(time, tablet, tool, button, state);
                    Delivery::Native
                } else {
                    // BTN_STYLUS -> right click, BTN_STYLUS2 / BTN_STYLUS3 -> middle click.
                    let emulated = if button == BTN_STYLUS {
                        BTN_RIGHT
                    } else {
                        BTN_MIDDLE
                    };
                    // The mouse may have moved since the pen did: bring the pointer back to the
                    // pen so the click lands in the application under the pen.
                    let location = tool_state.location;
                    if self.pointer_location != location {
                        self.tablet_pointer_warp(time, location);
                    }
                    self.emulated_press(time, &key, button, emulated);
                    Delivery::Emulated(emulated)
                };
                if let Some(tool_state) = self.tablet.tools.get_mut(&key) {
                    tool_state.buttons.insert(button, delivery);
                }
            }
            ButtonState::Released => {
                let delivery = self
                    .tablet
                    .tools
                    .get_mut(&key)
                    .and_then(|t| t.buttons.remove(&button));
                match delivery {
                    Some(Delivery::Native) => self.tool_button(time, tablet, tool, button, state),
                    Some(Delivery::Emulated(emulated)) => {
                        self.emulated_release(time, &key, button, emulated)
                    }
                    None => {}
                }
            }
        }
    }

    /// A stylus button on the tool's tablet-v2 focus.
    fn tool_button(
        &mut self,
        time: u32,
        tablet: &TabletDescriptor,
        tool: &TabletToolDescriptor,
        button: u32,
        state: ButtonState,
    ) {
        let tool_handle = self.tool_handle(tablet, tool);
        let time = InputTime::from_millis(time);
        let event = ToolButtonEvent {
            serial: SERIAL_COUNTER.next_serial(),
            button,
            state,
            time,
        };
        tool_handle.button(self, &event);
        tool_handle.frame(self, time);
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
impl State {
    /// Whether any pointer button is still held on behalf of a pen.
    pub(crate) fn pen_holds_pointer_buttons(&self) -> bool {
        self.tablet
            .pointer_buttons
            .values()
            .any(|owners| owners.iter().any(|o| !matches!(o, ButtonOwner::Mouse)))
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
                false,
                crate::utils::vulkan_nv12::ConversionOptions::default().hdr_peak_nits,
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
                serial_number: String::new(),
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
                serial_number: String::new(),
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
