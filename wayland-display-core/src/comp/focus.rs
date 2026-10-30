use crate::comp::State;
use smithay::input::dnd::{DndFocus, Source};
use smithay::reexports::wayland_server::DisplayHandle;
use smithay::utils::{Logical, Point};
use smithay::{
    backend::input::{InputTime, KeyState},
    desktop::{PopupKind, Window, WindowSurface},
    input::{
        Seat,
        keyboard::{KeyboardTarget, KeysymHandle, ModifiersState},
        pointer::{
            AxisFrame, ButtonEvent, GestureHoldBeginEvent, GestureHoldEndEvent,
            GesturePinchBeginEvent, GesturePinchEndEvent, GesturePinchUpdateEvent,
            GestureSwipeBeginEvent, GestureSwipeEndEvent, GestureSwipeUpdateEvent, MotionEvent,
            PointerTarget, RelativeMotionEvent,
        },
        touch::{DownEvent, FrameMarker, OrientationEvent, ShapeEvent, TouchTarget, UpEvent},
    },
    reexports::wayland_server::{backend::ObjectId, protocol::wl_surface::WlSurface},
    utils::{IsAlive, Serial},
    wayland::seat::WaylandFocus,
};
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub enum FocusTarget {
    Wayland(Window),
    Popup(PopupKind),
}

impl IsAlive for FocusTarget {
    fn alive(&self) -> bool {
        match self {
            FocusTarget::Wayland(w) => w.alive(),
            FocusTarget::Popup(p) => p.alive(),
        }
    }
}

impl From<Window> for FocusTarget {
    fn from(w: Window) -> Self {
        FocusTarget::Wayland(w)
    }
}

impl From<PopupKind> for FocusTarget {
    fn from(p: PopupKind) -> Self {
        FocusTarget::Popup(p)
    }
}

impl KeyboardTarget<State> for FocusTarget {
    fn enter(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        keys: Vec<KeysymHandle<'_>>,
        serial: Serial,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    KeyboardTarget::enter(w.wl_surface(), seat, data, keys, serial)
                }
            },
            FocusTarget::Popup(p) => {
                KeyboardTarget::enter(p.wl_surface(), seat, data, keys, serial)
            }
        }
    }

    fn leave(&self, seat: &Seat<State>, data: &mut State, serial: Serial) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    KeyboardTarget::leave(w.wl_surface(), seat, data, serial)
                }
            },
            FocusTarget::Popup(p) => KeyboardTarget::leave(p.wl_surface(), seat, data, serial),
        }
    }

    fn key(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        key: KeysymHandle<'_>,
        state: KeyState,
        serial: Serial,
        time: InputTime,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    KeyboardTarget::key(w.wl_surface(), seat, data, key, state, serial, time)
                }
            },
            FocusTarget::Popup(p) => {
                KeyboardTarget::key(p.wl_surface(), seat, data, key, state, serial, time)
            }
        }
    }

    fn modifiers(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        modifiers: ModifiersState,
        serial: Serial,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    KeyboardTarget::modifiers(w.wl_surface(), seat, data, modifiers, serial)
                }
            },
            FocusTarget::Popup(p) => p.wl_surface().modifiers(seat, data, modifiers, serial),
        }
    }
}

impl PointerTarget<State> for FocusTarget {
    fn enter(&self, seat: &Seat<State>, data: &mut State, event: &MotionEvent) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::enter(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => PointerTarget::enter(p.wl_surface(), seat, data, event),
        }
    }

    fn motion(&self, seat: &Seat<State>, data: &mut State, event: &MotionEvent) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::motion(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => PointerTarget::motion(p.wl_surface(), seat, data, event),
        }
    }

    fn relative_motion(&self, seat: &Seat<State>, data: &mut State, event: &RelativeMotionEvent) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::relative_motion(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => {
                PointerTarget::relative_motion(p.wl_surface(), seat, data, event)
            }
        }
    }

    fn button(&self, seat: &Seat<State>, data: &mut State, event: &ButtonEvent) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::button(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => PointerTarget::button(p.wl_surface(), seat, data, event),
        }
    }

    fn axis(&self, seat: &Seat<State>, data: &mut State, frame: AxisFrame) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => PointerTarget::axis(w.wl_surface(), seat, data, frame),
            },
            FocusTarget::Popup(p) => PointerTarget::axis(p.wl_surface(), seat, data, frame),
        }
    }

    fn frame(&self, seat: &Seat<State>, data: &mut State) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => PointerTarget::frame(w.wl_surface(), seat, data),
            },
            FocusTarget::Popup(p) => PointerTarget::frame(p.wl_surface(), seat, data),
        }
    }

    fn gesture_swipe_begin(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        event: &GestureSwipeBeginEvent,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::gesture_swipe_begin(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => {
                PointerTarget::gesture_swipe_begin(p.wl_surface(), seat, data, event)
            }
        }
    }

    fn gesture_swipe_update(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        event: &GestureSwipeUpdateEvent,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::gesture_swipe_update(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => {
                PointerTarget::gesture_swipe_update(p.wl_surface(), seat, data, event)
            }
        }
    }

    fn gesture_swipe_end(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        event: &GestureSwipeEndEvent,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::gesture_swipe_end(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => {
                PointerTarget::gesture_swipe_end(p.wl_surface(), seat, data, event)
            }
        }
    }

    fn gesture_pinch_begin(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        event: &GesturePinchBeginEvent,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::gesture_pinch_begin(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => {
                PointerTarget::gesture_pinch_begin(p.wl_surface(), seat, data, event)
            }
        }
    }

    fn gesture_pinch_update(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        event: &GesturePinchUpdateEvent,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::gesture_pinch_update(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => {
                PointerTarget::gesture_pinch_update(p.wl_surface(), seat, data, event)
            }
        }
    }

    fn gesture_pinch_end(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        event: &GesturePinchEndEvent,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::gesture_pinch_end(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => {
                PointerTarget::gesture_pinch_end(p.wl_surface(), seat, data, event)
            }
        }
    }

    fn gesture_hold_begin(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        event: &GestureHoldBeginEvent,
    ) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::gesture_hold_begin(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => {
                PointerTarget::gesture_hold_begin(p.wl_surface(), seat, data, event)
            }
        }
    }

    fn gesture_hold_end(&self, seat: &Seat<State>, data: &mut State, event: &GestureHoldEndEvent) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::gesture_hold_end(w.wl_surface(), seat, data, event)
                }
            },
            FocusTarget::Popup(p) => {
                PointerTarget::gesture_hold_end(p.wl_surface(), seat, data, event)
            }
        }
    }

    fn leave(&self, seat: &Seat<State>, data: &mut State, serial: Serial, time: InputTime) {
        match self {
            FocusTarget::Wayland(w) => match w.underlying_surface() {
                WindowSurface::Wayland(w) => {
                    PointerTarget::leave(w.wl_surface(), seat, data, serial, time)
                }
            },
            FocusTarget::Popup(p) => PointerTarget::leave(p.wl_surface(), seat, data, serial, time),
        }
    }
}

impl WaylandFocus for FocusTarget {
    fn wl_surface(&self) -> Option<Cow<'_, WlSurface>> {
        match self {
            FocusTarget::Wayland(w) => w.wl_surface(),
            FocusTarget::Popup(p) => Some(Cow::Borrowed(p.wl_surface())),
        }
    }

    fn same_client_as(&self, object_id: &ObjectId) -> bool {
        match self {
            FocusTarget::Wayland(w) => w.same_client_as(object_id),
            FocusTarget::Popup(p) => p.wl_surface().same_client_as(object_id),
        }
    }
}

impl TouchTarget<State> for FocusTarget {
    fn down(&self, seat: &Seat<State>, data: &mut State, event: &DownEvent) {
        if let Some(surface) = self.wl_surface() {
            TouchTarget::down(&*surface, seat, data, event)
        }
    }

    fn up(&self, seat: &Seat<State>, data: &mut State, event: &UpEvent) {
        if let Some(surface) = self.wl_surface() {
            TouchTarget::up(&*surface, seat, data, event)
        }
    }

    fn motion(
        &self,
        seat: &Seat<State>,
        data: &mut State,
        event: &smithay::input::touch::MotionEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            TouchTarget::motion(&*surface, seat, data, event)
        }
    }

    fn frame(&self, seat: &Seat<State>, data: &mut State, marker: FrameMarker) {
        if let Some(surface) = self.wl_surface() {
            TouchTarget::frame(&*surface, seat, data, marker)
        }
    }

    fn cancel(&self, seat: &Seat<State>, data: &mut State, marker: FrameMarker) {
        if let Some(surface) = self.wl_surface() {
            TouchTarget::cancel(&*surface, seat, data, marker)
        }
    }

    fn shape(&self, seat: &Seat<State>, data: &mut State, event: &ShapeEvent) {
        if let Some(surface) = self.wl_surface() {
            TouchTarget::shape(&*surface, seat, data, event)
        }
    }

    fn orientation(&self, seat: &Seat<State>, data: &mut State, event: &OrientationEvent) {
        if let Some(surface) = self.wl_surface() {
            TouchTarget::orientation(&*surface, seat, data, event)
        }
    }

    fn last_frame(&self, seat: &Seat<State>, data: &mut State) -> Option<FrameMarker> {
        let surface = self.wl_surface()?;
        TouchTarget::last_frame(&*surface, seat, data)
    }
}

impl DndFocus<State> for FocusTarget {
    type OfferData<S>
        = <WlSurface as DndFocus<State>>::OfferData<S>
    where
        S: Source;

    fn enter<S: Source>(
        &self,
        data: &mut State,
        dh: &DisplayHandle,
        source: Arc<S>,
        seat: &Seat<State>,
        location: Point<f64, Logical>,
        serial: &Serial,
    ) -> Option<Self::OfferData<S>> {
        let surface = self.wl_surface()?;
        DndFocus::enter(&*surface, data, dh, source, seat, location, serial)
    }

    fn motion<S: Source>(
        &self,
        data: &mut State,
        offer: Option<&mut Self::OfferData<S>>,
        seat: &Seat<State>,
        location: Point<f64, Logical>,
        time: InputTime,
    ) {
        if let Some(surface) = self.wl_surface() {
            DndFocus::motion(&*surface, data, offer, seat, location, time)
        }
    }

    fn leave<S: Source>(
        &self,
        data: &mut State,
        offer: Option<&mut Self::OfferData<S>>,
        seat: &Seat<State>,
    ) {
        if let Some(surface) = self.wl_surface() {
            DndFocus::leave(&*surface, data, offer, seat)
        }
    }

    fn drop<S: Source>(
        &self,
        data: &mut State,
        offer: Option<&mut Self::OfferData<S>>,
        seat: &Seat<State>,
    ) {
        if let Some(surface) = self.wl_surface() {
            DndFocus::drop(&*surface, data, offer, seat)
        }
    }
}
