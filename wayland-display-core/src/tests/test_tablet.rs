//! Tablet tool (pen) input: native `zwp_tablet_v2` delivery to tablet-aware clients and
//! pointer emulation for the rest.
//!
//! Tablet support is detected per client connection (has it bound `zwp_tablet_seat_v2`?),
//! so these tests run two clients: one tablet-aware, one not, side by side. Windows are
//! placed server-side so a pen can cross from one to the other.

use crate::comp::ClientState;
use crate::comp::State;
use crate::tests::client::{MouseEvents, TabletEvents, WaylandClient};
use crate::tests::fixture::Fixture;
use smithay::backend::input::{
    ButtonState, TabletToolCapabilities, TabletToolDescriptor, TabletToolType,
};
use smithay::desktop::Window;
use smithay::input::tablet::TabletDescriptor;
use smithay::utils::{Logical, Point};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use test_log::test;
use wayland_client::protocol::wl_pointer;
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_tool_v2::Event as Tool;

const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;
const BTN_STYLUS: u32 = 0x14b;
const BTN_STYLUS2: u32 = 0x14c;
const BTN_STYLUS3: u32 = 0x149;

fn tablet() -> TabletDescriptor {
    TabletDescriptor {
        name: "Test tablet".into(),
        usb_id: None,
        syspath: None,
    }
}

fn pen(serial: u64) -> TabletToolDescriptor {
    TabletToolDescriptor {
        tool_type: TabletToolType::Pen,
        hardware_serial: serial,
        hardware_id_wacom: 0,
        capabilities: TabletToolCapabilities::PRESSURE | TabletToolCapabilities::TILT,
    }
}

fn at(x: f64, y: f64) -> Point<f64, Logical> {
    (x, y).into()
}

fn axes(pressure: Option<f64>, tilt: Option<(f64, f64)>) -> crate::comp::TabletToolAxes {
    crate::comp::TabletToolAxes {
        pressure,
        tilt,
        ..Default::default()
    }
}

/// A second client on the fixture's compositor.
fn connect(f: &mut Fixture) -> WaylandClient {
    let (server_end, client_end) = UnixStream::pair().unwrap();
    f.server
        .dh
        .insert_client(server_end, Arc::new(ClientState::default()))
        .unwrap();
    let mut c = WaylandClient::new(client_end);
    sync(f, &mut c);
    c
}

/// Round trip `c` (and the fixture's own client) through the compositor.
fn sync(f: &mut Fixture, c: &mut WaylandClient) {
    let done = c.send_sync();
    for _ in 0..200 {
        if done.done.load(Ordering::Relaxed) {
            break;
        }
        f.update_server();
        c.dispatch();
        f.update_client();
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(done.done.load(Ordering::Relaxed), "client sync timed out");
    f.round_trip();
}

fn new_elements(f: &Fixture, before: &[Window]) -> Window {
    f.server
        .space
        .elements()
        .find(|w| !before.contains(w))
        .cloned()
        .expect("no window was mapped")
}

/// Map a 320x240 window for `c` (None: the fixture's client) and return it, server-side.
fn window_for(f: &mut Fixture, c: Option<&mut WaylandClient>) -> Window {
    let before: Vec<Window> = f.server.space.elements().cloned().collect();
    match c {
        None => f.create_window(320, 240),
        Some(c) => {
            c.create_window();
            sync(f, c);
            c.setup_window(320, 240);
            sync(f, c);
            sync(f, c);
            for window in f.server.space.elements() {
                window.on_commit();
            }
        }
    }
    new_elements(f, &before)
}

/// Side-by-side placements for two 320x240 windows on the 320x240 output: x < 160 is only the
/// left window, x >= 160 only the right one, whatever the stacking (a tip-down raises the
/// window under the pen).
const LEFT: i32 = -160;
const RIGHT: i32 = 160;

/// Place `window` at (`x`, 0) and raise it.
fn place(f: &mut Fixture, window: &Window, x: i32) {
    f.server.space.map_element(window.clone(), (x, 0), true);
    f.server.space.refresh();
}

fn tool_events(c: &mut WaylandClient) -> Vec<Tool> {
    c.get_tablet_events()
        .drain(..)
        .filter_map(|e| match e {
            TabletEvents::Tool(e) => Some(e),
            TabletEvents::Seat(_) => None,
        })
        .collect()
}

fn pointer_events(c: &mut WaylandClient) -> Vec<wl_pointer::Event> {
    c.get_client_events()
        .drain(..)
        .filter_map(|e| match e {
            MouseEvents::Pointer(e) => Some(e),
            _ => None,
        })
        .collect()
}

fn buttons(events: &[wl_pointer::Event]) -> Vec<(u32, bool)> {
    events
        .iter()
        .filter_map(|e| match e {
            wl_pointer::Event::Button { button, state, .. } => Some((
                *button,
                matches!(
                    state,
                    wayland_client::WEnum::Value(wl_pointer::ButtonState::Pressed)
                ),
            )),
            _ => None,
        })
        .collect()
}

fn proximity_in_surfaces(events: &[Tool]) -> Vec<WlSurface> {
    events
        .iter()
        .filter_map(|e| match e {
            Tool::ProximityIn { surface, .. } => Some(surface.clone()),
            _ => None,
        })
        .collect()
}

fn position(events: &[Tool], pred: impl Fn(&Tool) -> bool) -> Option<usize> {
    events.iter().position(pred)
}

/// The tablet-aware fixture client, after the tablet was plugged in.
fn aware(f: &mut Fixture) {
    f.client.enable_tablet();
    f.round_trip();
}

fn plug(s: &mut State) {
    s.tablet_added(&tablet());
}

// A native stroke keeps its contact when it crosses into another window: the surface it
// started on keeps receiving motion until the tip comes up (implicit grab), and only then
// does the pen leave it. Once the tip is up no pressure lingers: pressure 0 reaches the client
// before `up`, even when the tip-up event reports no pressure change.
#[test]
fn native_stroke_keeps_contact_across_windows_and_ends_at_zero_pressure() {
    let mut f = Fixture::new();
    aware(&mut f);
    let mut other = connect(&mut f);
    let a = window_for(&mut f, None);
    let b = window_for(&mut f, Some(&mut other));
    place(&mut f, &a, LEFT);
    place(&mut f, &b, RIGHT);
    plug(&mut f.server);
    f.round_trip();
    tool_events(&mut f.client);

    let p = pen(1);
    let s = &mut f.server;
    s.tablet_tool_proximity(1, &tablet(), &p, at(50., 50.), axes(Some(0.0), None), true);
    s.tablet_tool_tip(2, &tablet(), &p, at(50., 50.), axes(Some(0.5), None), true);
    s.tablet_tool_axis(3, &tablet(), &p, at(250., 50.), axes(Some(0.6), None));
    s.tablet_tool_tip(4, &tablet(), &p, at(250., 50.), axes(None, None), false);
    sync(&mut f, &mut other);

    let events = tool_events(&mut f.client);
    assert_eq!(
        proximity_in_surfaces(&events),
        vec![f.client.window_surface(0).clone()]
    );
    let down = position(&events, |e| matches!(e, Tool::Down { .. })).expect("no down");
    let up = position(&events, |e| matches!(e, Tool::Up)).expect("no up");
    let crossed = position(
        &events,
        |e| matches!(e, Tool::Motion { x, .. } if *x == 250.0 - LEFT as f64),
    )
    .expect("the stroke stopped reaching its surface over the other window");
    assert!(down < crossed && crossed < up);
    let out = position(&events, |e| matches!(e, Tool::ProximityOut));
    assert!(
        out.is_none_or(|out| out > up),
        "the pen left the surface mid-stroke: {events:?}"
    );
    let last_pressure = events[..up]
        .iter()
        .rev()
        .find_map(|e| match e {
            Tool::Pressure { pressure } => Some(*pressure),
            _ => None,
        })
        .expect("no pressure");
    assert_eq!(last_pressure, 0, "pressure must be 0 by tip-up: {events:?}");
    // Nothing was emulated for the window the stroke crossed.
    assert!(buttons(&pointer_events(&mut other)).is_empty());
}

// The surface holding a native contact goes away mid-stroke: the contact moves to the surface
// now under the pen, which receives `down` after its proximity-in.
#[test]
fn native_contact_is_restored_on_the_surface_it_moves_to() {
    let mut f = Fixture::new();
    aware(&mut f);
    let _w0 = window_for(&mut f, None);
    let w1 = window_for(&mut f, None);
    place(&mut f, &w1, 0);
    plug(&mut f.server);
    f.round_trip();

    let p = pen(1);
    f.server
        .tablet_tool_proximity(1, &tablet(), &p, at(50., 50.), axes(None, None), true);
    f.server
        .tablet_tool_tip(2, &tablet(), &p, at(50., 50.), axes(Some(0.5), None), true);
    f.round_trip();
    tool_events(&mut f.client);

    f.client.destroy_window_at(1);
    f.round_trip();
    f.server.space.refresh();
    f.server
        .tablet_tool_axis(3, &tablet(), &p, at(60., 60.), axes(Some(0.5), None));
    f.round_trip();

    let events = tool_events(&mut f.client);
    let w0 = f.client.window_surface(0).clone();
    let enter = position(
        &events,
        |e| matches!(e, Tool::ProximityIn { surface, .. } if *surface == w0),
    )
    .expect("the remaining window never got the pen");
    let down = position(&events, |e| matches!(e, Tool::Down { .. }))
        .expect("the stroke's contact was not restored");
    assert!(enter < down, "{events:?}");
}

// An emulated drag (a client without tablet-v2) keeps receiving pointer motion when the pen
// crosses into a tablet-aware window, and the tablet-aware client sees none of it.
#[test]
fn emulated_drag_keeps_pointer_motion_over_a_tablet_aware_window() {
    let mut f = Fixture::new();
    let mut tablet_client = connect(&mut f);
    tablet_client.enable_tablet();
    sync(&mut f, &mut tablet_client);
    let n = window_for(&mut f, None);
    let t = window_for(&mut f, Some(&mut tablet_client));
    place(&mut f, &n, LEFT);
    place(&mut f, &t, RIGHT);
    plug(&mut f.server);
    sync(&mut f, &mut tablet_client);
    pointer_events(&mut f.client);
    tool_events(&mut tablet_client);

    let p = pen(1);
    let s = &mut f.server;
    s.tablet_tool_proximity(1, &tablet(), &p, at(50., 50.), axes(None, None), true);
    s.tablet_tool_tip(2, &tablet(), &p, at(50., 50.), axes(Some(0.5), None), true);
    s.tablet_tool_axis(3, &tablet(), &p, at(250., 50.), axes(Some(0.5), None));
    sync(&mut f, &mut tablet_client);

    let events = pointer_events(&mut f.client);
    assert_eq!(buttons(&events), vec![(BTN_LEFT, true)]);
    assert!(
        events.iter().any(
            |e| matches!(e, wl_pointer::Event::Motion { surface_x, .. } if *surface_x == 250.0 - LEFT as f64)
        ),
        "the drag lost pointer motion over the tablet-aware window: {events:?}"
    );
    assert!(proximity_in_surfaces(&tool_events(&mut tablet_client)).is_empty());

    f.server
        .tablet_tool_tip(4, &tablet(), &p, at(250., 50.), axes(None, None), false);
    sync(&mut f, &mut tablet_client);
    assert_eq!(
        buttons(&pointer_events(&mut f.client)),
        vec![(BTN_LEFT, false)]
    );
}

// Button ownership is per tool and per physical button: two stylus buttons mapped to middle
// click, and a second pen pressing it too, hold one middle press that is released only when
// the last of them lets go (a pen leaving proximity releases only its own buttons).
#[test]
fn emulated_button_is_released_by_its_last_owner() {
    let mut f = Fixture::new();
    window_for(&mut f, None);
    plug(&mut f.server);
    f.round_trip();
    pointer_events(&mut f.client);

    let (a, b) = (pen(1), pen(2));
    let s = &mut f.server;
    s.tablet_tool_proximity(1, &tablet(), &a, at(50., 50.), axes(None, None), true);
    s.tablet_tool_button(2, &tablet(), &a, BTN_STYLUS2, ButtonState::Pressed);
    s.tablet_tool_button(3, &tablet(), &a, BTN_STYLUS3, ButtonState::Pressed);
    s.tablet_tool_button(4, &tablet(), &a, BTN_STYLUS2, ButtonState::Released);
    s.tablet_tool_proximity(5, &tablet(), &b, at(50., 50.), axes(None, None), true);
    s.tablet_tool_button(6, &tablet(), &b, BTN_STYLUS2, ButtonState::Pressed);
    s.tablet_tool_proximity(7, &tablet(), &b, at(50., 50.), axes(None, None), false);
    f.round_trip();
    assert_eq!(
        buttons(&pointer_events(&mut f.client)),
        vec![(BTN_MIDDLE, true)],
        "middle released while another button or pen still held it"
    );

    f.server
        .tablet_tool_button(8, &tablet(), &a, BTN_STYLUS3, ButtonState::Released);
    f.round_trip();
    assert_eq!(
        buttons(&pointer_events(&mut f.client)),
        vec![(BTN_MIDDLE, false)]
    );
}

// A stylus button goes to the application under the pen, not under the mouse, for a native
// pen ...
#[test]
fn native_stylus_button_follows_the_pen_not_the_mouse() {
    let mut f = Fixture::new();
    aware(&mut f);
    let mut other = connect(&mut f);
    let t = window_for(&mut f, None);
    let n = window_for(&mut f, Some(&mut other));
    place(&mut f, &t, LEFT);
    place(&mut f, &n, RIGHT);
    plug(&mut f.server);
    f.round_trip();

    let p = pen(1);
    f.server
        .tablet_tool_proximity(1, &tablet(), &p, at(50., 50.), axes(None, None), true);
    f.server.pointer_motion_absolute(2, at(250., 50.));
    f.server
        .tablet_tool_button(3, &tablet(), &p, BTN_STYLUS, ButtonState::Pressed);
    sync(&mut f, &mut other);

    assert!(
        tool_events(&mut f.client)
            .iter()
            .any(|e| matches!(e, Tool::Button { button, .. } if *button == BTN_STYLUS)),
        "the pen's button missed the window under the pen"
    );
    assert!(buttons(&pointer_events(&mut other)).is_empty());
}

// ... and for an emulated one.
#[test]
fn emulated_stylus_button_follows_the_pen_not_the_mouse() {
    let mut f = Fixture::new();
    let mut tablet_client = connect(&mut f);
    tablet_client.enable_tablet();
    sync(&mut f, &mut tablet_client);
    let n = window_for(&mut f, None);
    let t = window_for(&mut f, Some(&mut tablet_client));
    place(&mut f, &n, LEFT);
    place(&mut f, &t, RIGHT);
    plug(&mut f.server);
    sync(&mut f, &mut tablet_client);

    let p = pen(1);
    f.server
        .tablet_tool_proximity(1, &tablet(), &p, at(50., 50.), axes(None, None), true);
    // The mouse moves over the other window (twice: relative motion settles the pointer
    // focus one event late), leaving the pen where it was.
    f.server.pointer_motion_absolute(2, at(250., 50.));
    f.server.pointer_motion_absolute(2, at(250., 50.));
    sync(&mut f, &mut tablet_client);
    pointer_events(&mut f.client);
    pointer_events(&mut tablet_client);
    f.server
        .tablet_tool_button(3, &tablet(), &p, BTN_STYLUS, ButtonState::Pressed);
    sync(&mut f, &mut tablet_client);

    assert_eq!(
        buttons(&pointer_events(&mut f.client)),
        vec![(BTN_RIGHT, true)]
    );
    assert!(buttons(&pointer_events(&mut tablet_client)).is_empty());
}

// The axes carried by a proximity-in, and by a motion that moves the pen to another surface,
// reach the client with that event, not with some later motion.
#[test]
fn proximity_and_focus_changes_deliver_their_axes() {
    let mut f = Fixture::new();
    aware(&mut f);
    let w0 = window_for(&mut f, None);
    let w1 = window_for(&mut f, None);
    place(&mut f, &w0, LEFT);
    place(&mut f, &w1, RIGHT);
    plug(&mut f.server);
    f.round_trip();
    tool_events(&mut f.client);

    let p = pen(1);
    f.server.tablet_tool_proximity(
        1,
        &tablet(),
        &p,
        at(50., 50.),
        axes(None, Some((10., 5.))),
        true,
    );
    f.round_trip();
    let events = tool_events(&mut f.client);
    assert!(
        events.iter().any(
            |e| matches!(e, Tool::Tilt { tilt_x, tilt_y } if *tilt_x == 10.0 && *tilt_y == 5.0)
        ),
        "proximity-in left its tilt queued: {events:?}"
    );

    f.server
        .tablet_tool_axis(2, &tablet(), &p, at(250., 50.), axes(None, Some((20., 0.))));
    f.round_trip();
    let events = tool_events(&mut f.client);
    let w1_surface = f.client.window_surface(1).clone();
    let enter = position(
        &events,
        |e| matches!(e, Tool::ProximityIn { surface, .. } if *surface == w1_surface),
    )
    .expect("the pen did not move to the second window");
    assert!(
        events[enter..]
            .iter()
            .any(|e| matches!(e, Tool::Tilt { tilt_x, .. } if *tilt_x == 20.0)),
        "the focus change left its tilt queued: {events:?}"
    );
}

// The pen targets the surface actually under it -- here a subsurface -- not just the window's
// root surface.
#[test]
fn pen_targets_the_subsurface_under_it() {
    let mut f = Fixture::new();
    aware(&mut f);
    window_for(&mut f, None);
    let sub = f.client.create_subsurface(0, 100, 100, 50, 50);
    f.round_trip();
    f.round_trip();
    plug(&mut f.server);
    f.round_trip();
    tool_events(&mut f.client);

    let p = pen(1);
    f.server
        .tablet_tool_proximity(1, &tablet(), &p, at(120., 120.), axes(None, None), true);
    f.round_trip();
    assert_eq!(
        proximity_in_surfaces(&tool_events(&mut f.client)),
        vec![sub]
    );

    f.server
        .tablet_tool_axis(2, &tablet(), &p, at(10., 10.), axes(None, None));
    f.round_trip();
    assert_eq!(
        proximity_in_surfaces(&tool_events(&mut f.client)),
        vec![f.client.window_surface(0).clone()]
    );
}

// A pointer lock holds the pointer against the pen as it does against the mouse: an emulated
// pen (non-tablet client) must not warp a locked pointer.
#[test]
fn emulated_pen_does_not_move_a_locked_pointer() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);
    plug(&mut f.server);
    let locked_at = at(15., 45.);
    f.server.pointer_motion_absolute(0, locked_at);
    f.round_trip();
    let _lock = f.client.lock_pointer();
    f.round_trip();

    let p = pen(1);
    f.server
        .tablet_tool_proximity(1, &tablet(), &p, at(200., 150.), axes(None, None), true);
    f.server
        .tablet_tool_proximity(2, &tablet(), &p, at(250., 180.), axes(None, None), true);
    f.round_trip();
    assert_eq!(f.server.pointer_location, locked_at);
}

// A pointer lock also holds the cursor against a native pen: hovering a tablet-aware window
// must not move the pointer, or the next mouse motion would start from the pen and escape
// the lock.
#[test]
fn native_pen_hover_does_not_move_a_locked_pointer() {
    let mut f = Fixture::new();
    aware(&mut f);
    window_for(&mut f, None);
    plug(&mut f.server);
    let locked_at = at(15., 45.);
    f.server.pointer_motion_absolute(0, locked_at);
    f.round_trip();
    let _lock = f.client.lock_pointer();
    f.round_trip();
    tool_events(&mut f.client);

    let p = pen(1);
    f.server
        .tablet_tool_proximity(1, &tablet(), &p, at(200., 150.), axes(None, None), true);
    f.server
        .tablet_tool_axis(2, &tablet(), &p, at(250., 180.), axes(None, None));
    f.round_trip();
    assert!(
        !proximity_in_surfaces(&tool_events(&mut f.client)).is_empty(),
        "the pen must still reach the tablet-aware window natively"
    );
    assert_eq!(f.server.pointer_location, locked_at);
}

// The mouse and a pen share pointer buttons: a mouse click during a pen's emulated drag must
// not end the drag, and lifting the pen afterwards releases the button exactly once.
#[test]
fn mouse_click_does_not_end_a_pen_drag() {
    let mut f = Fixture::new();
    window_for(&mut f, None);
    plug(&mut f.server);
    f.round_trip();
    pointer_events(&mut f.client);

    let p = pen(1);
    let s = &mut f.server;
    s.tablet_tool_proximity(1, &tablet(), &p, at(50., 50.), axes(None, None), true);
    s.tablet_tool_tip(2, &tablet(), &p, at(50., 50.), axes(Some(0.5), None), true);
    s.pointer_button(3, BTN_LEFT, ButtonState::Pressed);
    s.pointer_button(4, BTN_LEFT, ButtonState::Released);
    f.round_trip();
    assert_eq!(
        buttons(&pointer_events(&mut f.client)),
        vec![(BTN_LEFT, true)],
        "the mouse ended the pen's drag"
    );

    f.server
        .tablet_tool_tip(5, &tablet(), &p, at(50., 50.), axes(None, None), false);
    f.round_trip();
    assert_eq!(
        buttons(&pointer_events(&mut f.client)),
        vec![(BTN_LEFT, false)]
    );
}

// Pens without a serial (serial 0) on two tablets are separate tools: unplugging one tablet
// must not lift the other tablet's pen.
#[test]
fn serial_less_pens_on_two_tablets_are_separate_tools() {
    let mut f = Fixture::new();
    window_for(&mut f, None);
    let second = TabletDescriptor {
        name: "Second tablet".into(),
        usb_id: None,
        syspath: None,
    };
    plug(&mut f.server);
    f.server.tablet_added(&second);
    f.round_trip();
    pointer_events(&mut f.client);

    let p = pen(0);
    let s = &mut f.server;
    s.tablet_tool_proximity(1, &tablet(), &p, at(50., 50.), axes(None, None), true);
    s.tablet_tool_tip(2, &tablet(), &p, at(50., 50.), axes(Some(0.5), None), true);
    s.tablet_tool_proximity(3, &second, &p, at(60., 60.), axes(None, None), true);
    s.tablet_removed(&second);
    f.round_trip();
    assert_eq!(
        buttons(&pointer_events(&mut f.client)),
        vec![(BTN_LEFT, true)],
        "unplugging one tablet lifted the other tablet's pen"
    );
}

// The same two serial-less pens for a tablet-aware client: each is its own
// `zwp_tablet_tool_v2`, so unplugging the second tablet doesn't end the first pen's stroke.
#[test]
fn serial_less_pens_on_two_tablets_are_separate_native_tools() {
    let mut f = Fixture::new();
    aware(&mut f);
    window_for(&mut f, None);
    let second = TabletDescriptor {
        name: "Second tablet".into(),
        usb_id: None,
        syspath: None,
    };
    plug(&mut f.server);
    f.server.tablet_added(&second);
    f.round_trip();
    tool_events(&mut f.client);

    let p = pen(0);
    let s = &mut f.server;
    s.tablet_tool_proximity(1, &tablet(), &p, at(50., 50.), axes(None, None), true);
    s.tablet_tool_tip(2, &tablet(), &p, at(50., 50.), axes(Some(0.5), None), true);
    s.tablet_tool_proximity(3, &second, &p, at(60., 60.), axes(None, None), true);
    s.tablet_removed(&second);
    s.tablet_tool_axis(4, &tablet(), &p, at(55., 55.), axes(Some(0.6), None));
    f.round_trip();
    let events = tool_events(&mut f.client);
    assert!(
        !events.iter().any(|e| matches!(e, Tool::Up)),
        "unplugging one tablet lifted the other tablet's pen: {events:?}"
    );
    let down = position(&events, |e| matches!(e, Tool::Down { .. })).expect("no down");
    let last_motion = events
        .iter()
        .rposition(|e| matches!(e, Tool::Motion { x, .. } if *x == 55.0))
        .expect("the first pen's stroke stopped reaching the client");
    assert!(down < last_motion);
}

// Tip-up sends pressure 0 even when the device reports a small residual pressure with it.
#[test]
fn tip_up_zeroes_a_residual_pressure() {
    let mut f = Fixture::new();
    aware(&mut f);
    window_for(&mut f, None);
    plug(&mut f.server);
    f.round_trip();
    tool_events(&mut f.client);

    let p = pen(1);
    let s = &mut f.server;
    s.tablet_tool_proximity(1, &tablet(), &p, at(50., 50.), axes(Some(0.0), None), true);
    s.tablet_tool_tip(2, &tablet(), &p, at(50., 50.), axes(Some(0.5), None), true);
    s.tablet_tool_tip(
        3,
        &tablet(),
        &p,
        at(50., 50.),
        axes(Some(0.02), None),
        false,
    );
    f.round_trip();

    let events = tool_events(&mut f.client);
    let up = position(&events, |e| matches!(e, Tool::Up)).expect("no up");
    let last_pressure = events[..up].iter().rev().find_map(|e| match e {
        Tool::Pressure { pressure } => Some(*pressure),
        _ => None,
    });
    assert_eq!(last_pressure, Some(0), "{events:?}");
}

// A pen's cursor image (here: hidden) is the pen's: the mouse moving again shows the
// pointer's own image, also one its client set while the pen's was shown.
#[test]
fn pen_cursor_image_does_not_stick_for_the_mouse() {
    use smithay::input::SeatHandler;
    use smithay::input::pointer::CursorImageStatus;
    use smithay::input::tablet::TabletSeatHandler;
    let mut f = Fixture::new();
    window_for(&mut f, None);
    let pointer_image = f.server.cursor_state.clone();
    assert_ne!(pointer_image, CursorImageStatus::Hidden);

    f.server
        .tablet_tool_image(&pen(1), CursorImageStatus::Hidden);
    assert_eq!(f.server.cursor_state, CursorImageStatus::Hidden);
    f.server.pointer_motion_absolute(1, at(20., 20.));
    assert_eq!(f.server.cursor_state, pointer_image);

    f.server
        .tablet_tool_image(&pen(1), CursorImageStatus::Hidden);
    let seat = f.server.seat.clone();
    let named = CursorImageStatus::Named(smithay::input::pointer::CursorIcon::Crosshair);
    f.server.cursor_image(&seat, named.clone());
    assert_eq!(
        f.server.cursor_state,
        CursorImageStatus::Hidden,
        "the pointer's client took the cursor from the pen"
    );
    f.server.pointer_motion_absolute(2, at(30., 30.));
    assert_eq!(f.server.cursor_state, named);
}
