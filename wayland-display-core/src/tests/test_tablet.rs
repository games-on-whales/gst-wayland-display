use crate::comp::TabletToolAxes;
use crate::tests::client::{MouseEvents, TabletEvents};
use crate::tests::fixture::Fixture;
use smithay::backend::input::{
    ButtonState, TabletToolCapabilities, TabletToolDescriptor, TabletToolType,
};
use smithay::utils::Point;
use smithay::wayland::tablet_manager::TabletDescriptor;
use test_log::test;
use wayland_client::WEnum;
use wayland_client::protocol::wl_pointer;
use wayland_protocols::wp::tablet::zv2::client::{zwp_tablet_seat_v2, zwp_tablet_tool_v2};

const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_STYLUS: u32 = 0x14b;

fn tablet() -> TabletDescriptor {
    TabletDescriptor {
        name: "Wolf (virtual) pen tablet".into(),
        usb_id: Some((0xAB00, 0xAB04)),
        syspath: None,
    }
}

fn pen() -> TabletToolDescriptor {
    TabletToolDescriptor {
        tool_type: TabletToolType::Pen,
        hardware_serial: 0,
        hardware_id_wacom: 0,
        capabilities: TabletToolCapabilities::PRESSURE
            | TabletToolCapabilities::DISTANCE
            | TabletToolCapabilities::TILT,
    }
}

/// The eraser end of the same stylus: libinput reports it as a tool of its own, so it has to become a
/// zwp_tablet_tool_v2 of its own too (that's how a client learns it should erase rather than draw).
fn eraser() -> TabletToolDescriptor {
    TabletToolDescriptor {
        tool_type: TabletToolType::Eraser,
        ..pen()
    }
}

/// Mapping a toplevel resolves pointer focus immediately (the map handler emits a synthetic
/// zero-delta motion), so every client starts out with a `wl_pointer.enter` at the pointer's
/// location. Drop those: the assertions here are about what the *pen* delivers on top of them.
fn drain_map_time_pointer_events(f: &mut Fixture) {
    f.client.get_client_events().clear();
}

fn pointer_motions(events: &[MouseEvents]) -> Vec<(f64, f64)> {
    events
        .iter()
        .filter_map(|ev| match ev {
            MouseEvents::Pointer(wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            }) => Some((*surface_x, *surface_y)),
            MouseEvents::Pointer(wl_pointer::Event::Enter {
                surface_x,
                surface_y,
                ..
            }) => Some((*surface_x, *surface_y)),
            _ => None,
        })
        .collect()
}

fn axes(pressure: f64) -> TabletToolAxes {
    TabletToolAxes {
        pressure: Some(pressure),
        ..Default::default()
    }
}

fn pointer_buttons(events: &[MouseEvents]) -> Vec<(u32, wl_pointer::ButtonState)> {
    events
        .iter()
        .filter_map(|ev| match ev {
            MouseEvents::Pointer(wl_pointer::Event::Button {
                button,
                state: WEnum::Value(state),
                ..
            }) => Some((*button, *state)),
            _ => None,
        })
        .collect()
}

/// A client that speaks zwp_tablet_v2 gets the real thing: proximity, tip, pressure, motion
#[test]
fn tablet_aware_client_receives_tablet_events() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);
    f.client.enable_tablet();
    f.round_trip();
    drain_map_time_pointer_events(&mut f);

    f.server.tablet_added(&tablet());
    f.round_trip();

    let location = Point::from((100.0, 50.0));
    f.server
        .tablet_tool_proximity(1, &tablet(), &pen(), location, axes(0.0), true);
    f.server
        .tablet_tool_tip(2, &tablet(), &pen(), location, axes(0.4), true);
    f.server.tablet_tool_axis(
        3,
        &tablet(),
        &pen(),
        Point::from((110.0, 60.0)),
        TabletToolAxes {
            pressure: Some(0.5),
            tilt: Some((10.0, -20.0)),
            ..Default::default()
        },
    );
    f.server
        .tablet_tool_button(4, &pen(), BTN_STYLUS, ButtonState::Pressed);
    f.server
        .tablet_tool_button(5, &pen(), BTN_STYLUS, ButtonState::Released);
    // A tip up carries the pressure going back to 0 with it
    f.server.tablet_tool_tip(
        6,
        &tablet(),
        &pen(),
        Point::from((110.0, 60.0)),
        axes(0.0),
        false,
    );
    f.server.tablet_tool_proximity(
        7,
        &tablet(),
        &pen(),
        Point::from((110.0, 60.0)),
        TabletToolAxes::default(),
        false,
    );
    f.round_trip();

    // The cursor follows the pen
    assert_eq!(f.server.pointer_location, Point::from((110.0, 60.0)));

    let events = f.client.get_tablet_events();
    let has = |pred: &dyn Fn(&TabletEvents) -> bool| events.iter().any(pred);

    assert!(has(&|e| matches!(
        e,
        TabletEvents::Seat(zwp_tablet_seat_v2::Event::TabletAdded { .. })
    )));
    assert!(has(&|e| matches!(
        e,
        TabletEvents::Seat(zwp_tablet_seat_v2::Event::ToolAdded { .. })
    )));
    assert!(has(&|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::ProximityIn { .. })
    )));
    assert!(has(&|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::Down { .. })
    )));
    assert!(has(&|e| matches!(
        e,
        // 0.5 of the 0..65535 protocol range
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::Pressure { pressure }) if (32000..33500).contains(pressure)
    )));
    assert!(has(&|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::Tilt { .. })
    )));
    assert!(has(&|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::Motion { x, y }) if *x == 110.0 && *y == 60.0
    )));
    assert!(has(&|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::Button { button, .. }) if *button == BTN_STYLUS
    )));
    assert!(has(&|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::Up)
    )));
    assert!(has(&|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::ProximityOut)
    )));
    // The pressure goes back to 0 when the tip lifts (a tip event carries its axes with it)
    assert!(has(&|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::Pressure { pressure }) if *pressure == 0
    )));

    // ... and no emulated mouse clicks or motion on top of it (that would double every stroke)
    let client_events = f.client.get_client_events();
    assert!(pointer_buttons(client_events).is_empty());
    assert!(
        pointer_motions(client_events).is_empty(),
        "a tablet-aware client must not get wl_pointer motion for pen movement, got {:?}",
        pointer_motions(client_events)
    );
}

/// The pen and the eraser end of a stylus are separate tools, and both have to reach the client
#[test]
fn eraser_is_reported_as_its_own_tool() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);
    f.client.enable_tablet();
    f.round_trip();

    f.server.tablet_added(&tablet());
    f.round_trip();

    // A pen stroke, then the stylus is flipped over and the eraser draws
    let location = Point::from((100.0, 50.0));
    f.server
        .tablet_tool_proximity(1, &tablet(), &pen(), location, axes(0.0), true);
    f.server
        .tablet_tool_tip(2, &tablet(), &pen(), location, axes(0.5), true);
    f.server
        .tablet_tool_tip(3, &tablet(), &pen(), location, axes(0.0), false);
    f.server.tablet_tool_proximity(
        4,
        &tablet(),
        &pen(),
        location,
        TabletToolAxes::default(),
        false,
    );
    f.server
        .tablet_tool_proximity(5, &tablet(), &eraser(), location, axes(0.0), true);
    f.server
        .tablet_tool_tip(6, &tablet(), &eraser(), location, axes(0.6), true);
    f.round_trip();

    // Two tools were announced...
    let events = f.client.get_tablet_events();
    let tools_added = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                TabletEvents::Seat(zwp_tablet_seat_v2::Event::ToolAdded { .. })
            )
        })
        .count();
    assert_eq!(tools_added, 2, "events: {:?}", events);
    // ...one of which is an eraser
    assert!(events.iter().any(|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::Type {
            tool_type: WEnum::Value(zwp_tablet_tool_v2::Type::Eraser)
        })
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        TabletEvents::Tool(zwp_tablet_tool_v2::Event::Type {
            tool_type: WEnum::Value(zwp_tablet_tool_v2::Type::Pen)
        })
    )));
    // ...and the eraser drew: proximity in + down happened twice in total (once per tool)
    let downs = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                TabletEvents::Tool(zwp_tablet_tool_v2::Event::Down { .. })
            )
        })
        .count();
    assert_eq!(downs, 2, "events: {:?}", events);
}

/// A client that doesn't know about tablets (games, gamescope, ...) gets a mouse instead:
/// tip = left click, stylus button = right click
#[test]
fn non_tablet_client_gets_pointer_emulation() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);
    drain_map_time_pointer_events(&mut f);

    f.server.tablet_added(&tablet());
    f.round_trip();

    let location = Point::from((100.0, 50.0));
    f.server
        .tablet_tool_proximity(1, &tablet(), &pen(), location, axes(0.0), true);
    f.server
        .tablet_tool_tip(2, &tablet(), &pen(), location, axes(0.5), true);
    f.server.tablet_tool_axis(
        3,
        &tablet(),
        &pen(),
        Point::from((120.0, 70.0)),
        TabletToolAxes {
            pressure: Some(0.8),
            ..Default::default()
        },
    );
    f.server.tablet_tool_tip(
        4,
        &tablet(),
        &pen(),
        Point::from((120.0, 70.0)),
        axes(0.0),
        false,
    );
    f.server
        .tablet_tool_button(5, &pen(), BTN_STYLUS, ButtonState::Pressed);
    f.server
        .tablet_tool_button(6, &pen(), BTN_STYLUS, ButtonState::Released);
    f.round_trip();

    assert_eq!(f.server.pointer_location, Point::from((120.0, 70.0)));
    assert!(f.client.get_tablet_events().is_empty());
    let client_events = f.client.get_client_events();
    assert_eq!(
        pointer_buttons(client_events),
        vec![
            (BTN_LEFT, wl_pointer::ButtonState::Pressed),
            (BTN_LEFT, wl_pointer::ButtonState::Released),
            (BTN_RIGHT, wl_pointer::ButtonState::Pressed),
            (BTN_RIGHT, wl_pointer::ButtonState::Released),
        ]
    );
    // ...and, unlike a tablet-aware client, this one does get the pen as pointer motion
    assert!(
        !pointer_motions(client_events).is_empty(),
        "a client without tablet-v2 support must still see the pen as pointer motion"
    );
}

/// A pen that goes away mid-press must not leave a stuck mouse button behind
#[test]
fn pen_leaving_proximity_releases_emulated_buttons() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);
    drain_map_time_pointer_events(&mut f);

    f.server.tablet_added(&tablet());
    let location = Point::from((100.0, 50.0));
    f.server
        .tablet_tool_proximity(1, &tablet(), &pen(), location, axes(0.0), true);
    f.server
        .tablet_tool_tip(2, &tablet(), &pen(), location, axes(0.5), true);
    f.server.tablet_tool_proximity(
        3,
        &tablet(),
        &pen(),
        location,
        TabletToolAxes::default(),
        false,
    );
    f.round_trip();

    assert_eq!(
        pointer_buttons(f.client.get_client_events()),
        vec![
            (BTN_LEFT, wl_pointer::ButtonState::Pressed),
            (BTN_LEFT, wl_pointer::ButtonState::Released),
        ]
    );
    assert!(f.server.tablet_emulated_buttons.is_empty());
}
