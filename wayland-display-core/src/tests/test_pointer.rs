use crate::ButtonState;
use crate::tests::client::MouseEvents;
use crate::tests::fixture::Fixture;
use smithay::utils::Point;
use test_log::test;
use wayland_client::protocol::wl_pointer;
use wayland_protocols::wp::relative_pointer::zv1::client::zwp_relative_pointer_v1;

fn clean_events(events: &mut Vec<MouseEvents>) {
    while let Some(_event) = events.pop() {}
}

#[test]
fn move_mouse() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);

    {
        // Mapping a toplevel now resolves pointer focus immediately: the map handler emits a
        // synthetic zero-delta motion, so the client receives its wl_pointer.enter at the
        // pointer's *current* location without waiting for a physical motion event.
        let map_location = f.server.pointer_location;

        let client_events = f.client.get_client_events();
        assert!(!client_events.is_empty());
        let MouseEvents::Pointer(client_event) = client_events.remove(0) else {
            panic!("Unexpected event: {:?}", client_events);
        };
        let wl_pointer::Event::Enter {
            surface_x,
            surface_y,
            ..
        } = client_event
        else {
            panic!("Unexpected event: {:?}", client_event);
        };
        assert_eq!(surface_x, map_location.x);
        assert_eq!(surface_y, map_location.y);

        clean_events(client_events);
    }

    let expected_location = Point::from((0.0, 0.0));
    f.server.pointer_motion_absolute(0, expected_location);
    f.round_trip();

    {
        // Server logic test
        assert_eq!(f.server.pointer_location, expected_location);

        // Client logic test
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 1);
        // This first real motion after the map consumes the pending refocus edge trigger,
        // which cycles focus exactly once: one forced leave (plus its frame), then the
        // enter asserted below, so that a wl_pointer created after the map still learns it
        // has focus. Assert that shape precisely -- more than one leave would mean the
        // trigger is not edge-triggered.
        let leave = client_events.remove(0);
        assert!(
            matches!(leave, MouseEvents::Pointer(wl_pointer::Event::Leave { .. })),
            "expected the forced leave first, got: {:?}",
            leave
        );
        if matches!(
            client_events.first(),
            Some(MouseEvents::Pointer(wl_pointer::Event::Frame))
        ) {
            client_events.remove(0);
        }
        let MouseEvents::Pointer(client_event) = client_events.remove(0) else {
            panic!("Unexpected event: {:?}", client_events);
        };
        let wl_pointer::Event::Enter {
            // First time, we are entering the window
            surface_x,
            surface_y,
            ..
        } = client_event
        else {
            panic!("Unexpected event: {:?}", client_event);
        };
        assert_eq!(surface_x, expected_location.x);
        assert_eq!(surface_y, expected_location.y);

        clean_events(client_events);
    }

    let delta = Point::from((10.0, 15.0));
    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();

    {
        // Server logic test
        assert_eq!(f.server.pointer_location, expected_location + delta);

        // Client logic test
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 1);
        let MouseEvents::Pointer(client_event) = client_events.remove(0) else {
            panic!("Unexpected event: {:?}", client_events);
        };
        let wl_pointer::Event::Motion {
            // Second time, we are moving thru it
            surface_x,
            surface_y,
            ..
        } = client_event
        else {
            panic!("Unexpected event: {:?}", client_event);
        };
        assert_eq!(surface_x, delta.x);
        assert_eq!(surface_y, delta.y);

        clean_events(client_events);
    }
}

#[test]
fn lock_mouse() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);

    let expected_location = Point::from((15.0, 45.0));
    f.server.pointer_motion_absolute(0, expected_location);
    f.round_trip();
    {
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 1);
        clean_events(client_events);
    }

    let _lock = f.client.lock_pointer();
    let _relative_pointer = f.client.get_relative_pointer();
    f.round_trip();

    // Test pointer_motion()
    let delta = Point::from((10.0, 15.0));
    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();
    {
        // Mouse shouldn't be moved!
        assert_eq!(f.server.pointer_location, expected_location);

        // But we should still get Relative mouse events
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 2);
        let MouseEvents::Relative(client_event) = client_events.remove(0) else {
            panic!("Unexpected event: {:?}", client_events);
        };
        let zwp_relative_pointer_v1::Event::RelativeMotion { dx, dy, .. } = client_event else {
            panic!("Unexpected event: {:?}", client_event);
        };
        assert_eq!(dx, delta.x);
        assert_eq!(dy, delta.y);

        // And no Pointer Motion events
        while let Some(event) = client_events.pop() {
            match event {
                MouseEvents::Pointer(p_event) => match p_event {
                    wl_pointer::Event::Motion { .. } => panic!("Unexpected event: {:?}", p_event),
                    _ => {}
                },
                _ => {}
            }
        }
    }

    // Test pointer_motion_absolute()
    let absolute_position = Point::from((100.0, 150.0));
    f.server.pointer_motion_absolute(0, absolute_position);
    f.round_trip();

    {
        // Mouse shouldn't be moved!
        assert_eq!(f.server.pointer_location, expected_location);

        // But we should still get Relative mouse events
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 2);
        let MouseEvents::Relative(client_event) = client_events.remove(0) else {
            panic!("Unexpected event: {:?}", client_events);
        };
        let zwp_relative_pointer_v1::Event::RelativeMotion { dx, dy, .. } = client_event else {
            panic!("Unexpected event: {:?}", client_event);
        };
        assert_eq!(dx, absolute_position.x - f.server.pointer_location.x);
        assert_eq!(dy, absolute_position.y - f.server.pointer_location.y);

        // And no Pointer Motion events
        while let Some(event) = client_events.pop() {
            match event {
                MouseEvents::Pointer(p_event) => match p_event {
                    wl_pointer::Event::Motion { .. } => panic!("Unexpected event: {:?}", p_event),
                    _ => {}
                },
                _ => {}
            }
        }
    }
}

#[test]
fn confine_mouse_absolute_movement() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);

    // We start with the pointer at (0,0)
    let expected_location = Point::from((0.0, 0.0));
    f.server.pointer_motion_absolute(0, expected_location);
    f.round_trip();

    {
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 1);
        clean_events(client_events);
    }

    // We create a confine region in the south right corner
    let _confine = f.client.confine_pointer(200, 100, 120, 140);
    let _relative_pointer = f.client.get_relative_pointer();
    f.round_trip();

    let outside_position = Point::from((50.0, 50.0));
    f.server.pointer_motion_absolute(0, outside_position);
    f.round_trip();
    {
        // The confinement shouldn't be active, we aren't in the region
        assert!(!f.client.is_confined());

        // The pointer has been moved correctly
        assert_eq!(f.server.pointer_location, outside_position);

        // Pointer Motion and Relative Motion events have been triggered
        let client_events = f.client.get_client_events();
        assert_eq!(client_events.len(), 3); // motion, relative_motion, frame
        while let Some(event) = client_events.pop() {
            match event {
                MouseEvents::Pointer(p_event) => match p_event {
                    wl_pointer::Event::Motion {
                        surface_x,
                        surface_y,
                        ..
                    } => {
                        assert_eq!(surface_x, outside_position.x);
                        assert_eq!(surface_y, outside_position.y);
                    }
                    _ => {}
                },
                MouseEvents::Relative(r_event) => match r_event {
                    zwp_relative_pointer_v1::Event::RelativeMotion { dx, dy, .. } => {
                        assert_eq!(dx, outside_position.x);
                        assert_eq!(dy, outside_position.y);
                    }
                    _ => {}
                },
            }
        }
    }

    // Let's now move to the confined region
    let inside_position = Point::from((250.0, 150.0));
    f.server.pointer_motion_absolute(0, inside_position);
    f.round_trip();
    {
        // The confinement should have been activated now
        assert!(f.client.is_confined());

        // The pointer has been moved correctly
        assert_eq!(f.server.pointer_location, inside_position);

        // Pointer Motion and Relative Motion events have been triggered
        let client_events = f.client.get_client_events();
        assert_eq!(client_events.len(), 3); // motion, relative_motion, frame
        while let Some(event) = client_events.pop() {
            match event {
                MouseEvents::Pointer(p_event) => match p_event {
                    wl_pointer::Event::Motion {
                        surface_x,
                        surface_y,
                        ..
                    } => {
                        assert_eq!(surface_x, inside_position.x);
                        assert_eq!(surface_y, inside_position.y);
                    }
                    _ => {}
                },
                MouseEvents::Relative(r_event) => match r_event {
                    zwp_relative_pointer_v1::Event::RelativeMotion { dx, dy, .. } => {
                        assert_eq!(dx, inside_position.x - outside_position.x);
                        assert_eq!(dy, inside_position.y - outside_position.y);
                    }
                    _ => {}
                },
            }
        }
    }

    // Now, we shouldn't be able to move back out to the confined region
    f.server.pointer_motion_absolute(0, outside_position);
    f.round_trip();
    {
        // The confinement should still be active
        assert!(f.client.is_confined());

        // The pointer is still where it was
        assert_eq!(f.server.pointer_location, inside_position);

        // We'll only get a relative motion event in this case
        // Pointer Motion and Relative Motion events have been triggered
        let client_events = f.client.get_client_events();
        assert_eq!(client_events.len(), 2); // relative_motion, frame
        while let Some(event) = client_events.pop() {
            match event {
                MouseEvents::Pointer(p_event) => match p_event {
                    wl_pointer::Event::Frame { .. } => {}
                    _ => {
                        panic!("Unexpected event: {:?}", p_event)
                    }
                },
                MouseEvents::Relative(r_event) => match r_event {
                    zwp_relative_pointer_v1::Event::RelativeMotion { dx, dy, .. } => {
                        assert_eq!(dx, -(inside_position.x - outside_position.x));
                        assert_eq!(dy, -(inside_position.y - outside_position.y));
                    }
                    _ => {}
                },
            }
        }
    }
}

/// The map-time refocus reaches a client whose `wl_pointer` did not exist when focus was
/// first resolved.
///
/// This is the race the fix exists for: the map handler's synthetic motion delivers
/// `wl_pointer.enter` only to the resources alive at that instant, and smithay records the
/// new focus unconditionally, so every later motion takes its same-target arm. A client that
/// calls `wl_seat.get_pointer` afterwards would otherwise never learn it has focus.
///
/// The flag is armed by the map handler itself -- the test never touches server state.
#[test]
fn map_time_refocus_reaches_a_late_pointer() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);
    f.round_trip();

    // The client loses the race: its wl_pointer is created after the map resolved focus,
    // so it has never seen an enter.
    f.client.recreate_pointer();
    f.round_trip();
    clean_events(f.client.get_client_events());

    // The first real motion after the map consumes the armed refocus and cycles smithay's
    // focus, so the new resource gets its enter.
    let delta = Point::from((5.0, 5.0));
    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();

    let client_events = f.client.get_client_events();
    assert!(
        client_events
            .iter()
            .any(|e| matches!(e, MouseEvents::Pointer(wl_pointer::Event::Enter { .. }))),
        "a wl_pointer created after the map must still receive an enter, got: {:?}",
        client_events
    );

    // Edge-triggered: exactly one leave, so the refocus fires once and not on every motion.
    let leaves = client_events
        .iter()
        .filter(|e| matches!(e, MouseEvents::Pointer(wl_pointer::Event::Leave { .. })))
        .count();
    assert_eq!(
        leaves, 1,
        "refocus must fire once, got: {:?}",
        client_events
    );
    clean_events(client_events);

    // A second motion is motion-only again (the trigger is spent, see `move_mouse`).
    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();
    let client_events = f.client.get_client_events();
    assert!(
        !client_events
            .iter()
            .any(|e| matches!(e, MouseEvents::Pointer(wl_pointer::Event::Enter { .. }))),
        "the refocus must not repeat on later motions, got: {:?}",
        client_events
    );
}

/// The refocus is DEFERRED while the pointer is grabbed. smithay's default click grab is
/// live for as long as a button is held, and a forced leave mid-click-drag would break the
/// drag.
#[test]
fn refocus_deferred_while_grabbed() {
    const BTN_LEFT: u32 = 0x110;

    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);
    f.round_trip();

    // No absolute motion here: it delegates to `pointer_motion` and would spend the
    // map-time trigger before the grab is installed.
    f.client.recreate_pointer();
    f.round_trip();

    // Press: smithay's DefaultGrab installs the click grab, live until the release below.
    f.server.pointer_button(0, BTN_LEFT, ButtonState::Pressed);
    f.round_trip();
    clean_events(f.client.get_client_events());

    let delta = Point::from((5.0, 5.0));
    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();

    let client_events = f.client.get_client_events();
    assert!(
        !client_events
            .iter()
            .any(|e| matches!(e, MouseEvents::Pointer(wl_pointer::Event::Leave { .. }))),
        "no forced leave may be sent mid-click-drag, got: {:?}",
        client_events
    );
    clean_events(client_events);

    // Release, and the very next motion delivers the deferred refocus.
    f.server.pointer_button(0, BTN_LEFT, ButtonState::Released);
    f.round_trip();
    clean_events(f.client.get_client_events());

    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();

    let client_events = f.client.get_client_events();
    assert!(
        client_events
            .iter()
            .any(|e| matches!(e, MouseEvents::Pointer(wl_pointer::Event::Enter { .. }))),
        "the deferred refocus must fire once the grab ends, got: {:?}",
        client_events
    );
}

/// CONSTRAINT GUARD: an ACTIVE pointer constraint proves the client already received its
/// enter, because a lock/confine only activates on a surface the pointer has entered. The
/// refocus is therefore unnecessary rather than deferred, and must NOT force a leave --
/// smithay deactivates a constraint on focus loss, so cycling focus here would silently
/// break the client's pointer lock (nested gamescope with --force-grab-cursor).
#[test]
fn active_constraint_suppresses_the_forced_leave() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);
    f.round_trip();

    // Enter the surface so the lock is allowed to activate.
    f.server
        .pointer_motion_absolute(0, Point::from((10.0, 10.0)));
    f.round_trip();

    let _lock = f.client.lock_pointer();
    let _relative = f.client.get_relative_pointer();
    f.round_trip();
    clean_events(f.client.get_client_events());

    // The map-time trigger is still armed here; this motion hits the constraint guard.
    let delta = Point::from((5.0, 5.0));
    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();

    let client_events = f.client.get_client_events();
    assert!(
        !client_events
            .iter()
            .any(|e| matches!(e, MouseEvents::Pointer(wl_pointer::Event::Leave { .. }))),
        "an active constraint must suppress the forced leave, got: {:?}",
        client_events
    );
    clean_events(client_events);

    // And the trigger is CLEARED, not deferred: a further motion must not force one either.
    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();
    let client_events = f.client.get_client_events();
    assert!(
        !client_events
            .iter()
            .any(|e| matches!(e, MouseEvents::Pointer(wl_pointer::Event::Leave { .. }))),
        "the guard must clear the trigger, not retry it, got: {:?}",
        client_events
    );
}
