use crate::comp::{NixInterface, State};
use crate::utils::RenderTarget;
use crate::utils::uinput::{KEY_A, UinputKeyboard};
use smithay::reexports::calloop::EventLoop;
use smithay::reexports::input::event::keyboard::{KeyState, KeyboardEventTrait};
use smithay::reexports::input::event::{DeviceEvent, Event, EventTrait};
use smithay::reexports::input::{Device, Libinput};
use smithay::reexports::wayland_server::Display;
use std::time::Duration;

struct Harness {
    state: State,
    libinput: Libinput,
    _display: Display<State>,
    _event_loop: EventLoop<'static, State>,
}

impl Harness {
    fn new() -> Self {
        let libinput = Libinput::new_from_path(NixInterface);
        let event_loop = EventLoop::<State>::try_new().unwrap();
        let display = Display::<State>::new().unwrap();
        let state = State::new(
            &RenderTarget::Software,
            &display.handle(),
            &libinput,
            event_loop.handle(),
            false,
            crate::utils::vulkan_nv12::ConversionOptions::default().hdr_peak_nits,
        );
        Harness {
            state,
            libinput,
            _display: display,
            _event_loop: event_loop,
        }
    }

    /// Drains libinput, passing device removals on to the compositor like its input backend
    /// does, and returns how many presses of `key` arrived.
    fn presses(&mut self, key: u16) -> usize {
        // Lets the kernel deliver what the test just wrote.
        std::thread::sleep(Duration::from_millis(100));
        self.libinput.dispatch().unwrap();
        let mut presses = 0;
        let mut removed: Vec<Device> = Vec::new();
        for event in &mut self.libinput {
            match event {
                Event::Device(DeviceEvent::Removed(event)) => removed.push(event.device()),
                Event::Keyboard(event)
                    if event.key() == key as u32 && event.key_state() == KeyState::Pressed =>
                {
                    presses += 1
                }
                _ => {}
            }
        }
        for device in removed {
            self.state.input_device_removed(&device);
        }
        presses
    }
}

// libinput doesn't de-duplicate paths, so a device announced twice got two handles
// and every key press arrived twice.
#[test]
#[ignore = "needs /dev/uinput and udev; run with --ignored"]
fn reannounced_device_delivers_each_key_once() {
    let mut harness = Harness::new();
    let keyboard = UinputKeyboard::new();
    harness.state.add_input_device(&keyboard.path);
    harness.state.add_input_device(&keyboard.path);
    harness.presses(KEY_A);

    keyboard.tap(KEY_A);
    assert_eq!(harness.presses(KEY_A), 1);
}

// A reconnected device can reuse its old path; its announcement must still be added, whether
// or not libinput has noticed the old device is gone.
#[test]
#[ignore = "needs /dev/uinput and udev; run with --ignored"]
fn reconnected_device_on_a_reused_path_is_added() {
    for noticed_removal in [true, false] {
        let mut harness = Harness::new();
        let first = UinputKeyboard::new();
        let path = first.path.clone();
        harness.state.add_input_device(&path);
        harness.presses(KEY_A);
        drop(first);
        if noticed_removal {
            harness.presses(KEY_A);
        }

        // The kernel hands out the lowest free event number, so this usually lands on `path`.
        let second = UinputKeyboard::new();
        harness.state.add_input_device(&second.path);
        harness.presses(KEY_A);
        second.tap(KEY_A);
        assert_eq!(
            harness.presses(KEY_A),
            1,
            "noticed_removal={noticed_removal}, reused path: {}",
            second.path == path
        );
    }
}
