//! Regression guard for the per-session `memfd:smithay-keymap` fd leak.
//!
//! smithay mints one sealed memfd named `smithay-keymap` per `Seat::add_keyboard`
//! (`KeymapFile::new`), i.e. one per compositor instance, and keeps it in the
//! `Arc<KbdRc>` behind the seat's `KeyboardHandle`. A grab that is still installed
//! when the session ends lives *inside* that same `KbdRc` while holding a clone of
//! the handle, so the `Arc` references itself and dropping the compositor cannot
//! close the fd. [`State::release_seat`] unsets the grab at shutdown; these tests
//! pin both halves: a plain teardown stays clean, and a teardown with a grab still
//! active stays clean too.

use crate::comp::{FocusTarget, State};
use crate::tests::fixture::Fixture;
use smithay::backend::input::{KeyState, Keycode};
use smithay::input::keyboard::{
    GrabStartData, KeyboardGrab, KeyboardHandle, KeyboardInnerHandle, ModifiersState,
};
use smithay::utils::{SERIAL_COUNTER, Serial};
use test_log::test;

/// Open fds in this process whose target names the smithay keymap memfd.
fn keymap_fd_count() -> usize {
    let Ok(entries) = std::fs::read_dir("/proc/self/fd") else {
        // Not Linux / no procfs: the count is meaningless, keep it at 0 so the
        // assertions below trivially hold.
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            std::fs::read_link(entry.path())
                .map(|target| target.to_string_lossy().contains("smithay-keymap"))
                .unwrap_or(false)
        })
        .count()
}

/// One compositor lifecycle, round-tripped far enough that the client has bound a
/// `wl_keyboard`, a window is mapped, and the keyboard has focus and key history.
fn cycle() -> Fixture {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);
    f.round_trip();
    let keycode = f.server.scancode_to_keycode(30);
    f.server.keyboard_input(0, keycode, KeyState::Pressed);
    f.round_trip();
    f.server.keyboard_input(10, keycode, KeyState::Released);
    f.round_trip();
    f
}

/// A grab with the retaining shape of smithay's `PopupKeyboardGrab`: it holds a
/// clone of the `KeyboardHandle` whose internal state stores the grab
/// (`PopupGrab::keyboard_handle`). Installing one and never unsetting it is what a
/// client leaves behind when it dies with a popup grab open.
struct StaleGrab {
    _handle: KeyboardHandle<State>,
    start_data: GrabStartData<State>,
}

impl KeyboardGrab<State> for StaleGrab {
    fn input(
        &mut self,
        data: &mut State,
        handle: &mut KeyboardInnerHandle<'_, State>,
        keycode: Keycode,
        state: KeyState,
        modifiers: Option<ModifiersState>,
        serial: Serial,
        time: u32,
    ) {
        handle.input(data, keycode, state, modifiers, serial, time);
    }

    fn set_focus(
        &mut self,
        data: &mut State,
        handle: &mut KeyboardInnerHandle<'_, State>,
        focus: Option<FocusTarget>,
        serial: Serial,
    ) {
        handle.set_focus(data, focus, serial);
    }

    fn start_data(&self) -> &GrabStartData<State> {
        &self.start_data
    }

    fn unset(&mut self, _data: &mut State) {}
}

/// Growth tolerance: other tests in this binary may hold a fixture (and so a
/// keymap) alive while this one samples. It is well under `CYCLES`, which is what
/// the one-per-session leak produces.
const TOLERANCE: usize = 2;
const CYCLES: usize = 8;

/// `keymap_fd_count` reads a PROCESS-global number while cargo runs the rest of this
/// binary's tests in parallel threads, each holding its own fixture (and so its own
/// keymap fd) for the duration. A single sample can therefore read another test's
/// fixtures as growth. Re-sample a few times and keep the lowest reading: the leak
/// under test is permanent, so it survives every sample, while a neighbouring
/// fixture's fd disappears as soon as that test finishes.
fn settled_keymap_fd_count() -> usize {
    let mut lowest = keymap_fd_count();
    for _ in 0..5 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        lowest = lowest.min(keymap_fd_count());
    }
    lowest
}

#[test]
fn keymap_memfd_is_released_on_compositor_teardown() {
    // Warm-up cycle: the first compositor pulls in process-lifetime singletons
    // (GStreamer, the EGL display cache) whose fds must not count as growth.
    drop(cycle());

    let baseline = keymap_fd_count();
    for _ in 0..CYCLES {
        let mut f = cycle();
        f.server.release_seat();
        drop(f);
    }
    let after = settled_keymap_fd_count();

    assert!(
        after <= baseline + TOLERANCE,
        "smithay-keymap memfds grew across {CYCLES} compositor teardowns: \
         {baseline} -> {after}"
    );
}

#[test]
fn keymap_memfd_is_released_when_a_grab_is_still_active() {
    drop(cycle());

    let baseline = keymap_fd_count();
    for _ in 0..CYCLES {
        let mut f = cycle();
        let keyboard = f.server.seat.get_keyboard().unwrap();
        let grab = StaleGrab {
            _handle: keyboard.clone(),
            start_data: GrabStartData { focus: None },
        };
        keyboard.set_grab(&mut f.server, grab, SERIAL_COUNTER.next_serial());
        drop(keyboard);

        // The session ends with the grab still installed -- without
        // `release_seat` this is the self-referential Arc that leaks the keymap.
        f.server.release_seat();
        drop(f);
    }
    let after = settled_keymap_fd_count();

    assert!(
        after <= baseline + TOLERANCE,
        "smithay-keymap memfds grew across {CYCLES} teardowns with an active grab: \
         {baseline} -> {after}"
    );
}
