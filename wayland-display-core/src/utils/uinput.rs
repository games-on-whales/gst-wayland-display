//! A virtual keyboard made with /dev/uinput, for tests that need libinput to open a real device.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

const UI_DEV_CREATE: libc::c_ulong = 0x5501;
const UI_DEV_DESTROY: libc::c_ulong = 0x5502;
const UI_DEV_SETUP: libc::c_ulong = 0x405c5503;
const UI_SET_EVBIT: libc::c_ulong = 0x40045564;
const UI_SET_KEYBIT: libc::c_ulong = 0x40045565;
const UI_GET_SYSNAME_64: libc::c_ulong = 0x8040552c;
const EV_SYN: u16 = 0;
const EV_KEY: u16 = 1;
pub const KEY_A: u16 = 30;

#[repr(C)]
struct UinputSetup {
    bustype: u16,
    vendor: u16,
    product: u16,
    version: u16,
    name: [u8; 80],
    ff_effects_max: u32,
}

#[repr(C)]
struct InputEvent {
    time: libc::timeval,
    kind: u16,
    code: u16,
    value: i32,
}

pub struct UinputKeyboard {
    file: File,
    /// The device's /dev/input/eventN node.
    pub path: String,
}

impl UinputKeyboard {
    /// Creates the keyboard and waits until udev has tagged it, as libinput ignores untagged
    /// devices.
    pub fn new() -> Self {
        let file = OpenOptions::new()
            .write(true)
            .open("/dev/uinput")
            .expect("needs a writable /dev/uinput");
        let fd = file.as_raw_fd();
        let mut setup = UinputSetup {
            bustype: 0x06, // BUS_VIRTUAL
            vendor: 0x1234,
            product: 0x5678,
            version: 1,
            name: [0; 80],
            ff_effects_max: 0,
        };
        setup.name[..14].copy_from_slice(b"gst-wd-test-kb");
        let mut sysname = [0u8; 64];
        // Safety: each ioctl gets the argument type the uinput ABI defines for it.
        unsafe {
            assert_eq!(libc::ioctl(fd, UI_SET_EVBIT, EV_KEY as libc::c_int), 0);
            // A full key range, so udev tags it ID_INPUT_KEYBOARD.
            for key in 1..=88 {
                assert_eq!(libc::ioctl(fd, UI_SET_KEYBIT, key as libc::c_int), 0);
            }
            assert_eq!(libc::ioctl(fd, UI_DEV_SETUP, &setup), 0);
            assert_eq!(libc::ioctl(fd, UI_DEV_CREATE), 0);
            assert!(libc::ioctl(fd, UI_GET_SYSNAME_64, sysname.as_mut_ptr()) >= 0);
        }
        let sysname = std::ffi::CStr::from_bytes_until_nul(&sysname)
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();

        let deadline = Instant::now() + Duration::from_secs(5);
        let path = loop {
            let event = std::fs::read_dir(format!("/sys/devices/virtual/input/{sysname}"))
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .find(|name| name.starts_with("event"));
            if let Some(event) = event {
                let dev = std::fs::read_to_string(format!("/sys/class/input/{event}/dev"))
                    .unwrap_or_default();
                let tagged = std::fs::read_to_string(format!("/run/udev/data/c{}", dev.trim()))
                    .is_ok_and(|data| data.contains("E:ID_INPUT=1"));
                let path = format!("/dev/input/{event}");
                if tagged && std::path::Path::new(&path).exists() {
                    break path;
                }
            }
            assert!(Instant::now() < deadline, "udev never set up {sysname}");
            std::thread::sleep(Duration::from_millis(20));
        };
        UinputKeyboard { file, path }
    }

    /// Presses and releases `key`.
    pub fn tap(&self, key: u16) {
        for (kind, code, value) in [
            (EV_KEY, key, 1),
            (EV_SYN, 0, 0),
            (EV_KEY, key, 0),
            (EV_SYN, 0, 0),
        ] {
            let event = InputEvent {
                time: libc::timeval {
                    tv_sec: 0,
                    tv_usec: 0,
                },
                kind,
                code,
                value,
            };
            // Safety: writes one plain-data input_event.
            let written = unsafe {
                libc::write(
                    self.file.as_raw_fd(),
                    &event as *const InputEvent as *const libc::c_void,
                    size_of::<InputEvent>(),
                )
            };
            assert_eq!(written, size_of::<InputEvent>() as isize);
        }
    }
}

impl Drop for UinputKeyboard {
    fn drop(&mut self) {
        // Safety: UI_DEV_DESTROY takes no argument.
        unsafe { libc::ioctl(self.file.as_raw_fd(), UI_DEV_DESTROY) };
    }
}
