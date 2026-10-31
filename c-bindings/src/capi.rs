//! C API over `WaylandDisplay`.
//!
//! Safety contract for every function taking a pointer: `dpy`/`wd` is NULL or a pointer returned
//! by `display_init` that has not been passed to `display_finish`; string arguments are NULL or
//! NUL-terminated; output arrays hold at least `max_len` slots. A NULL display is logged and
//! ignored rather than dereferenced, since NULL is `display_init`'s failure return. Calls on
//! one display must not overlap: a caller using it from several threads (input and frames,
//! say) serializes them with its own lock. Separate displays are independent.
#![allow(clippy::missing_safety_doc)] // covered by the module-level contract above

use gst::ffi::GstBuffer;
use gst::glib::translate::FromGlibPtrNone;
use gst_video::ffi::GstVideoInfo;
use gst_video::VideoInfo;
use std::ffi::{c_char, c_uint, c_void, CStr, CString};
use std::ptr;
use waylanddisplaycore::{Tracer, WaylandDisplay};

/// Starts a compositor on `render_node` (NULL: the default node) and returns its display, or
/// NULL if it failed to come up. Calls on the returned display must not overlap; serialize
/// them when using it from several threads.
#[no_mangle]
pub unsafe extern "C" fn display_init(render_node: *const c_char) -> *mut WaylandDisplay {
    let render_node = if !render_node.is_null() {
        Some(
            unsafe { CStr::from_ptr(render_node) }
                .to_string_lossy()
                .into_owned(),
        )
    } else {
        None
    };

    tracing_subscriber::fmt::try_init().ok();

    let mut dpy = match WaylandDisplay::new(render_node) {
        Ok(dpy) => dpy,
        Err(err) => {
            tracing::error!(?err, "Failed to create wayland display.");
            return ptr::null_mut();
        }
    };
    // Like the gst element's start(): only hand out a display whose compositor came up, waiting
    // at most the reply timeout. Dropping one that didn't won't wait out a second timeout.
    if let Err(err) = dpy.env_vars() {
        tracing::error!(?err, "The compositor did not finish initializing.");
        return ptr::null_mut();
    }
    Box::into_raw(Box::new(dpy))
}

/// Borrows the display behind `dpy`, or logs and returns None for NULL.
unsafe fn display_ref<'a>(
    dpy: *mut WaylandDisplay,
    caller: &str,
) -> Option<&'a mut WaylandDisplay> {
    let display = unsafe { dpy.as_mut() };
    if display.is_none() {
        tracing::error!(caller, "Called with a NULL display.");
    }
    display
}

/// Installs the trace callbacks. Passing NULL for either one turns tracing off: `Option` of a
/// function pointer has the same ABI as the bare pointer, with NULL as `None`.
#[no_mangle]
pub unsafe extern "C" fn display_set_trace_fn(
    wd: *mut WaylandDisplay,
    trace_start: Option<extern "C" fn(*const c_char) -> *mut c_void>,
    trace_end: Option<extern "C" fn(*mut c_void)>,
) {
    let Some(display) = (unsafe { display_ref(wd, "display_set_trace_fn") }) else {
        return;
    };
    display.tracer = match (trace_start, trace_end) {
        (Some(trace_start), Some(trace_end)) => Some(Tracer::new(trace_start, trace_end)),
        _ => {
            tracing::warn!("NULL trace callback; tracing disabled.");
            None
        }
    };
}

#[no_mangle]
pub unsafe extern "C" fn display_finish(dpy: *mut WaylandDisplay) {
    if !dpy.is_null() {
        std::mem::drop(unsafe { Box::from_raw(dpy) })
    }
}

/// Writes up to `max_len` pointers from `list` into `out` and returns how many it wrote, so a C
/// caller can iterate exactly the returned count without touching unset slots. The pointers stay
/// owned by the display and are valid until `display_finish`.
fn write_c_strings(list: &[CString], out: *mut *const c_char, max_len: c_uint) -> c_uint {
    let n = list.len().min(max_len as usize);
    if n == 0 || out.is_null() {
        return 0;
    }
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for (slot, string) in out.iter_mut().zip(list) {
        *slot = string.as_ptr();
    }
    n as c_uint
}

#[no_mangle]
pub unsafe extern "C" fn display_get_devices_len(dpy: *mut WaylandDisplay) -> c_uint {
    let Some(display) = (unsafe { display_ref(dpy, "display_get_devices_len") }) else {
        return 0;
    };
    match display.devices.get(display.reply_timeout()) {
        Ok(devices) => devices.len() as c_uint,
        Err(err) => {
            tracing::error!(?err, "Compositor failed before publishing its devices.");
            0
        }
    }
}

/// Fills `devices` with up to `max_len` device paths; returns the number written.
#[no_mangle]
pub unsafe extern "C" fn display_get_devices(
    dpy: *mut WaylandDisplay,
    devices: *mut *const c_char,
    max_len: c_uint,
) -> c_uint {
    let Some(display) = (unsafe { display_ref(dpy, "display_get_devices") }) else {
        return 0;
    };
    let Ok(list) = display.devices.get(display.reply_timeout()) else {
        tracing::error!("Compositor failed before publishing its devices.");
        return 0;
    };
    write_c_strings(list, devices, max_len)
}

#[no_mangle]
pub unsafe extern "C" fn display_get_envvars_len(dpy: *mut WaylandDisplay) -> c_uint {
    let Some(display) = (unsafe { display_ref(dpy, "display_get_envvars_len") }) else {
        return 0;
    };
    match display.envs.get(display.reply_timeout()) {
        Ok(env_vars) => env_vars.len() as c_uint,
        Err(err) => {
            tracing::error!(?err, "Compositor failed before publishing its environment.");
            0
        }
    }
}

/// Fills `env_vars` with up to `max_len` `KEY=VALUE` strings; returns the number written.
#[no_mangle]
pub unsafe extern "C" fn display_get_envvars(
    dpy: *mut WaylandDisplay,
    env_vars: *mut *const c_char,
    max_len: c_uint,
) -> c_uint {
    let Some(display) = (unsafe { display_ref(dpy, "display_get_envvars") }) else {
        return 0;
    };
    let Ok(list) = display.envs.get(display.reply_timeout()) else {
        tracing::error!("Compositor failed before publishing its environment.");
        return 0;
    };
    write_c_strings(list, env_vars, max_len)
}

#[no_mangle]
pub unsafe extern "C" fn display_add_input_device(dpy: *mut WaylandDisplay, path: *const c_char) {
    let Some(display) = (unsafe { display_ref(dpy, "display_add_input_device") }) else {
        return;
    };
    if path.is_null() {
        tracing::error!("Input device path is null");
        return;
    }
    let path = unsafe { CStr::from_ptr(path) }
        .to_string_lossy()
        .into_owned();

    display.add_input_device(path);
}

/// Show `status` on the stream while no application window is mapped (e.g. while the
/// application downloads or starts); it clears once a window is mapped. NULL or an empty
/// string clears it now.
#[no_mangle]
pub unsafe extern "C" fn display_set_status(dpy: *mut WaylandDisplay, status: *const c_char) {
    let Some(display) = (unsafe { display_ref(dpy, "display_set_status") }) else {
        return;
    };
    let status = (!status.is_null()).then(|| {
        unsafe { CStr::from_ptr(status) }
            .to_string_lossy()
            .into_owned()
    });
    display.set_status(status);
}

#[no_mangle]
pub unsafe extern "C" fn display_set_video_info(
    dpy: *mut WaylandDisplay,
    info: *const GstVideoInfo,
) {
    let Some(display) = (unsafe { display_ref(dpy, "display_set_video_info") }) else {
        return;
    };
    if info.is_null() {
        tracing::error!("Video Info is null");
        return;
    }
    let video_info = unsafe { VideoInfo::from_glib_none(info) };

    display.set_video_info(video_info.into());
}

#[no_mangle]
pub unsafe extern "C" fn display_keyboard_input(
    dpy: *mut WaylandDisplay,
    key: c_uint,
    pressed: bool,
) {
    let Some(display) = (unsafe { display_ref(dpy, "display_keyboard_input") }) else {
        return;
    };
    display.keyboard_input(key, pressed);
}

#[no_mangle]
pub unsafe extern "C" fn display_pointer_motion(dpy: *mut WaylandDisplay, x: f64, y: f64) {
    let Some(display) = (unsafe { display_ref(dpy, "display_pointer_motion") }) else {
        return;
    };
    display.pointer_motion(x, y);
}

#[no_mangle]
pub unsafe extern "C" fn display_pointer_motion_absolute(dpy: *mut WaylandDisplay, x: f64, y: f64) {
    let Some(display) = (unsafe { display_ref(dpy, "display_pointer_motion_absolute") }) else {
        return;
    };
    display.pointer_motion_absolute(x, y);
}

#[no_mangle]
pub unsafe extern "C" fn display_pointer_button(
    dpy: *mut WaylandDisplay,
    button: c_uint,
    pressed: bool,
) {
    let Some(display) = (unsafe { display_ref(dpy, "display_pointer_button") }) else {
        return;
    };
    display.pointer_button(button, pressed);
}

#[no_mangle]
pub unsafe extern "C" fn display_pointer_axis(dpy: *mut WaylandDisplay, x: f64, y: f64) {
    let Some(display) = (unsafe { display_ref(dpy, "display_pointer_axis") }) else {
        return;
    };
    display.pointer_axis(x, y);
}

#[no_mangle]
pub unsafe extern "C" fn display_touch_down(
    dpy: *mut WaylandDisplay,
    id: c_uint,
    rel_x: f64,
    rel_y: f64,
) {
    let Some(display) = (unsafe { display_ref(dpy, "display_touch_down") }) else {
        return;
    };
    display.touch_down(id, rel_x, rel_y);
}

#[no_mangle]
pub unsafe extern "C" fn display_touch_up(dpy: *mut WaylandDisplay, id: c_uint) {
    let Some(display) = (unsafe { display_ref(dpy, "display_touch_up") }) else {
        return;
    };
    display.touch_up(id);
}

#[no_mangle]
pub unsafe extern "C" fn display_touch_motion(
    dpy: *mut WaylandDisplay,
    id: c_uint,
    rel_x: f64,
    rel_y: f64,
) {
    let Some(display) = (unsafe { display_ref(dpy, "display_touch_motion") }) else {
        return;
    };
    display.touch_motion(id, rel_x, rel_y);
}

#[no_mangle]
pub unsafe extern "C" fn display_touch_cancel(dpy: *mut WaylandDisplay) {
    let Some(display) = (unsafe { display_ref(dpy, "display_touch_cancel") }) else {
        return;
    };
    display.touch_cancel();
}

#[no_mangle]
pub unsafe extern "C" fn display_touch_frame(dpy: *mut WaylandDisplay) {
    let Some(display) = (unsafe { display_ref(dpy, "display_touch_frame") }) else {
        return;
    };
    display.touch_frame();
}

#[no_mangle]
pub unsafe extern "C" fn display_get_frame(dpy: *mut WaylandDisplay) -> *mut GstBuffer {
    let Some(display) = (unsafe { display_ref(dpy, "display_get_frame") }) else {
        return ptr::null_mut();
    };
    let _span = display
        .tracer
        .as_ref()
        .map(|tracer| tracer.trace("display_get_frame"));
    match display.frame() {
        Ok(mut frame) => {
            let ptr = frame.make_mut().as_mut_ptr();
            std::mem::forget(frame);
            ptr
        }
        Err(err) => {
            tracing::error!("Rendering error: {}", err);
            ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list() -> Vec<CString> {
        vec![CString::new("WAYLAND_DISPLAY=wayland-1").unwrap()]
    }

    #[test]
    fn larger_buffer_returns_entries_written() {
        let list = list();
        let mut buf = [ptr::null::<c_char>(); 8];
        assert_eq!(write_c_strings(&list, buf.as_mut_ptr(), 8), 1);
        assert_eq!(buf[0], list[0].as_ptr());
        assert!(buf[1..].iter().all(|p| p.is_null()));
    }

    #[test]
    fn smaller_buffer_is_not_overrun() {
        let list = vec![CString::new("a").unwrap(), CString::new("b").unwrap()];
        let mut buf = [ptr::null::<c_char>(); 2];
        assert_eq!(write_c_strings(&list, buf.as_mut_ptr(), 1), 1);
        assert_eq!(buf[0], list[0].as_ptr());
        assert!(buf[1].is_null());
    }

    #[test]
    fn null_or_empty_buffer_writes_nothing() {
        assert_eq!(write_c_strings(&list(), ptr::null_mut(), 4), 0);
        assert_eq!(write_c_strings(&list(), ptr::null_mut(), 0), 0);
    }

    extern "C" fn trace_start(_name: *const c_char) -> *mut c_void {
        ptr::null_mut()
    }

    extern "C" fn trace_end(_ctx: *mut c_void) {}

    // A C caller passing NULL callbacks must turn tracing off, not store a NULL function pointer
    // that the next frame would call.
    #[test]
    fn null_trace_callbacks_disable_tracing() {
        unsafe {
            let dpy = display_init(c"software".as_ptr());
            assert!(!dpy.is_null());
            display_set_trace_fn(dpy, Some(trace_start), Some(trace_end));
            assert!((*dpy).tracer.is_some());
            display_set_trace_fn(dpy, Some(trace_start), None);
            assert!((*dpy).tracer.is_none());
            display_set_trace_fn(dpy, Some(trace_start), Some(trace_end));
            display_set_trace_fn(dpy, None, Some(trace_end));
            assert!((*dpy).tracer.is_none());
            display_finish(dpy);
        }
    }

    // display_init must report a compositor that can't start as NULL, not a display whose every
    // call then fails, and a started one as usable right away.
    #[test]
    fn display_init_returns_a_started_display_or_null() {
        unsafe {
            assert!(display_init(c"/nonexistent/renderD128".as_ptr()).is_null());
            let dpy = display_init(c"software".as_ptr());
            assert!(!dpy.is_null());
            let mut envs = [ptr::null::<c_char>(); 4];
            assert_eq!(display_get_envvars(dpy, envs.as_mut_ptr(), 4), 1);
            let env = CStr::from_ptr(envs[0]).to_str().unwrap();
            assert!(env.starts_with("WAYLAND_DISPLAY="), "{env}");
            display_finish(dpy);
        }
    }

    #[test]
    fn null_display_is_ignored() {
        let dpy = ptr::null_mut();
        let mut buf = [ptr::null::<c_char>(); 4];
        unsafe {
            assert_eq!(display_get_devices_len(dpy), 0);
            assert_eq!(display_get_devices(dpy, buf.as_mut_ptr(), 4), 0);
            assert_eq!(display_get_envvars_len(dpy), 0);
            assert_eq!(display_get_envvars(dpy, buf.as_mut_ptr(), 4), 0);
            assert!(display_get_frame(dpy).is_null());
            display_keyboard_input(dpy, 30, true);
            display_add_input_device(dpy, c"/dev/input/event0".as_ptr());
            display_set_trace_fn(dpy, None, None);
            display_finish(dpy);
        }
        assert!(buf.iter().all(|p| p.is_null()));
    }
}
