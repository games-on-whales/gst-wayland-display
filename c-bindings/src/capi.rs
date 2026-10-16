use gst::ffi::GstBuffer;
use gst::glib::translate::FromGlibPtrNone;
use gst_video::ffi::GstVideoInfo;
use gst_video::VideoInfo;
use std::ffi::{c_char, c_uint, c_void, CStr, CString};
use std::ptr;
use tracing_subscriber;
use waylanddisplaycore::{Tracer, WaylandDisplay};

#[no_mangle]
pub extern "C" fn display_init(render_node: *const c_char) -> *mut WaylandDisplay {
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

    match WaylandDisplay::new(render_node) {
        Ok(dpy) => Box::into_raw(Box::new(dpy)),
        Err(err) => {
            tracing::error!(?err, "Failed to create wayland display.");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn display_set_trace_fn(
    wd: *mut WaylandDisplay,
    trace_start: extern "C" fn(*const c_char) -> *mut c_void,
    trace_end: extern "C" fn(*mut c_void),
) {
    let display = unsafe { &mut *wd };
    display.tracer = Some(Tracer::new(trace_start, trace_end));
}

#[no_mangle]
pub extern "C" fn display_finish(dpy: *mut WaylandDisplay) {
    std::mem::drop(unsafe { Box::from_raw(dpy) })
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
pub extern "C" fn display_get_devices_len(dpy: *mut WaylandDisplay) -> c_uint {
    let display = unsafe { &mut *dpy };
    match display.devices.get() {
        Ok(devices) => devices.len() as c_uint,
        Err(err) => {
            tracing::error!(?err, "Compositor failed before publishing its devices.");
            0
        }
    }
}

/// Fills `devices` with up to `max_len` device paths; returns the number written.
#[no_mangle]
pub extern "C" fn display_get_devices(
    dpy: *mut WaylandDisplay,
    devices: *mut *const c_char,
    max_len: c_uint,
) -> c_uint {
    let display = unsafe { &mut *dpy };
    let Ok(list) = display.devices.get() else {
        tracing::error!("Compositor failed before publishing its devices.");
        return 0;
    };
    write_c_strings(list, devices, max_len)
}

#[no_mangle]
pub extern "C" fn display_get_envvars_len(dpy: *mut WaylandDisplay) -> c_uint {
    let display = unsafe { &mut *dpy };
    match display.envs.get() {
        Ok(env_vars) => env_vars.len() as c_uint,
        Err(err) => {
            tracing::error!(?err, "Compositor failed before publishing its environment.");
            0
        }
    }
}

/// Fills `env_vars` with up to `max_len` `KEY=VALUE` strings; returns the number written.
#[no_mangle]
pub extern "C" fn display_get_envvars(
    dpy: *mut WaylandDisplay,
    env_vars: *mut *const c_char,
    max_len: c_uint,
) -> c_uint {
    let display = unsafe { &mut *dpy };
    let Ok(list) = display.envs.get() else {
        tracing::error!("Compositor failed before publishing its environment.");
        return 0;
    };
    write_c_strings(list, env_vars, max_len)
}

#[no_mangle]
pub extern "C" fn display_add_input_device(dpy: *mut WaylandDisplay, path: *const c_char) {
    let display = unsafe { &mut *dpy };
    let path = unsafe { CStr::from_ptr(path) }
        .to_string_lossy()
        .into_owned();

    display.add_input_device(path);
}

#[no_mangle]
pub extern "C" fn display_set_video_info(dpy: *mut WaylandDisplay, info: *const GstVideoInfo) {
    let display = unsafe { &mut *dpy };
    if info.is_null() {
        tracing::error!("Video Info is null");
    }
    let video_info = unsafe { VideoInfo::from_glib_none(info) };

    display.set_video_info(video_info.into());
}

#[no_mangle]
pub extern "C" fn display_keyboard_input(dpy: *mut WaylandDisplay, key: c_uint, pressed: bool) {
    let display = unsafe { &mut *dpy };
    display.keyboard_input(key, pressed);
}

#[no_mangle]
pub extern "C" fn display_pointer_motion(dpy: *mut WaylandDisplay, x: f64, y: f64) {
    let display = unsafe { &mut *dpy };
    display.pointer_motion(x, y);
}

#[no_mangle]
pub extern "C" fn display_pointer_motion_absolute(dpy: *mut WaylandDisplay, x: f64, y: f64) {
    let display = unsafe { &mut *dpy };
    display.pointer_motion_absolute(x, y);
}

#[no_mangle]
pub extern "C" fn display_pointer_button(dpy: *mut WaylandDisplay, button: c_uint, pressed: bool) {
    let display = unsafe { &mut *dpy };
    display.pointer_button(button, pressed);
}

#[no_mangle]
pub extern "C" fn display_pointer_axis(dpy: *mut WaylandDisplay, x: f64, y: f64) {
    let display = unsafe { &mut *dpy };
    display.pointer_axis(x, y);
}

#[no_mangle]
pub extern "C" fn display_touch_down(dpy: *mut WaylandDisplay, id: c_uint, rel_x: f64, rel_y: f64) {
    let display = unsafe { &mut *dpy };
    display.touch_down(id, rel_x, rel_y);
}

#[no_mangle]
pub extern "C" fn display_touch_up(dpy: *mut WaylandDisplay, id: c_uint) {
    let display = unsafe { &mut *dpy };
    display.touch_up(id);
}

#[no_mangle]
pub extern "C" fn display_touch_motion(
    dpy: *mut WaylandDisplay,
    id: c_uint,
    rel_x: f64,
    rel_y: f64,
) {
    let display = unsafe { &mut *dpy };
    display.touch_motion(id, rel_x, rel_y);
}

#[no_mangle]
pub extern "C" fn display_touch_cancel(dpy: *mut WaylandDisplay) {
    let display = unsafe { &mut *dpy };
    display.touch_cancel();
}

#[no_mangle]
pub extern "C" fn display_touch_frame(dpy: *mut WaylandDisplay) {
    let display = unsafe { &mut *dpy };
    display.touch_frame();
}

#[no_mangle]
pub extern "C" fn display_get_frame(dpy: *mut WaylandDisplay) -> *mut GstBuffer {
    let display = unsafe { &mut *dpy };
    let _span = match display.tracer.as_ref() {
        Some(tracer) => Some(tracer.trace("display_get_frame")),
        None => None,
    };
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
}
