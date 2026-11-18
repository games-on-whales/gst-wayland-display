//! A process that made an EGL display exits without a thread-local-storage panic, even when
//! EGL reports through its debug callback after `exit()` has destroyed the thread locals (as
//! NVIDIA's libEGL does when it terminates the never-terminated display at exit).

use crate::utils::renderer::setup_renderer;
use smithay::backend::egl::ffi;

const CHILD: &str = "GWD_EGL_EXIT_CHILD";

/// An EGL call that fails, so EGL reports `EGL_BAD_DISPLAY` through the debug callback.
fn egl_error() {
    unsafe { ffi::egl::QueryString(ffi::egl::NO_DISPLAY, ffi::egl::VENDOR as i32) };
}

extern "C" fn egl_error_at_exit() {
    egl_error();
}

#[test]
fn exit_after_egl_reports_no_thread_local_panic() {
    if std::env::var_os(CHILD).is_some() {
        tracing_subscriber::fmt::try_init().ok();
        // Registered before the display exists, so it runs after any exit handler the
        // display registers, standing in for the driver's own.
        unsafe { libc::atexit(egl_error_at_exit) };
        let _renderer = setup_renderer(None);
        egl_error(); // reported while the thread locals are alive
        std::process::exit(0);
    }
    let name = module_path!().split_once("::").unwrap().1;
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            &format!("{name}::exit_after_egl_reports_no_thread_local_panic"),
            "--exact",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .output()
        .expect("run the child");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "{text}");
    assert!(
        text.contains("[EGL]"),
        "EGL reports errors through the debug callback here: {text}"
    );
    assert!(!text.contains("Thread Local Storage"), "{text}");
}
