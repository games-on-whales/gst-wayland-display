//! Ask the VA driver which NV12 DRM modifier its encoder imports.
//!
//! The encoder advertises exactly one NV12 modifier per render node: the one
//! `gst_va_dmabuf_get_modifier_for_format` returns (gst-va creates a surface, exports it,
//! and reads back `drm_format_modifier`). Our source has to export that same modifier for
//! a direct `! vah265enc` to negotiate without a re-import. Behind interpipe there's no
//! encoder to negotiate against, so we query the driver out of band instead of guessing.
//!
//! Loads the same `libgstva-1.0` as [`super::va_share`] and calls two public symbols, so a
//! missing lib is a runtime warning, not a link error. Best-effort: any failure returns
//! `None` and the caller treats the encoder's modifier as unknown.

use libloading::Library;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

const GST_VIDEO_FORMAT_NV12: c_int = 23;
// VASurfaceAttribUsageHint: encoder input (matches va_share's export hint).
const VA_USAGE_HINT_ENCODER: c_uint = 0x0000_0002;
/// `DRM_FORMAT_MOD_INVALID` -- gst-va's "driver has no modifier for this format".
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

type DisplayDrmNewFromPath = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type ModifierForFormat = unsafe extern "C" fn(*mut c_void, c_int, c_uint) -> u64;

struct QueryLib {
    _lib: Library,
    display_new: DisplayDrmNewFromPath,
    modifier_for_format: ModifierForFormat,
}

fn querylib() -> Option<&'static QueryLib> {
    static LIB: OnceLock<Option<QueryLib>> = OnceLock::new();
    LIB.get_or_init(|| match unsafe { load() } {
        Ok(l) => Some(l),
        Err(e) => {
            tracing::warn!("va_query: encoder modifier probe disabled ({e})");
            None
        }
    })
    .as_ref()
}

unsafe fn load() -> Result<QueryLib, Box<dyn std::error::Error>> {
    unsafe {
        let lib = ["libgstva-1.0.so.0", "libgstva-1.0.so"]
            .iter()
            .find_map(|n| Library::new(n).ok())
            .ok_or("libgstva-1.0 not found")?;
        let display_new =
            *lib.get::<DisplayDrmNewFromPath>(b"gst_va_display_drm_new_from_path\0")?;
        let modifier_for_format =
            *lib.get::<ModifierForFormat>(b"gst_va_dmabuf_get_modifier_for_format\0")?;
        Ok(QueryLib {
            _lib: lib,
            display_new,
            modifier_for_format,
        })
    }
}

/// The NV12 DRM modifier the VA encoder on `render_node` imports (its preferred encode-input
/// layout), or `None` if libgstva is missing, the node has no VA driver, or the driver
/// reports no modifier. `Some(0)` means LINEAR. Queried once per node and cached.
pub fn import_nv12_modifier(render_node: &str) -> Option<u64> {
    static CACHE: OnceLock<Mutex<Vec<(String, Option<u64>)>>> = OnceLock::new();
    let mut guard = CACHE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    if let Some((_, m)) = guard.iter().find(|(n, _)| n == render_node) {
        return *m;
    }
    let m = unsafe { query(render_node) };
    guard.push((render_node.to_string(), m));
    m
}

unsafe fn query(render_node: &str) -> Option<u64> {
    let lib = querylib()?;
    let path = CString::new(render_node).ok()?;
    unsafe {
        let display = (lib.display_new)(path.as_ptr());
        if display.is_null() {
            tracing::warn!("va_query: no VA display on {render_node}");
            return None;
        }
        let modifier =
            (lib.modifier_for_format)(display, GST_VIDEO_FORMAT_NV12, VA_USAGE_HINT_ENCODER);
        // gst_va_display_drm_new_from_path returns a full ref (GstObject).
        gst::glib::gobject_ffi::g_object_unref(display as *mut _);
        if modifier == DRM_FORMAT_MOD_INVALID {
            tracing::warn!("va_query: {render_node} VA encoder reports no NV12 modifier");
            return None;
        }
        tracing::debug!("va_query: {render_node} VA encoder imports NV12 modifier {modifier:#x}");
        Some(modifier)
    }
}
