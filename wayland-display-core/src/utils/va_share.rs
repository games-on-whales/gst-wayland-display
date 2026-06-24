//! Make our exported NV12 dmabuf a *shareable* VA surface.
//!
//! When the VA encoder receives a raw dmabuf it `vaCreateSurfaces`-imports a brand-new
//! VA surface **every frame** and radeonsi-VA doesn't free them, so the per-context
//! surface budget runs out and the encoder's reconstruct pool starves after ~4 frames.
//! gst-va elements avoid this by passing each other buffers that already carry a
//! `GstVaSurface` (qdata) on a shared `GstVaDisplay`; the encoder then reuses that one
//! surface (zero imports).
//!
//! This module dynamically loads `libgstva-1.0` (so a missing lib is a runtime, not a
//! link, error, matching the cuda path) and calls `gst_va_dmabuf_memories_setup` to
//! attach a VA surface to one of our cached ring buffers. Best-effort: on any failure
//! the buffer stays a plain dmabuf (the previous behaviour).

use gst::Buffer as GstBuffer;
use gst::ffi::GstMemory;
use gst::glib::translate::FromGlibPtrFull;
use gst::prelude::*;
use gst_video::ffi::{GstVideoFormat, GstVideoInfoDmaDrm, gst_video_info_set_format};
use gst_video::{VideoFormat, VideoMeta};
use gstreamer_allocators::{DmaBufAllocator, DmaBufAllocatorExtManual, FdMemoryFlags};
use libloading::Library;
use std::ffi::CString;
use std::os::fd::RawFd;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

// VASurfaceAttribUsageHint: encoder input.
const VA_USAGE_HINT_ENCODER: u32 = 0x0000_0002;
const GST_VIDEO_FORMAT_NV12: GstVideoFormat = 23; // GST_VIDEO_FORMAT_NV12

type DmabufMemoriesSetup = unsafe extern "C" fn(
    *mut c_void,             // GstVaDisplay *
    *mut GstVideoInfoDmaDrm, // drm_info
    *mut *mut GstMemory,     // mem[GST_VIDEO_MAX_PLANES]
    *mut usize,              // fds[GST_VIDEO_MAX_PLANES]
    *mut usize,              // offset[GST_VIDEO_MAX_PLANES]
    u32,                     // usage_hint
) -> c_int;
type EnsureElementData =
    unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> c_int;
type HandleSetContext =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_char, *mut *mut c_void) -> c_int;
type DmabufAllocatorNew = unsafe extern "C" fn(*mut c_void) -> *mut c_void;

struct VaLib {
    _lib: Library,
    dmabuf_memories_setup: DmabufMemoriesSetup,
    ensure_element_data: EnsureElementData,
    handle_set_context: HandleSetContext,
    dmabuf_allocator_new: DmabufAllocatorNew,
}

fn valib() -> Option<&'static VaLib> {
    static LIB: OnceLock<Option<VaLib>> = OnceLock::new();
    LIB.get_or_init(|| match unsafe { load_valib() } {
        Ok(lib) => Some(lib),
        Err(e) => {
            // Not fatal: NV12 buffers stay plain dmabufs that the encoder re-imports each
            // frame (the previous behaviour) -- but that starves radeonsi-VA's surface
            // budget, so make the cause visible rather than silently degrading.
            tracing::warn!(
                "va_share: VA surface sharing disabled ({e}); NV12 buffers fall back to \
                 per-frame dmabuf import"
            );
            None
        }
    })
    .as_ref()
}

unsafe fn load_valib() -> Result<VaLib, Box<dyn std::error::Error>> {
    unsafe {
        let lib = ["libgstva-1.0.so.0", "libgstva-1.0.so"]
            .iter()
            .find_map(|n| Library::new(n).ok())
            .ok_or("libgstva-1.0 not found")?;
        let dmabuf_memories_setup =
            *lib.get::<DmabufMemoriesSetup>(b"gst_va_dmabuf_memories_setup\0")?;
        let ensure_element_data = *lib.get::<EnsureElementData>(b"gst_va_ensure_element_data\0")?;
        let handle_set_context = *lib.get::<HandleSetContext>(b"gst_va_handle_set_context\0")?;
        let dmabuf_allocator_new =
            *lib.get::<DmabufAllocatorNew>(b"gst_va_dmabuf_allocator_new\0")?;
        Ok(VaLib {
            _lib: lib,
            dmabuf_memories_setup,
            ensure_element_data,
            handle_set_context,
            dmabuf_allocator_new,
        })
    }
}

/// Persistent `GstVaDisplay *` slot, shared by `set_context` and `ensure_element_data`
/// exactly like a gst-va element's `self->display`: gst-va coordinates through this one
/// storage (set_context fills it with the encoder's display; ensure keeps it if already
/// set). Surfaces must be attached on the *same* display the encoder uses. Leaked for
/// process life; a stable `GstVaDisplay **` for the C side.
fn display_slot() -> *mut *mut c_void {
    static SLOT: OnceLock<usize> = OnceLock::new();
    *SLOT.get_or_init(|| {
        Box::leak(Box::new(std::ptr::null_mut::<c_void>())) as *mut *mut c_void as usize
    }) as *mut *mut c_void
}

/// The element absorbed a gst-va display context (downstream encoder's display).
/// Call from `ElementImpl::set_context`. `element`/`context` are the raw GstElement* /
/// GstContext*.
pub fn handle_set_context(element: *mut c_void, context: *mut c_void, render_path: &str) {
    let Some(lib) = valib() else { return };
    let Ok(cpath) = CString::new(render_path) else {
        return;
    };
    let slot = display_slot();
    unsafe {
        (lib.handle_set_context)(element, context, cpath.as_ptr(), slot);
        tracing::debug!(
            "va_share: handle_set_context display={:#x}",
            (*slot) as usize
        );
    }
}

/// Ensure we have the gst-va display shared with downstream (uses the slot if already
/// filled by set_context, else queries the encoder's display). Call once the element is
/// in the pipeline (e.g. `start()`), before buffers are produced.
pub fn ensure_shared_display(element: *mut c_void, render_path: &str) -> bool {
    let Some(lib) = valib() else { return false };
    let Ok(cpath) = CString::new(render_path) else {
        return false;
    };
    let slot = display_slot();
    let ok = unsafe { (lib.ensure_element_data)(element, cpath.as_ptr(), slot) };
    unsafe {
        tracing::debug!(
            "va_share: ensure_shared_display ok={} display={:#x}",
            ok,
            (*slot) as usize
        );
        ok != 0 && !(*slot).is_null()
    }
}

/// Plane layout of our exported NV12 dmabuf.
pub struct Nv12Layout {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub y_offset: usize,
    pub y_stride: i32,
    pub uv_offset: usize,
    pub uv_stride: i32,
}

/// The shared (encoder) display, if known. `None` means no VA encoder downstream yet.
fn shared_display() -> Option<*mut c_void> {
    let p = unsafe { std::ptr::read_volatile(display_slot()) };
    (!p.is_null()).then_some(p)
}

/// A gst-va dmabuf allocator bound to the shared display (cached). Crucially this is a
/// `GstDmaBufAllocator` subclass, so we can wrap our exported fd with it AND the encoder's
/// `gst_va_memory_peek_display` then returns the display (it reads the *allocator's*
/// display, not the surface qdata) -- which is what its reuse check requires.
fn va_dmabuf_allocator(display: *mut c_void) -> Option<DmaBufAllocator> {
    static CACHE: OnceLock<Mutex<Option<(usize, DmaBufAllocator)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().ok()?;
    if let Some((d, a)) = guard.as_ref() {
        if *d == display as usize {
            return Some(a.clone());
        }
    }
    let lib = valib()?;
    let ptr = unsafe { (lib.dmabuf_allocator_new)(display) };
    if ptr.is_null() {
        return None;
    }
    let alloc: gst::Allocator =
        unsafe { gst::Allocator::from_glib_full(ptr as *mut gst::ffi::GstAllocator) };
    let dmabuf = alloc.downcast::<DmaBufAllocator>().ok()?;
    *guard = Some((display as usize, dmabuf.clone()));
    Some(dmabuf)
}

/// Build an NV12 gst buffer over `fd` that the VA encoder can *reuse* (zero per-frame
/// imports): the memory is from a gst-va dmabuf allocator (so `peek_display` resolves to
/// the shared display) and a VA surface is attached via `gst_va_dmabuf_memories_setup`.
/// Returns `None` (caller falls back to a plain dmabuf buffer) if no VA display is shared
/// or anything fails.
pub fn build_shared_buffer(fd: RawFd, size: usize, l: &Nv12Layout) -> Option<GstBuffer> {
    let lib = valib()?;
    let disp = shared_display()?;
    let allocator = va_dmabuf_allocator(disp)?;

    let mut buffer = GstBuffer::new();
    {
        let b = buffer.get_mut().unwrap();
        let gmem =
            unsafe { allocator.alloc_dmabuf_with_flags(fd, size, FdMemoryFlags::DONT_CLOSE) }
                .ok()?;
        b.append_memory(gmem);
        VideoMeta::add_full(
            b,
            gst_video::VideoFrameFlags::empty(),
            VideoFormat::Nv12,
            l.width,
            l.height,
            &[l.y_offset, l.uv_offset],
            &[l.y_stride, l.uv_stride],
        )
        .ok()?;
    }

    let mem = buffer.peek_memory(0).as_ptr() as *mut GstMemory;
    let mut drm: GstVideoInfoDmaDrm = unsafe { std::mem::zeroed() };
    unsafe {
        gst_video_info_set_format(&mut drm.vinfo, GST_VIDEO_FORMAT_NV12, l.width, l.height);
    }
    drm.vinfo.stride[0] = l.y_stride;
    drm.vinfo.stride[1] = l.uv_stride;
    drm.vinfo.offset[0] = l.y_offset;
    drm.vinfo.offset[1] = l.uv_offset;
    drm.drm_fourcc = l.fourcc;
    drm.drm_modifier = l.modifier;

    let mut mems: [*mut GstMemory; 4] = [mem, mem, std::ptr::null_mut(), std::ptr::null_mut()];
    let mut fds: [usize; 4] = [fd as usize, fd as usize, 0, 0];
    let mut offs: [usize; 4] = [l.y_offset, l.uv_offset, 0, 0];
    let ok = unsafe {
        (lib.dmabuf_memories_setup)(
            disp,
            &mut drm,
            mems.as_mut_ptr(),
            fds.as_mut_ptr(),
            offs.as_mut_ptr(),
            VA_USAGE_HINT_ENCODER,
        )
    };
    tracing::debug!(
        "va_share: build_shared_buffer display={:#x} setup_ok={ok}",
        disp as usize
    );
    if ok == 0 {
        return None;
    }
    Some(buffer)
}
