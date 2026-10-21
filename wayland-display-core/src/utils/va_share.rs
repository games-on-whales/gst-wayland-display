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
use gst::glib::translate::{FromGlibPtrFull, IntoGlib};
use gst::prelude::*;
use gst_video::ffi::{GstVideoInfoDmaDrm, gst_video_info_set_format};
use gst_video::{VideoFormat, VideoMeta};
use gstreamer_allocators::{DmaBufAllocator, DmaBufAllocatorExtManual, FdMemoryFlags};
use libloading::Library;
use std::ffi::CString;
use std::os::fd::{IntoRawFd, OwnedFd};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

// VASurfaceAttribUsageHint: encoder input.
const VA_USAGE_HINT_ENCODER: u32 = 0x0000_0002;

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

/// Per-source VA display owner. The C helpers may synchronously re-enter set_context,
/// so never hold the lock across a helper call. Each call owns a temporary reference.
#[derive(Default)]
pub struct VaShare {
    display: Mutex<Option<gst::Object>>,
}

impl VaShare {
    fn display_ref(&self) -> *mut c_void {
        self.display
            .lock()
            .unwrap()
            .as_ref()
            .map_or(std::ptr::null_mut(), |d| unsafe {
                gst::glib::gobject_ffi::g_object_ref(d.as_ptr().cast()) as *mut c_void
            })
    }

    unsafe fn adopt(&self, ptr: *mut c_void) {
        if !ptr.is_null() {
            let object = unsafe { gst::Object::from_glib_full(ptr.cast()) };
            let mut slot = self.display.lock().unwrap();
            if slot.is_none() {
                *slot = Some(object);
            }
        }
    }
}

impl VaShare {
    /// The element absorbed a gst-va display context (downstream encoder's display).
    /// Call from `ElementImpl::set_context`. `element`/`context` are the raw GstElement* /
    /// GstContext*.
    pub fn handle_set_context(
        &self,
        element: *mut c_void,
        context: *mut c_void,
        render_path: &str,
    ) {
        let Some(lib) = valib() else { return };
        let Ok(cpath) = CString::new(render_path) else {
            return;
        };
        let mut display = self.display_ref();
        let slot = &mut display;
        unsafe {
            (lib.handle_set_context)(element, context, cpath.as_ptr(), slot);
            tracing::debug!(
                "va_share: handle_set_context display={:#x}",
                (*slot) as usize
            );
            self.adopt(display);
        }
    }

    /// Ensure we have the gst-va display shared with downstream (uses the slot if already
    /// filled by set_context, else queries the encoder's display). Call once the element is
    /// in the pipeline (e.g. `start()`), before buffers are produced.
    pub fn ensure_shared_display(&self, element: *mut c_void, render_path: &str) -> bool {
        let Some(lib) = valib() else { return false };
        let Ok(cpath) = CString::new(render_path) else {
            return false;
        };
        let mut display = self.display_ref();
        let slot = &mut display;
        let ok = unsafe { (lib.ensure_element_data)(element, cpath.as_ptr(), slot) };
        unsafe {
            tracing::debug!(
                "va_share: ensure_shared_display ok={} display={:#x}",
                ok,
                (*slot) as usize
            );
            let ready = ok != 0 && !(*slot).is_null();
            self.adopt(display);
            ready
        }
    }
}

/// Plane layout of our exported NV12 dmabuf.
pub struct Nv12Layout {
    /// NV12 or P010_10LE: the two-plane 4:2:0 formats the converter exports.
    pub format: VideoFormat,
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub y_offset: usize,
    pub y_stride: i32,
    pub uv_offset: usize,
    pub uv_stride: i32,
}

impl VaShare {
    /// Build an NV12/P010 gst buffer over `fd` that the VA encoder can *reuse* (zero per-frame
    /// imports): the memory is from a gst-va dmabuf allocator (so `peek_display` resolves to
    /// the shared display) and a VA surface is attached via `gst_va_dmabuf_memories_setup`.
    /// Returns `None` (caller falls back to a plain dmabuf buffer) if no VA display is shared
    /// or anything fails. Takes ownership of `fd`: the buffer's memory closes it when freed,
    /// and it is closed right away on failure.
    pub fn build_shared_buffer(
        &self,
        fd: OwnedFd,
        size: usize,
        l: &Nv12Layout,
    ) -> Option<GstBuffer> {
        let lib = valib()?;
        let display = self.display.lock().ok()?.clone()?;
        let disp = display.as_ptr() as *mut c_void;
        let ptr = unsafe { (lib.dmabuf_allocator_new)(disp) };
        if ptr.is_null() {
            return None;
        }
        let alloc = unsafe { gst::Allocator::from_glib_full(ptr.cast()) };
        let allocator = alloc.downcast::<DmaBufAllocator>().ok()?;

        let mut buffer = GstBuffer::new();
        let fd = fd.into_raw_fd();
        {
            let b = buffer.get_mut().unwrap();
            let Ok(gmem) =
                (unsafe { allocator.alloc_dmabuf_with_flags(fd, size, FdMemoryFlags::NONE) })
            else {
                unsafe { libc::close(fd) };
                return None;
            };
            b.append_memory(gmem);
            VideoMeta::add_full(
                b,
                gst_video::VideoFrameFlags::empty(),
                l.format,
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
            gst_video_info_set_format(&mut drm.vinfo, l.format.into_glib(), l.width, l.height);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use gst::glib::translate::IntoGlibPtr;

    #[test]
    fn displays_are_isolated_and_released_with_the_owner() {
        gst::init().unwrap();
        let a: gst::Object = gst::Bin::new().upcast();
        let b: gst::Object = gst::Bin::new().upcast();
        let weak_a = a.downgrade();
        let weak_b = b.downgrade();
        let first = VaShare::default();
        let second = VaShare::default();
        unsafe {
            let a_ptr: *mut gst::ffi::GstObject = a.into_glib_ptr();
            let b_ptr: *mut gst::ffi::GstObject = b.into_glib_ptr();
            first.adopt(a_ptr.cast());
            second.adopt(b_ptr.cast());
        }
        assert_ne!(
            first.display.lock().unwrap().as_ref().unwrap().as_ptr(),
            second.display.lock().unwrap().as_ref().unwrap().as_ptr()
        );
        drop(first);
        assert!(weak_a.upgrade().is_none());
        assert!(weak_b.upgrade().is_some());
        drop(second);
        assert!(weak_b.upgrade().is_none());
    }
}
