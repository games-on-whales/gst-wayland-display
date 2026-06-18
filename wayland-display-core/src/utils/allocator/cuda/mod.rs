pub use ffi::GstCudaContext;
use ffi::{CUgraphicsResource, PFN_eglDestroyImageKHR};
use gst::glib::ffi as glib_ffi;
use gst::glib::translate::ToGlibPtr;
use gst::query::Allocation;
use gst::{Buffer as GstBuffer, Context, Element, QueryRef};
use gst_video::{VideoInfoDmaDrm, VideoMeta};
use smithay::backend::allocator::Buffer;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::egl;
use smithay::backend::egl::ffi::egl::types::{EGLDisplay, EGLImageKHR, EGLint};
use std::ffi::c_int;
use std::os::fd::AsRawFd;
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

mod ffi;

// Helper to load EGL extension functions
#[derive(Debug, Clone)]
pub struct EglExtensions {
    pub create_image: ffi::PFN_eglCreateImageKHR,
    pub destroy_image: PFN_eglDestroyImageKHR,
}

impl EglExtensions {
    unsafe fn load() -> Option<Self> {
        let create_image_ptr = unsafe { egl::get_proc_address("eglCreateImageKHR") };
        let destroy_image_ptr = unsafe { egl::get_proc_address("eglDestroyImageKHR") };

        if create_image_ptr.is_null() || destroy_image_ptr.is_null() {
            return None;
        }

        Some(EglExtensions {
            create_image: unsafe { std::mem::transmute(create_image_ptr) },
            destroy_image: unsafe { std::mem::transmute(destroy_image_ptr) },
        })
    }

    pub fn new() -> Option<Self> {
        unsafe { EglExtensions::load() }
    }
}

#[derive(Debug)]
pub struct EGLImage {
    image: EGLImageKHR,
    egl_display: Arc<EGLDisplay>,
    egl_extensions: EglExtensions,
}

impl EGLImage {
    pub fn from(dmabuf: &Dmabuf, egl_display: &EGLDisplay) -> Result<Self, String> {
        // Get dmabuf properties
        let width = dmabuf.width();
        let height = dmabuf.height();
        let fourcc = dmabuf.format().code as u32;

        // Get modifier if available
        let modifier: u64 = dmabuf.format().modifier.into();
        let modifier_lo = (modifier & 0xFFFFFFFF) as EGLint;
        let modifier_hi = ((modifier >> 32) & 0xFFFFFFFF) as EGLint;

        // Build EGL attribute list for DMA-BUF import
        let mut attribs = [
            ffi::EGL_WIDTH,
            width as EGLint,
            ffi::EGL_HEIGHT,
            height as EGLint,
            ffi::EGL_LINUX_DRM_FOURCC_EXT,
            fourcc as EGLint,
        ]
        .to_vec();

        let offsets = dmabuf.offsets().map(|o| o as usize).collect::<Vec<_>>();

        let strides = dmabuf.strides().map(|s| s as i32).collect::<Vec<_>>();

        for (idx, handle) in dmabuf.handles().enumerate() {
            let fd = handle.as_raw_fd();
            // Add to attribs the current plane data
            if idx == 0 {
                attribs.extend_from_slice(&[
                    ffi::EGL_DMA_BUF_PLANE0_FD_EXT,
                    fd,
                    ffi::EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                    offsets[idx] as EGLint,
                    ffi::EGL_DMA_BUF_PLANE0_PITCH_EXT,
                    strides[idx],
                    ffi::EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                    modifier_lo,
                    ffi::EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                    modifier_hi,
                ]);
            } else if idx == 1 {
                attribs.extend_from_slice(&[
                    ffi::EGL_DMA_BUF_PLANE1_FD_EXT,
                    fd,
                    ffi::EGL_DMA_BUF_PLANE1_OFFSET_EXT,
                    offsets[idx] as EGLint,
                    ffi::EGL_DMA_BUF_PLANE1_PITCH_EXT,
                    strides[idx],
                    ffi::EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
                    modifier_lo,
                    ffi::EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
                    modifier_hi,
                ]);
            }
        }

        attribs.push(ffi::EGL_NONE);
        let egl_extensions = EglExtensions::new().ok_or("Failed to load EGL extensions")?;

        let egl_image = unsafe {
            (egl_extensions.create_image)(
                *egl_display,
                ptr::null_mut(),
                ffi::EGL_LINUX_DMA_BUF_EXT,
                ptr::null_mut(),
                attribs.as_ptr(),
            )
        };
        if egl_image != ffi::EGL_NO_IMAGE_KHR {
            Ok(EGLImage {
                image: egl_image,
                egl_display: Arc::new(egl_display.clone()),
                egl_extensions,
            })
        } else {
            Err("Failed to create EGLImage".into())
        }
    }
}

impl Drop for EGLImage {
    fn drop(&mut self) {
        unsafe {
            (self.egl_extensions.destroy_image)(*self.egl_display, self.image);
        }
    }
}

/// Convert an already-filled external dmabuf (e.g. the NV12 buffer the compositor
/// produced) into a CUDAMemory gst buffer, importing it via EGLImage and copying
/// each plane on-GPU. This is the "convert to CUDA late" step for the Nvidia encode
/// branch: the dmabuf flows through the system unchanged across all vendors, and
/// only the Nvidia consumer maps it into CUDA right before the encoder. Works for
/// any plane layout EGLImage::from handles (NV12 today, P010 for 10-bit/HDR).
pub fn external_dmabuf_to_cuda_buffer(
    dmabuf: &Dmabuf,
    video_info: VideoInfoDmaDrm,
    egl_display: &EGLDisplay,
    cuda_context: &CUDAContext,
    buffer_pool: Option<&CUDABufferPool>,
) -> Result<GstBuffer, Box<dyn std::error::Error>> {
    let egl_image = EGLImage::from(dmabuf, egl_display)?;
    let cuda_image = CUDAImage::from(egl_image, cuda_context)?;
    cuda_image.to_gst_buffer(video_info, cuda_context, buffer_pool)
}

/// Owns an EGLDisplay and turns incoming NV12 DMABuf gst buffers into CUDAMemory
/// gst buffers - the "convert to CUDA late" step for the Nvidia encode branch. This
/// keeps all smithay/EGL handling in the core so the gst plugin only deals with gst
/// and CUDA types.
pub struct CudaUploader {
    egl: std::sync::Arc<smithay::backend::egl::EGLDisplay>,
}

// The EGLDisplay is Send+Sync (it lives in the shared EGL_DISPLAYS cache).
unsafe impl Send for CudaUploader {}

impl CudaUploader {
    /// Create an uploader bound to the given render node (or auto-selected when None).
    pub fn new(render_node_path: Option<&str>) -> Self {
        let node = render_node_path
            .and_then(|p| smithay::backend::drm::DrmNode::from_path(p).ok());
        CudaUploader {
            egl: crate::utils::renderer::setup_egl_display(node),
        }
    }

    /// Reconstruct the incoming NV12 dmabuf gst buffer into a single 2-plane Dmabuf
    /// and convert it to a CUDAMemory gst buffer.
    pub fn upload(
        &self,
        inbuf: &gst::BufferRef,
        in_info: &VideoInfoDmaDrm,
        cuda_context: &CUDAContext,
        buffer_pool: Option<&CUDABufferPool>,
    ) -> Result<GstBuffer, Box<dyn std::error::Error>> {
        let dmabuf =
            reconstruct_nv12_dmabuf(inbuf, in_info).ok_or("failed to reconstruct NV12 dmabuf")?;
        let raw_display = self.egl.get_display_handle().handle;
        external_dmabuf_to_cuda_buffer(
            &dmabuf,
            in_info.clone(),
            &raw_display,
            cuda_context,
            buffer_pool,
        )
    }
}

/// Rebuild a single 2-plane NV12 Dmabuf from a gst buffer's dmabuf memories (the
/// inverse of Nv12Target::to_gst_buffer). Handles both the 2-memory layout the
/// compositor produces (one plane per fd) and a single contiguous memory.
fn reconstruct_nv12_dmabuf(inbuf: &gst::BufferRef, in_info: &VideoInfoDmaDrm) -> Option<Dmabuf> {
    use smithay::backend::allocator::dmabuf::DmabufFlags;
    use smithay::reexports::drm::buffer::DrmFourcc;
    use smithay::reexports::gbm::Modifier;
    use std::os::fd::BorrowedFd;

    let width = in_info.width();
    let height = in_info.height();
    let modifier = Modifier::from(in_info.modifier());
    let vmeta = inbuf.meta::<VideoMeta>();
    let stride = |plane: usize| -> u32 {
        vmeta
            .as_ref()
            .map(|m| m.stride()[plane] as u32)
            .unwrap_or(width)
    };
    let offset = |plane: usize| -> u32 {
        vmeta.as_ref().map(|m| m.offset()[plane] as u32).unwrap_or(0)
    };

    let n_mem = inbuf.n_memory();
    let mut builder = Dmabuf::builder(
        (width as i32, height as i32),
        DrmFourcc::Nv12,
        modifier,
        DmabufFlags::empty(),
    );
    for plane in 0..2usize {
        let (mem, off) = if n_mem >= 2 {
            (inbuf.peek_memory(plane), 0u32)
        } else {
            (inbuf.peek_memory(0), offset(plane))
        };
        let raw_fd =
            unsafe { gstreamer_allocators::ffi::gst_dmabuf_memory_get_fd(mem.as_ptr() as *mut _) };
        if raw_fd < 0 {
            return None;
        }
        let owned = unsafe { BorrowedFd::borrow_raw(raw_fd) }
            .try_clone_to_owned()
            .ok()?;
        builder.add_plane(owned, plane as u32, off, stride(plane));
    }
    builder.build()
}

pub const CAPS_FEATURE_MEMORY_CUDA_MEMORY: &str = "memory:CUDAMemory"; // TODO: get it from FFI from gstcudamemory.h (https://github.com/GStreamer/gstreamer/blob/9d6abcc18cc9a60a212966a2daaf4a1af243f5da/subprojects/gst-plugins-bad/gst-libs/gst/cuda/gstcudamemory.h#L113-L121)

pub fn init_cuda() -> Result<(), String> {
    static mut INITIALIZED: bool = false;
    if !unsafe { INITIALIZED } {
        unsafe {
            if ffi::gst_cuda_load_library() == glib_ffi::GFALSE {
                return Err("Failed to load CUDA library".into());
            }
            ffi::gst_cuda_memory_init_once();
            ffi::init_cuda_egl()?;

            INITIALIZED = true;
            Ok(())
        }
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub struct CUDAContext {
    ptr: *mut GstCudaContext,
    stream: Option<StreamHandle>,
}

impl Drop for CUDAContext {
    fn drop(&mut self) {
        // Release the CUDA stream first: destroying a stream pushes its parent
        // context, which must still be a valid GstCudaContext at that point.
        // Dropping the context before the stream triggers GST_IS_CUDA_CONTEXT
        // failures (and a use-after-free) during teardown.
        self.stream = None;
        unsafe {
            gst::ffi::gst_object_unref(self.ptr as *mut gst::ffi::GstObject);
        }
    }
}

#[derive(Debug)]
pub struct StreamHandle {
    stream: ffi::GstCudaStreamHandle,
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        unsafe {
            ffi::gst_cuda_stream_unref(self.stream);
        }
    }
}

#[derive(Debug)]
pub struct CUDABufferPool {
    pool: ffi::GstBufferPool,
}

impl CUDABufferPool {
    pub fn new(cudacontext: &CUDAContext) -> Result<Self, String> {
        let pool = unsafe { ffi::gst_cuda_buffer_pool_new(cudacontext.ptr) };
        if pool.is_null() {
            Err("Failed to create CUDA buffer pool".into())
        } else {
            Ok(CUDABufferPool { pool })
        }
    }

    pub fn from(pool: *mut gst::ffi::GstBufferPool) -> Result<Self, String> {
        if ffi::gst_is_cuda_buffer_pool(pool) {
            unsafe { gst::ffi::gst_object_ref(pool as *mut gst::ffi::GstObject) };
            Ok(CUDABufferPool {
                pool: pool as ffi::GstBufferPool,
            })
        } else {
            Err("Input buffer pool isn't a CUDABufferPool".into())
        }
    }

    pub fn configure(
        &self,
        caps: &gst::Caps,
        stream_handle: &StreamHandle,
        size: u32,
        min_buffers: u32,
        max_buffers: u32,
    ) -> Result<(), String> {
        let config = unsafe {
            gst::ffi::gst_buffer_pool_get_config(self.pool as *mut gst::ffi::GstBufferPool)
        };
        if config.is_null() {
            return Err("Failed to get buffer pool config".into());
        }

        // TODO: support getting the stream handler too here
        //       https://github.com/GStreamer/gstreamer/blob/c5a470e5164ce7fa8fd5fa80650d9ee35ce214d8/subprojects/gst-plugins-bad/sys/nvcodec/gstcudaconvertscale.c#L1293-L1300
        //       ultimately it doesn't matter for us because we are creating a new pool every time

        // Configure the pool
        unsafe {
            // Set the CUDA stream in the config
            ffi::gst_buffer_pool_config_set_cuda_stream(config, stream_handle.stream);

            gst::ffi::gst_buffer_pool_config_add_option(
                config,
                ffi::GST_BUFFER_POOL_OPTION_VIDEO_META.as_ptr() as *const c_char,
            );
            gst::ffi::gst_buffer_pool_config_set_params(
                config,
                caps.to_glib_none().0,
                size,
                min_buffers,
                max_buffers,
            );
        }

        // Set the configuration
        let result = unsafe {
            gst::ffi::gst_buffer_pool_set_config(self.pool as *mut gst::ffi::GstBufferPool, config)
        };
        if result == glib_ffi::GFALSE {
            Err("Failed to set buffer pool config".into())
        } else {
            Ok(())
        }
    }

    pub fn get_updated_size(&self) -> Result<u32, String> {
        let config = unsafe {
            gst::ffi::gst_buffer_pool_get_config(self.pool as *mut gst::ffi::GstBufferPool)
        };
        if config.is_null() {
            return Err("Failed to get buffer pool config".into());
        }

        let mut size = 0;
        unsafe {
            gst::ffi::gst_buffer_pool_config_get_params(
                config,
                ptr::null_mut(),
                &mut size,
                ptr::null_mut(),
                ptr::null_mut(),
            );
        }
        Ok(size)
    }

    pub fn activate(&self) -> Result<(), String> {
        let result = unsafe {
            gst::ffi::gst_buffer_pool_set_active(
                self.pool as *mut gst::ffi::GstBufferPool,
                glib_ffi::GTRUE,
            )
        };
        if result == glib_ffi::GFALSE {
            Err("Failed to activate buffer pool".into())
        } else {
            Ok(())
        }
    }

    pub fn set_nth_allocation_pool(
        &self,
        query: &mut Allocation,
        idx: u32,
        size: u32,
        min_buffers: u32,
        max_buffers: u32,
    ) {
        unsafe {
            gst::ffi::gst_query_set_nth_allocation_pool(
                query.as_mut_ptr(),
                idx,
                self.pool as *mut gst::ffi::GstBufferPool,
                size,
                min_buffers,
                max_buffers,
            );
        }
    }

    pub fn add_allocation_pool(
        &self,
        query: &mut Allocation,
        size: u32,
        min_buffers: u32,
        max_buffers: u32,
    ) {
        unsafe {
            gst::ffi::gst_query_add_allocation_pool(
                query.as_mut_ptr(),
                self.pool as *mut gst::ffi::GstBufferPool,
                size,
                min_buffers,
                max_buffers,
            )
        }
    }
}

impl Drop for CUDABufferPool {
    fn drop(&mut self) {
        unsafe {
            gst::glib::gobject_ffi::g_object_unref(
                self.pool as *mut gst::glib::gobject_ffi::GObject,
            );
        }
    }
}

unsafe impl Send for CUDAContext {}
unsafe impl Send for CUDABufferPool {}

impl CUDAContext {
    pub fn new(device_id: c_int) -> Result<Self, String> {
        let ptr = unsafe { ffi::gst_cuda_context_new(device_id) };
        if ptr.is_null() {
            return Err("Failed to create CUDA context".into());
        }

        // Create a CUDA stream
        let stream = unsafe { ffi::gst_cuda_stream_new(ptr) };

        Ok(CUDAContext {
            ptr,
            stream: if stream.is_null() {
                None
            } else {
                Some(StreamHandle { stream })
            },
        })
    }

    pub fn new_from_gstreamer(
        element: &Element,
        default_device_id: c_int,
        cuda_raw_ptr: *mut *mut GstCudaContext,
    ) -> Result<Self, String> {
        let result = unsafe {
            ffi::gst_cuda_ensure_element_context(
                element.to_glib_none().0,
                default_device_id,
                cuda_raw_ptr,
            )
        };

        if result == glib_ffi::GFALSE {
            Err("Failed to create CUDA context".into())
        } else {
            let stream = unsafe { ffi::gst_cuda_stream_new(*cuda_raw_ptr) };
            Ok(CUDAContext {
                ptr: unsafe { *cuda_raw_ptr },
                stream: if stream.is_null() {
                    None
                } else {
                    Some(StreamHandle { stream })
                },
            })
        }
    }

    pub fn new_from_set_context(
        element: &Element,
        context: &Context,
        default_device_id: c_int,
        cuda_raw_ptr: *mut *mut GstCudaContext,
    ) -> Result<Self, String> {
        let result = unsafe {
            ffi::gst_cuda_handle_set_context(
                element.to_glib_none().0,
                context.to_glib_none().0,
                default_device_id,
                cuda_raw_ptr,
            )
        };

        if result == glib_ffi::GFALSE {
            Err("Failed to create CUDA context".into())
        } else {
            let stream = unsafe { ffi::gst_cuda_stream_new(*cuda_raw_ptr) };
            Ok(CUDAContext {
                ptr: unsafe { *cuda_raw_ptr },
                stream: if stream.is_null() {
                    None
                } else {
                    Some(StreamHandle { stream })
                },
            })
        }
    }

    pub fn as_ptr(&self) -> *mut GstCudaContext {
        self.ptr
    }

    pub fn stream(&self) -> Option<&StreamHandle> {
        self.stream.as_ref()
    }
}

#[derive(Debug)]
pub struct CUDAImage {
    #[allow(dead_code)]
    egl_image: EGLImage,
    cuda_graphic_resource: CUgraphicsResource,
}

impl CUDAImage {
    pub fn from(egl_image: EGLImage, cuda_context: &CUDAContext) -> Result<Self, String> {
        let _cuda_context_guard = ffi::CudaContextGuard::new(cuda_context)?;
        let cuda_egl_fn = ffi::get_cuda_egl_functions()?;
        // Let's import the EGLImage into CUDA
        let mut cuda_resource: CUgraphicsResource = ptr::null_mut();
        unsafe {
            cuda_egl_fn.register_egl_image(
                &mut cuda_resource,
                egl_image.image,
                0, // flags (0 = read/write)
            )?;
        }
        Ok(CUDAImage {
            egl_image,
            cuda_graphic_resource: cuda_resource,
        })
    }

    pub fn to_gst_buffer(
        &self,
        dma_video_info: VideoInfoDmaDrm,
        cuda_context: &CUDAContext,
        buffer_pool: Option<&CUDABufferPool>,
    ) -> Result<GstBuffer, Box<dyn std::error::Error>> {
        let _cuda_context_guard = ffi::CudaContextGuard::new(cuda_context)?;
        let cuda_egl_fn = ffi::get_cuda_egl_functions()?;

        let egl_frame =
            unsafe { cuda_egl_fn.get_mapped_egl_frame(self.cuda_graphic_resource, 0, 0)? };

        // Acquire buffer from pool or allocate directly
        let mut buffer = ffi::acquire_or_alloc_buffer(
            buffer_pool.as_ref().map(|p| p.pool),
            cuda_context,
            &dma_video_info,
        )?;

        // Copy data to the buffer
        let video_info = ffi::copy_to_gst_buffer(egl_frame, &mut buffer, cuda_context)?;

        let buffer_ref = buffer.get_mut().unwrap();
        VideoMeta::add_full(
            buffer_ref,
            gst_video::VideoFrameFlags::empty(),
            video_info.format(),
            video_info.width(),
            video_info.height(),
            video_info.offset(),
            video_info.stride(),
        )?;

        Ok(buffer)
    }
}

impl Drop for CUDAImage {
    fn drop(&mut self) {
        let cuda_egl_fn = ffi::get_cuda_egl_functions().expect("Failed to get CUDA EGL functions");
        unsafe {
            cuda_egl_fn
                .unregister_resource(self.cuda_graphic_resource)
                .expect("Failed to unregister resource");
        }
    }
}

pub fn gst_cuda_handle_context_query_wrapped(
    element: &Element,
    query: &mut QueryRef,
    cuda_context: &CUDAContext,
) -> bool {
    let result = unsafe {
        ffi::gst_cuda_handle_context_query(
            element.to_glib_none().0,
            query.as_mut_ptr(),
            cuda_context.ptr,
        )
    };
    result == glib_ffi::GTRUE
}
