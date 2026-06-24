#[cfg(feature = "cuda")]
pub mod cuda;

use crate::DrmModifier;
#[cfg(feature = "cuda")]
use crate::utils::allocator::cuda::{CUDABufferPool, CUDAContext, CUDAImage, EGLImage};
use crate::utils::device::PCIVendor;
use crate::utils::device::gpu::GPUDevice;
use crate::utils::vulkan_nv12::VulkanNv12;
use gst::Buffer as GstBuffer;
use gst_video::{VideoFormat, VideoInfo, VideoInfoDmaDrm, VideoMeta};
use gstreamer_allocators::{DmaBufAllocator, DmaBufAllocatorExtManual, FdMemoryFlags};
use smithay::backend::allocator::dmabuf::{Dmabuf, DmabufAllocator};
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Allocator, Buffer, Fourcc};
use smithay::backend::drm::DrmNode;
#[cfg(feature = "cuda")]
use smithay::backend::egl::ffi::egl::types::EGLDisplay;
use smithay::backend::renderer::gles::{GlesError, GlesRenderbuffer, GlesRenderer, GlesTarget};
use smithay::backend::renderer::{Bind, ExportMem, Offscreen, Renderer};
use smithay::reexports::drm::buffer::DrmFourcc;
use smithay::reexports::gbm::Modifier;
use smithay::reexports::rustix::fs::{SeekFrom, seek};
use smithay::utils::{DeviceFd, Rectangle};
use std::fs::File;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct GsGlesbuffer {
    buffer: GlesRenderbuffer,
    format: DrmFourcc,
    video_info: VideoInfo,
}

impl GsGlesbuffer {
    pub fn new(renderer: &mut GlesRenderer, video_info: VideoInfo) -> Option<Self> {
        let format = Fourcc::try_from(video_info.format().to_fourcc()).unwrap_or(Fourcc::Abgr8888);

        let result = renderer.create_buffer(
            format,
            (video_info.width() as i32, video_info.height() as i32).into(),
        );
        match result {
            Ok(buffer) => Some(GsGlesbuffer {
                buffer,
                format,
                video_info,
            }),
            Err(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GsDmaBuf {
    buffer: Dmabuf,
    video_info: VideoInfoDmaDrm,
    gst_allocator: DmaBufAllocator,
}

pub fn new_gbm_device(render_node: DrmNode) -> Option<GbmDevice<DeviceFd>> {
    let file = File::options()
        .read(true)
        .write(true)
        .open(render_node.dev_path()?.as_path())
        .ok()?;
    let fd = DeviceFd::from(Into::<OwnedFd>::into(file));
    GbmDevice::new(fd).ok()
}

impl GsDmaBuf {
    pub fn new(render_node: DrmNode, video_info: VideoInfoDmaDrm) -> Option<Self> {
        tracing::debug!("Creating DMA buffer from {:?}", &video_info);
        let drm_fourcc = gst_video_format_to_drm_fourcc(&video_info)?;
        let mut drm_modifier = gst_video_format_to_drm_modifier(&video_info)?;
        tracing::info!(
            "Creating DMA buffer - DrmFourcc: {:?}, Modifier: {:?}",
            drm_fourcc,
            drm_modifier
        );

        // NOTE: This is a workaround for the i915 4-tiled modifiers
        //       not being advertised by gstreamer elements.
        // - In this part we check for y-tiled modifiers and
        //   change them back to 4-tiled modifiers to make them actually work.
        //   (These modifiers overlap well enough to work interchangeably)
        // Earlier part in gst-plugin-wayland-display waylandsrc/imp.rs.
        let mut workaround_modifier = None;
        if drm_modifier == DrmModifier::I915_y_tiled {
            workaround_modifier = Some(DrmModifier::Unrecognized(0x0100000000000009));
        }

        let gbm = new_gbm_device(render_node)?;
        let allocator = GbmAllocator::new(gbm, GbmBufferFlags::RENDERING);
        let mut dma_allocator = DmabufAllocator(allocator);

        let modifiers = [drm_modifier];
        let mut result = dma_allocator.create_buffer(
            video_info.width(),
            video_info.height(),
            drm_fourcc,
            &modifiers,
        );
        if result.is_err() && workaround_modifier.is_some() {
            tracing::warn!(
                "Failed to create buffer with modifier {:?}, trying workaround modifier",
                drm_modifier
            );
            // Try the workaround modifier
            drm_modifier = workaround_modifier.unwrap();
            result = dma_allocator.create_buffer(
                video_info.width(),
                video_info.height(),
                drm_fourcc,
                &[drm_modifier],
            );
        }

        match result {
            Ok(buffer) => Some(GsDmaBuf {
                buffer,
                video_info,
                gst_allocator: DmaBufAllocator::new(),
            }),
            Err(_) => {
                tracing::warn!("Failed to create DMA buffer: {}", result.unwrap_err());
                None
            }
        }
    }
}

/// NV12 output via the Vulkan converter: the compositor renders the scene into
/// `rgba` (a GLES-renderable RGBA dmabuf), then [`VulkanNv12`] imports it, runs the
/// RGBA->NV12 compute shader, and exports an NV12 dmabuf (the negotiated modifier --
/// DCC on AMD, LINEAR elsewhere) for the encoders.
#[derive(Debug, Clone)]
pub struct GsNv12Buf {
    /// GLES render target (RGBA); also the Vulkan converter's input dmabuf.
    pub rgba: Dmabuf,
    vulkan: Arc<Mutex<VulkanNv12>>,
    /// The negotiated NV12 video info.
    video_info: VideoInfoDmaDrm,
}

impl GsNv12Buf {
    pub fn new(
        renderer: &mut GlesRenderer,
        render_node: DrmNode,
        video_info: VideoInfoDmaDrm,
    ) -> Option<Self> {
        let (w, h) = (video_info.width(), video_info.height());
        // RGBA render-target modifier candidates the GLES renderer supports (INVALID last).
        let formats =
            <GlesRenderer as Bind<Dmabuf>>::supported_formats(renderer).unwrap_or_default();
        let mut mods: Vec<Modifier> = formats
            .iter()
            .filter(|f| f.code == DrmFourcc::Abgr8888)
            .map(|f| f.modifier)
            .collect();
        mods.sort_by_key(|m| *m == Modifier::Invalid);
        let gbm = new_gbm_device(render_node)?;
        let mut dma = DmabufAllocator(GbmAllocator::new(gbm, GbmBufferFlags::RENDERING));

        // Pick the RGBA modifier. VulkanNv12 imports this buffer on the GPU that produced it.
        //  - Nvidia: keep the GPU's preferred modifier -- its Vulkan imports its own
        //    block-linear RGBA, and forcing LINEAR makes the import fail (no frames).
        //  - Everyone else: prefer LINEAR. On AMD the preferred Abgr8888 modifier is
        //    DCC-compressed, and a DCC render target is mis-sampled when VulkanNv12 imports
        //    it cross-API on radv -- the cursor overlay (drawn last) is silently dropped from
        //    the converted NV12. This buffer is a transient render-once/import-once
        //    intermediate, so DCC buys nothing; LINEAR avoids it and imports cleanly (and is
        //    no slower in practice). Fall back to the other modifiers if LINEAR won't allocate.
        let is_nvidia = matches!(
            GPUDevice::try_from(render_node).map(|d| *d.pci_vendor() == PCIVendor::NVIDIA),
            Ok(true)
        );
        let order: Vec<Modifier> = if is_nvidia {
            mods.iter()
                .copied()
                .chain(std::iter::once(Modifier::Linear))
                .collect()
        } else {
            std::iter::once(Modifier::Linear)
                .chain(mods.iter().copied())
                .collect()
        };
        let rgba = order
            .iter()
            .find_map(|m| dma.create_buffer(w, h, DrmFourcc::Abgr8888, &[*m]).ok())?;
        tracing::debug!(
            "GsNv12Buf: nvidia={is_nvidia} RGBA render target modifier = {:?}",
            rgba.format().modifier
        );
        let vulkan = VulkanNv12::new(render_node, video_info.clone())?;
        Some(GsNv12Buf {
            rgba,
            vulkan: Arc::new(Mutex::new(vulkan)),
            video_info,
        })
    }
}

/// NV12 output as `memory:VulkanImage` on the downstream encoder's shared `GstVulkanDevice`
/// (the Vulkan-encode/interpipe path). Renders the scene into an RGBA dmabuf like
/// [`GsNv12Buf`], but the Vulkan converter writes into the encoder's own encode-src image
/// pool, so `vulkanh264enc` consumes the result zero-copy.
#[derive(Debug, Clone)]
pub struct GsVulkanBuf {
    pub rgba: Dmabuf,
    vulkan: Arc<Mutex<VulkanNv12>>,
    video_info: VideoInfo,
}

impl GsVulkanBuf {
    /// `profile` is the negotiated H.264 profile (for the encode-src image's video profile).
    /// Returns `None` if no shared `GstVulkanDevice` has been received yet (caller then
    /// falls back to the dmabuf path).
    pub fn new(
        renderer: &mut GlesRenderer,
        render_node: DrmNode,
        video_info: VideoInfo,
        profile: String,
    ) -> Option<Self> {
        let (w, h) = (video_info.width(), video_info.height());

        // RGBA render-target modifier (same policy as GsNv12Buf: LINEAR except on Nvidia).
        let formats =
            <GlesRenderer as Bind<Dmabuf>>::supported_formats(renderer).unwrap_or_default();
        let mut mods: Vec<Modifier> = formats
            .iter()
            .filter(|f| f.code == DrmFourcc::Abgr8888)
            .map(|f| f.modifier)
            .collect();
        mods.sort_by_key(|m| *m == Modifier::Invalid);
        let gbm = new_gbm_device(render_node)?;
        let mut dma = DmabufAllocator(GbmAllocator::new(gbm, GbmBufferFlags::RENDERING));
        let is_nvidia = matches!(
            GPUDevice::try_from(render_node).map(|d| *d.pci_vendor() == PCIVendor::NVIDIA),
            Ok(true)
        );
        let order: Vec<Modifier> = if is_nvidia {
            mods.iter()
                .copied()
                .chain(std::iter::once(Modifier::Linear))
                .collect()
        } else {
            std::iter::once(Modifier::Linear)
                .chain(mods.iter().copied())
                .collect()
        };
        let rgba = order
            .iter()
            .find_map(|m| dma.create_buffer(w, h, DrmFourcc::Abgr8888, &[*m]).ok())?;

        // The shared device must already have been absorbed from a GstContext.
        let dev = crate::utils::vulkan_share::shared_device()?;
        let raw = crate::utils::vulkan_share::raw_handles(&dev)?;
        let nv12_caps = gst::Caps::builder("video/x-raw")
            .features(["memory:VulkanImage"])
            .field("format", "NV12")
            .field("width", w as i32)
            .field("height", h as i32)
            .field("framerate", video_info.fps())
            .build();
        let vulkan = VulkanNv12::new_on_shared(dev, raw, &nv12_caps, &profile, w, h)?;
        Some(GsVulkanBuf {
            rgba,
            vulkan: Arc::new(Mutex::new(vulkan)),
            video_info,
        })
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct GsCUDABuf {
    buffer: Dmabuf,
    video_info: VideoInfoDmaDrm,
    // Set in compositor by UpdateCUDABufferPool
    pub(crate) buffer_pool: Arc<Mutex<Option<CUDABufferPool>>>,
    // Cached for CUDA needs
    cuda_image: Arc<Mutex<CUDAImage>>,
    // Order here matters, we want to keep the CUDAContext alive when dropping the CUDAImage
    cuda_context: Arc<Mutex<CUDAContext>>,
}

#[cfg(feature = "cuda")]
impl GsCUDABuf {
    pub fn new(
        render_node: DrmNode,
        cuda_context: Arc<Mutex<CUDAContext>>,
        video_info: VideoInfoDmaDrm,
        buffer_pool: Arc<Mutex<Option<CUDABufferPool>>>,
        egl_display: &EGLDisplay,
    ) -> Option<Self> {
        tracing::debug!("Creating CUDA buffer from {:?}", &video_info);
        let drm_fourcc = gst_video_format_to_drm_fourcc(&video_info)?;
        let drm_modifier = gst_video_format_to_drm_modifier(&video_info)?;
        tracing::info!(
            "Creating CUDA buffer - DrmFourcc: {:?}, Modifier: {:?}",
            drm_fourcc,
            drm_modifier
        );
        let gbm = new_gbm_device(render_node)?;
        let allocator = GbmAllocator::new(gbm, GbmBufferFlags::RENDERING);
        let mut dma_allocator = DmabufAllocator(allocator);

        let modifiers = [drm_modifier];
        let result = dma_allocator.create_buffer(
            video_info.width(),
            video_info.height(),
            drm_fourcc,
            &modifiers,
        );

        match result {
            Ok(buffer) => {
                // Create EGLImage once during initialization
                let egl_image = EGLImage::from(&buffer, egl_display)
                    .expect("Failed to create EGLImage from DMA-BUF");

                // Create CUDAImage once during initialization
                let cuda_image = {
                    let ctx = cuda_context.lock().unwrap();
                    CUDAImage::from(egl_image, &ctx)
                        .expect("Failed to create CUDA image from EGLImage")
                };

                Some(GsCUDABuf {
                    buffer,
                    video_info,
                    buffer_pool,
                    cuda_image: Arc::new(Mutex::new(cuda_image)),
                    cuda_context,
                })
            }
            Err(_) => {
                tracing::warn!("Failed to create DMA buffer: {}", result.unwrap_err());
                None
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum GsBufferType {
    RAW(GsGlesbuffer),
    DMA(GsDmaBuf),
    NV12(GsNv12Buf),
    VULKAN(GsVulkanBuf),
    #[cfg(feature = "cuda")]
    CUDA(GsCUDABuf),
}

pub enum VideoInfoTypes {
    VideoInfo(VideoInfo),
    VideoInfoDmaDrm(VideoInfoDmaDrm),
}

pub trait GsBuffer<R: Renderer> {
    fn bind(&mut self, renderer: &mut R) -> Result<GlesTarget, R::Error>;

    fn to_gs_buffer(
        &self,
        target: &mut GlesTarget,
        renderer: &mut R,
    ) -> Result<GstBuffer, Box<dyn std::error::Error>>;

    // Returns the underlying VideoInfo or VideoInfoDmaDrm
    fn get_video_info(&self) -> VideoInfoTypes;
}

impl GsBuffer<GlesRenderer> for GsBufferType {
    fn bind(&mut self, renderer: &mut GlesRenderer) -> Result<GlesTarget, GlesError> {
        match self {
            GsBufferType::RAW(buffer) => renderer.bind(&mut buffer.buffer),
            GsBufferType::DMA(buffer) => renderer.bind(&mut buffer.buffer),
            // NV12 mode renders the scene into the RGBA dmabuf; Vulkan converts it
            // to NV12 in to_gs_buffer().
            GsBufferType::NV12(buffer) => renderer.bind(&mut buffer.rgba),
            // Vulkan-encode path: render into the RGBA dmabuf; convert in to_gs_buffer().
            GsBufferType::VULKAN(buffer) => renderer.bind(&mut buffer.rgba),
            #[cfg(feature = "cuda")]
            GsBufferType::CUDA(buffer) => renderer.bind(&mut buffer.buffer),
        }
    }

    #[cfg(feature = "cuda")]
    fn to_gs_buffer(
        &self,
        target: &mut GlesTarget,
        renderer: &mut GlesRenderer,
    ) -> Result<GstBuffer, Box<dyn std::error::Error>> {
        match self {
            GsBufferType::RAW(buffer) => {
                let mapping = renderer.copy_framebuffer(
                    target,
                    Rectangle::from_size(
                        (
                            buffer.video_info.width() as i32,
                            buffer.video_info.height() as i32,
                        )
                            .into(),
                    ),
                    buffer.format,
                )?;
                let map = renderer.map_texture(&mapping)?;

                let mut gst_buffer =
                    gst::Buffer::with_size(map.len()).expect("failed to create buffer");
                {
                    let gst_buffer = gst_buffer.get_mut().unwrap();

                    let mut vframe = gst_video::VideoFrameRef::from_buffer_ref_writable(
                        gst_buffer,
                        &buffer.video_info,
                    )
                    .unwrap();
                    let plane_data = vframe.plane_data_mut(0).unwrap();
                    plane_data.clone_from_slice(map);
                }

                Ok(gst_buffer)
            }
            GsBufferType::DMA(buffer) => {
                let mut gst_buffer = GstBuffer::new();
                {
                    let video_format =
                        match VideoFormat::from_fourcc(buffer.buffer.format().code as u32) {
                            // TODO: this seems to always fail
                            VideoFormat::Unknown => {
                                tracing::debug!(
                                    "Failed to convert fourcc to video format: {:?}",
                                    buffer.buffer.format().code
                                );
                                VideoFormat::Bgrx // TODO: Use a more appropriate fallback, can't pass DmaDRM format
                            }
                            format => format,
                        };

                    // Calculate the required size based on GStreamer's expectations
                    let required_size = gst_video::VideoInfo::builder(
                        video_format,
                        buffer.video_info.width(),
                        buffer.video_info.height(),
                    )
                    .build()?
                    .size();

                    let gst_buffer = gst_buffer.get_mut().unwrap();
                    buffer.buffer.handles().for_each(|handle| {
                        let fd = handle.as_raw_fd();
                        let actual_size = seek(&handle.as_fd(), SeekFrom::End(0)).unwrap() as usize;
                        let _ = seek(&handle.as_fd(), SeekFrom::Start(0)); // Reset seek point

                        // Use the larger of the two sizes to ensure we have enough space
                        let allocation_size = required_size.max(actual_size);

                        let memory = unsafe {
                            buffer
                                .gst_allocator
                                .alloc_dmabuf_with_flags(
                                    fd,
                                    allocation_size,
                                    FdMemoryFlags::DONT_CLOSE,
                                )
                                .expect("Failed to allocate memory")
                        };
                        gst_buffer.append_memory(memory);
                    });

                    let offsets = buffer
                        .buffer
                        .offsets()
                        .map(|o| o as usize)
                        .collect::<Vec<_>>();

                    let strides = buffer
                        .buffer
                        .strides()
                        .map(|s| s as i32)
                        .collect::<Vec<_>>();

                    let meta_result = VideoMeta::add_full(
                        gst_buffer,
                        gst_video::VideoFrameFlags::empty(),
                        video_format,
                        buffer.video_info.width(),
                        buffer.video_info.height(),
                        &offsets,
                        &strides,
                    );
                    if let Err(error) = meta_result {
                        tracing::warn!("Failed to add video meta: {:?}", error);
                    }
                }
                Ok(gst_buffer)
            }
            GsBufferType::NV12(buffer) => {
                let mut v = buffer.vulkan.lock().unwrap();
                v.convert(&buffer.rgba)?;
                v.to_gst_buffer()
            }
            GsBufferType::VULKAN(buffer) => {
                let mut v = buffer.vulkan.lock().unwrap();
                v.convert(&buffer.rgba)?;
                v.to_gst_buffer()
            }
            #[cfg(feature = "cuda")]
            GsBufferType::CUDA(buffer) => {
                let cuda_ctx = buffer.cuda_context.lock().unwrap();
                let buffer_pool = buffer.buffer_pool.lock().unwrap();
                let cuda_image = buffer.cuda_image.lock().unwrap();

                Ok(cuda_image.to_gst_buffer(
                    buffer.video_info.clone(),
                    &cuda_ctx,
                    buffer_pool.as_ref(),
                )?)
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn to_gs_buffer(
        &self,
        target: &mut GlesTarget,
        renderer: &mut GlesRenderer,
    ) -> Result<GstBuffer, Box<dyn std::error::Error>> {
        match self {
            GsBufferType::RAW(buffer) => {
                let mapping = renderer.copy_framebuffer(
                    target,
                    Rectangle::from_size(
                        (
                            buffer.video_info.width() as i32,
                            buffer.video_info.height() as i32,
                        )
                            .into(),
                    ),
                    buffer.format,
                )?;
                let map = renderer.map_texture(&mapping)?;

                let mut gst_buffer =
                    gst::Buffer::with_size(map.len()).expect("failed to create buffer");
                {
                    let gst_buffer = gst_buffer.get_mut().unwrap();

                    let mut vframe = gst_video::VideoFrameRef::from_buffer_ref_writable(
                        gst_buffer,
                        &buffer.video_info,
                    )
                    .unwrap();
                    let plane_data = vframe.plane_data_mut(0).unwrap();
                    plane_data.clone_from_slice(map);
                }

                Ok(gst_buffer)
            }
            GsBufferType::DMA(buffer) => {
                let mut gst_buffer = GstBuffer::new();
                {
                    let video_format =
                        gst_video::dma_drm_fourcc_to_format(buffer.buffer.format().code as u32)
                            .unwrap_or_else(|_| {
                                tracing::debug!(
                                    "Failed to convert fourcc to video format: {:?}",
                                    buffer.buffer.format().code
                                );
                                VideoFormat::Bgrx // TODO: Use a more appropriate fallback, can't pass DmaDRM format
                            });

                    // Calculate the required size based on GStreamer's expectations
                    let required_size = gst_video::VideoInfo::builder(
                        video_format,
                        buffer.video_info.width(),
                        buffer.video_info.height(),
                    )
                    .build()?
                    .size();

                    let gst_buffer = gst_buffer.get_mut().unwrap();
                    buffer.buffer.handles().for_each(|handle| {
                        let fd = handle.as_raw_fd();
                        let actual_size = seek(&handle.as_fd(), SeekFrom::End(0)).unwrap() as usize;
                        let _ = seek(&handle.as_fd(), SeekFrom::Start(0)); // Reset seek point

                        // Use the larger of the two sizes to ensure we have enough space
                        let allocation_size = required_size.max(actual_size);

                        let memory = unsafe {
                            buffer
                                .gst_allocator
                                .alloc_dmabuf_with_flags(
                                    fd,
                                    allocation_size,
                                    FdMemoryFlags::DONT_CLOSE,
                                )
                                .expect("Failed to allocate memory")
                        };
                        gst_buffer.append_memory(memory);
                    });

                    let offsets = buffer
                        .buffer
                        .offsets()
                        .map(|o| o as usize)
                        .collect::<Vec<_>>();

                    let strides = buffer
                        .buffer
                        .strides()
                        .map(|s| s as i32)
                        .collect::<Vec<_>>();

                    let meta_result = VideoMeta::add_full(
                        gst_buffer,
                        gst_video::VideoFrameFlags::empty(),
                        video_format,
                        buffer.video_info.width(),
                        buffer.video_info.height(),
                        &offsets,
                        &strides,
                    );
                    if let Err(error) = meta_result {
                        tracing::warn!("Failed to add video meta: {:?}", error);
                    }
                }
                Ok(gst_buffer)
            }
            GsBufferType::NV12(buffer) => {
                let mut v = buffer.vulkan.lock().unwrap();
                v.convert(&buffer.rgba)?;
                v.to_gst_buffer()
            }
            GsBufferType::VULKAN(buffer) => {
                let mut v = buffer.vulkan.lock().unwrap();
                v.convert(&buffer.rgba)?;
                v.to_gst_buffer()
            }
        }
    }

    fn get_video_info(&self) -> VideoInfoTypes {
        match self {
            GsBufferType::RAW(buffer) => VideoInfoTypes::VideoInfo(buffer.video_info.clone()),
            GsBufferType::DMA(buffer) => VideoInfoTypes::VideoInfoDmaDrm(buffer.video_info.clone()),
            GsBufferType::NV12(buffer) => {
                VideoInfoTypes::VideoInfoDmaDrm(buffer.video_info.clone())
            }
            GsBufferType::VULKAN(buffer) => VideoInfoTypes::VideoInfo(buffer.video_info.clone()),
            #[cfg(feature = "cuda")]
            GsBufferType::CUDA(buffer) => {
                VideoInfoTypes::VideoInfoDmaDrm(buffer.video_info.clone())
            }
        }
    }
}

pub fn gst_video_format_name_to_drm_fourcc(gst_format: String) -> Option<DrmFourcc> {
    match gst_format.to_lowercase().as_str() {
        "abgr" => Some(DrmFourcc::Rgba8888),
        "argb" => Some(DrmFourcc::Bgra8888),
        "bgra" => Some(DrmFourcc::Argb8888),
        "bgrx" => Some(DrmFourcc::Xrgb8888),
        "rgba" => Some(DrmFourcc::Abgr8888),
        "rgbx" => Some(DrmFourcc::Xbgr8888),
        "xbgr" => Some(DrmFourcc::Rgbx8888),
        "xrgb" => Some(DrmFourcc::Bgrx8888),
        _ => {
            tracing::warn!("Unsupported video format: {:?}", gst_format);
            None
        }
    }
}

pub fn gst_video_format_to_drm_fourcc(format: &VideoInfoDmaDrm) -> Option<DrmFourcc> {
    // VideoFormat::from_fourcc() returns format unknown for some reason, so we manually parse the caps
    let fourcc = DrmFourcc::try_from(format.fourcc());
    match fourcc {
        Ok(fourcc) => Some(fourcc),
        Err(error) => {
            tracing::warn!(
                "Failed to convert fourcc ({:?}): {:?}",
                format.fourcc(),
                error
            );
            let caps = format.to_caps().unwrap();
            let drm_format_str = caps.structure(0)?.get::<&str>("drm-format");
            if drm_format_str.is_err() {
                tracing::warn!("Failed to get DRM format from caps {:?}", caps);
                return None;
            }
            let gst_format = drm_format_str.unwrap().split(":").next().unwrap();
            gst_video_format_name_to_drm_fourcc(gst_format.into())
        }
    }
}

pub fn gst_video_format_to_drm_modifier(format: &VideoInfoDmaDrm) -> Option<DrmModifier> {
    let full_modifier = format.modifier();
    match Modifier::try_from(full_modifier) {
        Ok(modifier) => Some(modifier),
        Err(error) => {
            tracing::warn!(
                "Failed to convert modifier ({:?}): {:?}",
                full_modifier,
                error
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::renderer::setup_renderer;
    use crate::utils::tests::test_init;
    use smithay::backend::renderer::Frame;
    use smithay::utils::Transform;

    /// Skip the current hardware-gated test (print why, return).
    macro_rules! skip {
        ($($a:tt)*) => {{ eprintln!("skip: {}", format!($($a)*)); return; }};
    }

    /// A render node whose kernel driver is one of `drivers`, for hardware-gated
    /// tests -- so they target the right GPU on a multi-GPU host instead of a
    /// hardcoded `renderD12x` that may be a different vendor.
    fn pick_render_node(drivers: &[&str]) -> Option<DrmNode> {
        for entry in std::fs::read_dir("/dev/dri").ok()?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("renderD") {
                continue;
            }
            let drv = std::fs::read_to_string(format!("/sys/class/drm/{name}/device/uevent"))
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find_map(|l| l.strip_prefix("DRIVER=").map(str::to_owned))
                })
                .unwrap_or_default();
            if drivers.iter().any(|d| *d == drv) {
                if let Ok(node) = DrmNode::from_path(format!("/dev/dri/{name}")) {
                    return Some(node);
                }
            }
        }
        None
    }

    /// Adapted from: https://github.com/games-on-whales/smithay/blob/master/examples/buffer_test.rs#L277
    /// Produces a 2x2 grid of colored rectangles:
    /// ```
    /// ┌─────────┬─────────┐
    /// │   RED   │  GREEN  │
    /// │ (top-   │ (top-   │
    /// │  left)  │  right) │
    /// ├─────────┼─────────┤
    /// │  BLUE   │ YELLOW  │
    /// │ (bottom-│ (bottom-│
    /// │  left)  │  right) │
    /// └─────────┴─────────┘
    /// ```
    fn render_into<R, T>(renderer: &mut R, buffer: &mut T, w: i32, h: i32)
    where
        R: Renderer + Bind<T>,
    {
        let mut framebuffer = renderer.bind(buffer).expect("Failed to bind dmabuf");

        let mut frame = renderer
            .render(&mut framebuffer, (w, h).into(), Transform::Normal)
            .expect("Failed to create render frame");
        frame
            .clear(
                [1.0, 0.0, 0.0, 1.0].into(), // RED
                &[Rectangle::from_size((w / 2, h / 2).into())],
            )
            .expect("Render error");
        frame
            .clear(
                [0.0, 1.0, 0.0, 1.0].into(), // GREEN
                &[Rectangle::new((w / 2, 0).into(), (w / 2, h / 2).into())],
            )
            .expect("Render error");
        frame
            .clear(
                [0.0, 0.0, 1.0, 1.0].into(), // BLUE
                &[Rectangle::new((0, h / 2).into(), (w / 2, h / 2).into())],
            )
            .expect("Render error");
        frame
            .clear(
                [1.0, 1.0, 0.0, 1.0].into(), // YELLOW
                &[Rectangle::new((w / 2, h / 2).into(), (w / 2, h / 2).into())],
            )
            .expect("Render error");
        frame
            .finish()
            .expect("Failed to finish render frame")
            .wait()
            .expect("Synchronization error");
    }

    #[test]
    fn test_gsglesbuffer() {
        test_init();

        let mut renderer = setup_renderer(None);
        let video_info = VideoInfo::builder(gst_video::VideoFormat::Rgba, 10, 10)
            .build()
            .unwrap();

        let raw_buffer = GsGlesbuffer::new(&mut renderer, video_info.clone());
        assert!(raw_buffer.is_some());

        let mut buffer = GsBufferType::RAW(raw_buffer.clone().unwrap());
        let buffer_clone = buffer.clone();

        let bind_result = buffer.bind(&mut renderer);
        assert!(bind_result.is_ok());

        render_into(&mut renderer, &mut raw_buffer.unwrap().buffer, 10, 10);
        let gst_buffer = buffer_clone
            .to_gs_buffer(&mut bind_result.unwrap(), &mut renderer)
            .expect("Failed to convert buffer");
        assert!(gst_buffer.is_writable());
        assert_eq!(gst_buffer.size(), video_info.size());

        let read_buf = gst_buffer
            .into_mapped_buffer_readable()
            .expect("Failed to map buffer");
        let plane_data = read_buf.as_slice();
        assert_eq!(plane_data.len(), 10 * 10 * 4); // 10x10 pixels, 4 bytes per pixel (RGBA)
        assert_eq!(
            plane_data,
            [
                [
                    // R, G, B, A
                    255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
                    0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255
                ]
                .repeat(5),
                [
                    0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
                    255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255,
                    255, 0, 255
                ]
                .repeat(5)
            ]
            .concat()
        )
    }

    #[test]
    #[ignore = "needs a gbm-capable AMD/Intel GPU; run via ci/harness.sh gpu"]
    fn test_dmabuf() {
        test_init();

        let Some(render_node) = pick_render_node(&["amdgpu", "radeon", "i915", "xe"]) else {
            skip!("no gbm-capable (AMD/Intel) render node");
        };
        let mut renderer = setup_renderer(Some(render_node));
        let w = 10;
        let h = 10;
        let caps = gst_video::VideoCapsBuilder::new()
            .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .format(gst_video::VideoFormat::DmaDrm)
            .field("drm-format", "RGBA")
            .height(h)
            .width(w)
            .pixel_aspect_ratio(1.into())
            .framerate(gst::Fraction::new(30, 1))
            .build();
        assert!(caps.is_fixed()); // Required to pass gst_video_is_dma_drm_caps()
        let drm_video_info =
            VideoInfoDmaDrm::from_caps(&caps).expect("Failed to create video info");

        assert_eq!(
            gst_video_format_to_drm_fourcc(&drm_video_info),
            Some(DrmFourcc::Abgr8888)
        );
        assert_eq!(
            gst_video_format_to_drm_modifier(&drm_video_info),
            Some(Modifier::Linear)
        );

        let raw_buffer = GsDmaBuf::new(render_node, drm_video_info);
        if raw_buffer.is_none() {
            skip!("GsDmaBuf RGBA/LINEAR allocation unsupported on this GPU");
        }

        let mut buffer = GsBufferType::DMA(raw_buffer.clone().unwrap());
        let buffer_clone = buffer.clone();

        let bind_result = buffer.bind(&mut renderer);
        assert!(bind_result.is_ok());

        render_into(&mut renderer, &mut raw_buffer.clone().unwrap().buffer, w, h);
        let gst_buffer = buffer_clone
            .to_gs_buffer(&mut bind_result.unwrap(), &mut renderer)
            .expect("Failed to convert buffer");
        let gst_buffer_size = gst_buffer.size();
        assert!(gst_buffer_size >= 4096); // There might be padding but it should at least contain our data

        let read_buf = gst_buffer
            .clone()
            .into_mapped_buffer_readable()
            .expect("Failed to map buffer");
        let plane_data = read_buf.as_slice();

        assert_eq!(plane_data.len(), gst_buffer_size);
        let regions = [
            // Color format here is RGBA
            ((0, 0), [255, 0, 0]),           // Red
            ((w / 2, 0), [0, 255, 0]),       // Green
            ((0, h / 2), [0, 0, 255]),       // Blue
            ((w / 2, h / 2), [255, 255, 0]), // Yellow
        ];

        let stride = raw_buffer
            .unwrap()
            .buffer
            .strides()
            .next()
            .expect("Failed to get stride");
        for ((x_start, y_start), expected_color) in regions {
            let pixel = get_pixel(
                plane_data,
                x_start as usize,
                y_start as usize,
                stride as usize,
            );
            assert_eq!(pixel, expected_color, "Pixel at ({}, {})", x_start, y_start);
        }

        let buf_meta = gst_buffer
            .meta::<VideoMeta>()
            .expect("Failed to get buffer meta");
        assert_eq!(buf_meta.width(), w as u32);
        assert_eq!(buf_meta.height(), h as u32);
        assert_eq!(buf_meta.n_planes(), 1);
    }

    fn get_pixel(buffer: &[u8], x: usize, y: usize, stride: usize) -> [u8; 3] {
        let offset = y * stride + x * 4;
        [buffer[offset], buffer[offset + 1], buffer[offset + 2]]
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs an Nvidia GPU + CUDA; run via ci/harness.sh gpu"]
    fn test_cuda_buffer() {
        test_init();
        if cuda::init_cuda().is_err() {
            skip!("CUDA not available");
        }
        let w = 100;
        let h = 100;

        let Some(render_node) = pick_render_node(&["nvidia"]) else {
            skip!("no Nvidia render node");
        };
        let mut renderer = setup_renderer(Some(render_node));
        let caps = gst_video::VideoCapsBuilder::new()
            .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .format(gst_video::VideoFormat::DmaDrm)
            .field("drm-format", "AR24:0x300000000606010")
            .height(h)
            .width(w)
            .pixel_aspect_ratio(1.into())
            .framerate(gst::Fraction::new(30, 1))
            .build();
        assert!(caps.is_fixed()); // Required to pass gst_video_is_dma_drm_caps()
        let drm_video_info =
            VideoInfoDmaDrm::from_caps(&caps).expect("Failed to create video info");

        assert_eq!(
            gst_video_format_to_drm_fourcc(&drm_video_info),
            Some(DrmFourcc::Argb8888)
        );
        assert_eq!(
            gst_video_format_to_drm_modifier(&drm_video_info),
            Some(Modifier::Unrecognized(0x300000000606010))
        );

        let gst_cuda_ctx = CUDAContext::new(0).expect("Failed to create CUDA context");

        let cuda_caps = gst_video::VideoCapsBuilder::new()
            .features([cuda::CAPS_FEATURE_MEMORY_CUDA_MEMORY])
            .format(VideoFormat::Abgr)
            .height(h)
            .width(w)
            .pixel_aspect_ratio(1.into())
            .framerate(gst::Fraction::new(30, 1))
            .build();
        let buffer_pool = CUDABufferPool::new(&gst_cuda_ctx).expect("Failed to create buffer pool");
        buffer_pool
            .configure(
                &cuda_caps,
                gst_cuda_ctx
                    .stream()
                    .expect("Cuda context without a stream"),
                drm_video_info.size() as u32,
                0,
                0,
            )
            .expect("Failed to configure buffer pool");
        buffer_pool
            .activate()
            .expect("Failed to activate buffer pool");

        let egl_display = renderer.egl_context().display().get_display_handle().handle;
        let raw_buffer = GsCUDABuf::new(
            render_node,
            Arc::new(Mutex::new(gst_cuda_ctx)),
            drm_video_info.clone(),
            Arc::new(Mutex::new(Some(buffer_pool))),
            &egl_display,
        );
        if raw_buffer.is_none() {
            skip!("GsCUDABuf allocation unsupported on this GPU");
        }

        let mut buffer = GsBufferType::CUDA(raw_buffer.clone().unwrap());
        let buffer_clone = buffer.clone();

        let bind_result = buffer.bind(&mut renderer);
        assert!(bind_result.is_ok());

        render_into(&mut renderer, &mut raw_buffer.clone().unwrap().buffer, w, h);
        let gst_buffer = buffer_clone
            .to_gs_buffer(&mut bind_result.unwrap(), &mut renderer)
            .expect("Failed to convert buffer");

        let gst_buffer_size = gst_buffer.size();
        assert!(gst_buffer_size >= 4096); // There might be padding, but it should at least contain our data

        let read_buf = gst_buffer
            .clone()
            .into_mapped_buffer_readable()
            .expect("Failed to map buffer");
        let plane_data = read_buf.as_slice();

        assert_eq!(plane_data.len(), gst_buffer_size);
        let regions = [
            // Color format here is BGRA
            ((0, 0), [0, 0, 255]),           // Red
            ((w / 2, 0), [0, 255, 0]),       // Green
            ((0, h / 2), [255, 0, 0]),       // Blue
            ((w / 2, h / 2), [0, 255, 255]), // Yellow
        ];

        let stride = gst_buffer
            .meta::<VideoMeta>()
            .expect("Failed to get buffer meta")
            .stride()[0];
        for ((x_start, y_start), expected_color) in regions {
            let pixel = get_pixel(
                plane_data,
                x_start as usize,
                y_start as usize,
                stride as usize,
            );
            assert_eq!(pixel, expected_color, "Pixel at ({}, {})", x_start, y_start);
        }

        let buf_meta = gst_buffer
            .meta::<VideoMeta>()
            .expect("Failed to get buffer meta");
        assert_eq!(buf_meta.width(), w as u32);
        assert_eq!(buf_meta.height(), h as u32);
        assert_eq!(buf_meta.n_planes(), 1);
    }

    #[test]
    fn test_gst_video_format_conversions() {
        test_init();

        let caps = gst_video::VideoCapsBuilder::new()
            .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .format(gst_video::VideoFormat::DmaDrm)
            .field("drm-format", "AB24:0x0300000000606010")
            .height(10)
            .width(10)
            .pixel_aspect_ratio(1.into())
            .framerate(gst::Fraction::new(30, 1))
            .build();
        assert!(caps.is_fixed()); // Required to pass gst_video_is_dma_drm_caps()
        let drm_video_info =
            VideoInfoDmaDrm::from_caps(&caps).expect("Failed to create video info");

        assert_eq!(
            gst_video_format_to_drm_fourcc(&drm_video_info).unwrap(),
            DrmFourcc::try_from(875708993).unwrap()
        );

        assert_eq!(
            gst_video_format_to_drm_modifier(&drm_video_info).unwrap(),
            Modifier::Unrecognized(0x0300000000606010)
        )
    }

    #[test]
    fn test_gst_video_from_r8() {
        test_init();

        let caps = gst_video::VideoCapsBuilder::new()
            .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .format(gst_video::VideoFormat::DmaDrm)
            .field("drm-format", "R8  :0x0200000000042305")
            .height(10)
            .width(10)
            .pixel_aspect_ratio(1.into())
            .framerate(gst::Fraction::new(30, 1))
            .build();
        assert!(caps.is_fixed()); // Required to pass gst_video_is_dma_drm_caps()
        let drm_video_info =
            VideoInfoDmaDrm::from_caps(&caps).expect("Failed to create video info");

        assert_eq!(
            gst_video_format_to_drm_fourcc(&drm_video_info).unwrap(),
            DrmFourcc::R8
        );

        assert_eq!(
            gst_video_format_to_drm_modifier(&drm_video_info).unwrap(),
            Modifier::Unrecognized(0x0200000000042305)
        )
    }
}
