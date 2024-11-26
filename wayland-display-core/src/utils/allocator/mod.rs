use gst::Buffer as GstBuffer;
use gst_video::{VideoInfo, VideoInfoDmaDrm};
use gstreamer_allocators::{DmaBufAllocator, FdMemoryFlags};
use smithay::backend::allocator::dmabuf::{Dmabuf, DmabufAllocator};
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Allocator, Fourcc};
use smithay::backend::drm::DrmNode;
use smithay::backend::renderer::gles::{GlesRenderbuffer, GlesRenderer};
use smithay::backend::renderer::{Bind, ExportMem, Offscreen, Renderer};
use smithay::reexports::drm::buffer::DrmFourcc;
use smithay::reexports::gbm::Modifier;
use smithay::reexports::rustix::fs::{seek, SeekFrom};
use smithay::utils::{DeviceFd, Rectangle};
use std::fs::File;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};

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
    gst_allocator: DmaBufAllocator,
}

impl GsDmaBuf {
    pub fn new(render_node: DrmNode, video_info: VideoInfoDmaDrm) -> Option<Self> {
        tracing::debug!("Creating DMA buffer from {:?}", video_info);
        let drm_fourcc = Fourcc::try_from(video_info.fourcc()).ok()?;
        let drm_modifier = Modifier::try_from(video_info.modifier()).unwrap_or(Modifier::Linear);

        let file = File::options()
            .read(true)
            .write(true)
            .open(render_node.dev_path().unwrap().as_path())
            .expect("Failed to open device node");
        let fd = DeviceFd::from(Into::<OwnedFd>::into(file));
        let gbm = GbmDevice::new(fd).expect("Failed to create GBM device");
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
            Ok(buffer) => Some(GsDmaBuf {
                buffer,
                gst_allocator: DmaBufAllocator::new(),
            }),
            Err(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum GsBufferType {
    RAW(GsGlesbuffer),
    DMA(GsDmaBuf),
}

pub trait GsBuffer<R: Renderer> {
    fn bind(&mut self, renderer: &mut R) -> Result<(), R::Error>;

    fn to_gs_buffer(&self, renderer: &mut R) -> gst::Buffer;
}

impl GsBuffer<GlesRenderer> for GsBufferType {
    fn bind(
        &mut self,
        renderer: &mut GlesRenderer,
    ) -> Result<(), <GlesRenderer as Renderer>::Error> {
        match self {
            GsBufferType::RAW(buffer) => renderer.bind(buffer.clone().buffer),
            GsBufferType::DMA(buffer) => renderer.bind(buffer.clone().buffer),
        }
    }

    fn to_gs_buffer(&self, renderer: &mut GlesRenderer) -> GstBuffer {
        match self {
            GsBufferType::RAW(buffer) => {
                let mapping = renderer
                    .copy_framebuffer(
                        Rectangle::from_loc_and_size(
                            (0, 0),
                            (
                                buffer.video_info.width() as i32,
                                buffer.video_info.height() as i32,
                            ),
                        ),
                        buffer.format,
                    )
                    .expect("Failed to export framebuffer");
                let map = renderer
                    .map_texture(&mapping)
                    .expect("Failed to download framebuffer");

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

                gst_buffer
            }
            GsBufferType::DMA(buffer) => {
                let mut gst_buffer = GstBuffer::new();
                {
                    let gst_buffer = gst_buffer.get_mut().unwrap();
                    buffer.buffer.handles().for_each(|handle| {
                        let fd = handle.as_raw_fd();
                        let size = seek(&handle.as_fd(), SeekFrom::End(0)).unwrap();
                        let _ = seek(&handle.as_fd(), SeekFrom::Start(0)); // Reset seek point
                        let memory = unsafe {
                            buffer
                                .gst_allocator
                                .alloc_with_flags(fd, size as usize, FdMemoryFlags::DONT_CLOSE)
                                .expect("Failed to allocate memory")
                        };
                        gst_buffer.append_memory(memory);
                    });
                    // TODO: There may be some extra information about the pitch, stride and plane
                    //       offset when we export the surface, we also need to translate them into
                    //       GstVideoMeta and attached it to the GstBuffer
                }
                gst_buffer
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::renderer::setup_renderer;
    use smithay::backend::renderer::Frame;
    use smithay::utils::Transform;
    use std::sync::Once;
    static INIT: Once = Once::new();
    pub fn setup() -> () {
        INIT.call_once(|| {
            tracing_subscriber::fmt::try_init().ok();
            gst::init().expect("Failed to initialize GStreamer");
        });
    }

    // Adapted from: https://github.com/games-on-whales/smithay/blob/ef9782b8548c6e876bc61052e4e09351e4071a35/examples/buffer_test.rs#L326-L351
    fn render_into<R>(renderer: &mut R, w: i32, h: i32)
    where
        R: Renderer,
    {
        let mut frame = renderer
            .render((w, h).into(), Transform::Normal)
            .expect("Failed to create render frame");
        frame
            .clear(
                [1.0, 0.0, 0.0, 1.0],
                &[Rectangle::from_loc_and_size((0, 0), (w / 2, h / 2))],
            )
            .expect("Render error");
        frame
            .clear(
                [0.0, 1.0, 0.0, 1.0],
                &[Rectangle::from_loc_and_size((w / 2, 0), (w / 2, h / 2))],
            )
            .expect("Render error");
        frame
            .clear(
                [0.0, 0.0, 1.0, 1.0],
                &[Rectangle::from_loc_and_size((0, h / 2), (w / 2, h / 2))],
            )
            .expect("Render error");
        frame
            .clear(
                [1.0, 1.0, 0.0, 1.0],
                &[Rectangle::from_loc_and_size((w / 2, h / 2), (w / 2, h / 2))],
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
        setup();

        let mut renderer = setup_renderer(None);
        let video_info = VideoInfo::builder(gst_video::VideoFormat::Rgba, 10, 10)
            .build()
            .unwrap();

        let raw_buffer = GsGlesbuffer::new(&mut renderer, video_info.clone());
        assert!(raw_buffer.is_some());

        let mut buffer = GsBufferType::RAW(raw_buffer.unwrap());
        let bind_result = buffer.bind(&mut renderer);
        assert!(bind_result.is_ok());

        render_into(&mut renderer, 10, 10);
        let gst_buffer = buffer.to_gs_buffer(&mut renderer);
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
    fn test_dmabuf() {
        setup();

        let render_node =
            DrmNode::from_path("/dev/dri/renderD128").expect("Failed to create render node");
        let mut renderer = setup_renderer(Some(render_node));
        let video_info = VideoInfo::builder(gst_video::VideoFormat::DmaDrm, 10, 10)
            .build()
            .unwrap();
        let drm_video_info = VideoInfoDmaDrm::new(
            video_info.clone(),
            Fourcc::Abgr8888 as u32,
            Modifier::Linear.into(),
        );

        let raw_buffer = GsDmaBuf::new(render_node, drm_video_info.clone());
        assert!(raw_buffer.is_some());

        let mut buffer = GsBufferType::DMA(raw_buffer.clone().unwrap());
        let bind_result = buffer.bind(&mut renderer);
        assert!(bind_result.is_ok());

        render_into(&mut renderer, 10, 10);
        let gst_buffer = buffer.to_gs_buffer(&mut renderer);
        let gst_buffer_size = gst_buffer.size();
        assert_eq!(gst_buffer_size, 65536); // There's going to be a lot of padding

        let read_buf = gst_buffer
            .into_mapped_buffer_readable()
            .expect("Failed to map buffer");
        let plane_data = read_buf.as_slice();

        assert_eq!(plane_data.len(), gst_buffer_size);
        assert_eq!(
            plane_data[0..10 * 10 * 4],
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
}
