use gst::Buffer;
use gst_video::VideoInfo;
use gstreamer_allocators::DmaBufAllocator;
use smithay::backend::allocator::dmabuf::{Dmabuf, DmabufAllocator};
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Allocator, Fourcc};
use smithay::backend::drm::DrmNode;
use smithay::backend::renderer::gles::{GlesRenderbuffer, GlesRenderer, GlesTexture};
use smithay::backend::renderer::{Bind, ExportMem, Frame, ImportDma, Offscreen, Renderer};
use smithay::reexports::drm::buffer::DrmFourcc;
use smithay::reexports::gbm::Modifier;
use smithay::utils::{DeviceFd, Rectangle, Transform};
use std::fs::File;
use std::os::fd::{AsRawFd, OwnedFd};

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
    format: DrmFourcc,
    video_info: VideoInfo,
    gles_texture: Option<GlesTexture>,
}

impl GsDmaBuf {
    pub fn new(render_node: DrmNode, video_info: VideoInfo) -> Option<Self> {
        let format = Fourcc::Abgr8888; // TODO: format from drm-format

        let file = File::options()
            .read(true)
            .write(true)
            .open(render_node.dev_path().unwrap().as_path())
            .expect("Failed to open device node");
        let fd = DeviceFd::from(Into::<OwnedFd>::into(file));
        let gbm = GbmDevice::new(fd).expect("Failed to create GBM device");
        let allocator = GbmAllocator::new(gbm, GbmBufferFlags::RENDERING);
        let mut dma_allocator = DmabufAllocator(allocator);

        let modifiers = [Modifier::Linear]; // TODO: Support modifiers from video_info
        let result = dma_allocator.create_buffer(
            video_info.width(),
            video_info.height(),
            format,
            &modifiers,
        );
        match result {
            Ok(buffer) => Some(GsDmaBuf {
                buffer,
                format,
                video_info,
                gles_texture: None,
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
            GsBufferType::RAW(buffer) => renderer.bind(buffer.buffer.clone()),
            GsBufferType::DMA(buffer) => {
                let texture = renderer.import_dmabuf(&buffer.buffer, None)?;
                buffer.gles_texture = Some(texture);
                let size = (
                    buffer.video_info.width() as i32,
                    buffer.video_info.height() as i32,
                );
                let offscreen = Offscreen::<GlesRenderbuffer>::create_buffer(
                    renderer,
                    buffer.format,
                    size.into(),
                )?;
                renderer.bind(offscreen)
            }
        }
    }

    fn to_gs_buffer(&self, renderer: &mut GlesRenderer) -> Buffer {
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
                // Adapted from: https://github.com/games-on-whales/smithay/blob/ef9782b8548c6e876bc61052e4e09351e4071a35/examples/buffer_test.rs#L326-L351
                let size = (
                    buffer.video_info.width() as i32,
                    buffer.video_info.height() as i32,
                );
                let mut frame = renderer
                    .render(size.into(), Transform::Normal)
                    .expect("Failed to create frame");
                frame
                    .render_texture_at(
                        buffer.gles_texture.as_ref().unwrap(),
                        (0, 0).into(),
                        1,
                        1.,
                        Transform::Normal,
                        &[Rectangle::from_loc_and_size((0, 0), size)],
                        &[],
                        1.0,
                    )
                    .expect("Failed to render texture");
                frame
                    .finish()
                    .expect("Failed to finish frame")
                    .wait()
                    .expect("Synchronization error");

                let mut gst_buffer = gst::Buffer::new();
                {
                    // TODO: is this right?
                    let gst_buffer = gst_buffer.get_mut().unwrap();

                    let allocator = DmaBufAllocator::new();
                    buffer.buffer.handles().for_each(|handle| {
                        let fd = handle.as_raw_fd();
                        // TODO: should we leak the handle here somehow?
                        let memory = unsafe {
                            allocator
                                .alloc(fd, (size.0 * size.1) as usize)
                                .expect("Failed to allocate memory")
                        };
                        gst_buffer.append_memory(memory);
                    });
                }
                gst_buffer
            }
        }
    }
}
