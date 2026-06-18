//! GLES RGB->NV12 conversion.
//!
//! GLES can only render RGBA, so to emit NV12 we render the scene to an RGB
//! texture and then run two texture-sampling passes that write the luma (Y) and
//! chroma (UV) planes of an NV12 buffer:
//!   - Y plane: full resolution, single channel (R8), luma per pixel.
//!   - UV plane: half resolution, two channels (GR88), interleaved Cb/Cr; the
//!     half-res render with bilinear sampling does the 4:2:0 chroma subsample.
//!
//! Each plane is a single-plane DMABuf so we can bind it as a normal GLES render
//! target via smithay's `Bind<Dmabuf>` (no raw EGL). They are exported together
//! as a 2-memory NV12 gst buffer.

use gst::Buffer as GstBuffer;
use gst_video::{VideoFormat, VideoInfoDmaDrm, VideoMeta};
use gstreamer_allocators::{DmaBufAllocator, FdMemoryFlags};
use smithay::backend::allocator::dmabuf::{Dmabuf, DmabufAllocator};
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags};
use smithay::backend::allocator::{Allocator, Buffer, Fourcc};
use smithay::backend::drm::DrmNode;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexProgram, GlesTexture};
use smithay::backend::renderer::{Bind, Frame, Renderer};
use smithay::reexports::drm::buffer::DrmFourcc;
use smithay::reexports::gbm::Modifier;
use smithay::utils::{Rectangle, Transform};
use std::os::fd::{AsFd, AsRawFd};

use crate::utils::allocator::new_gbm_device;

// BT.601 limited-range RGB->YCbCr. Mirrors the matrix VA/CUDA converters use.
const Y_SHADER: &str = "//_DEFINES_
precision mediump float;
uniform sampler2D tex;
uniform float alpha;
varying vec2 v_coords;
void main() {
    vec3 c = texture2D(tex, v_coords).rgb;
    float y = 0.257 * c.r + 0.504 * c.g + 0.098 * c.b + 0.0625;
    gl_FragColor = vec4(y, y, y, 1.0);
}";

const UV_SHADER: &str = "//_DEFINES_
precision mediump float;
uniform sampler2D tex;
uniform float alpha;
varying vec2 v_coords;
void main() {
    vec3 c = texture2D(tex, v_coords).rgb;
    float u = -0.148 * c.r - 0.291 * c.g + 0.439 * c.b + 0.5;
    float v =  0.439 * c.r - 0.368 * c.g - 0.071 * c.b + 0.5;
    // GR88: .r and .g map to the two interleaved chroma bytes; channel order
    // is verified by test_nv12 and flipped here if needed.
    gl_FragColor = vec4(u, v, 0.0, 1.0);
}";

#[derive(Debug, Clone)]
pub struct Nv12Shaders {
    pub y: GlesTexProgram,
    pub uv: GlesTexProgram,
}

impl Nv12Shaders {
    pub fn compile(renderer: &mut GlesRenderer) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Nv12Shaders {
            y: renderer.compile_custom_texture_shader(Y_SHADER, &[])?,
            uv: renderer.compile_custom_texture_shader(UV_SHADER, &[])?,
        })
    }
}

fn alloc_plane(
    render_node: DrmNode,
    fourcc: DrmFourcc,
    w: u32,
    h: u32,
) -> Option<Dmabuf> {
    let gbm = new_gbm_device(render_node)?;
    let allocator = GbmAllocator::new(gbm, GbmBufferFlags::RENDERING);
    let mut dma = DmabufAllocator(allocator);
    dma.create_buffer(w, h, fourcc, &[Modifier::Linear]).ok()
}

/// Holds the intermediate RGB target, the two NV12 plane buffers, the compiled
/// conversion shaders, and the negotiated NV12 video info.
#[derive(Debug, Clone)]
pub struct Nv12Target {
    pub rgb: GlesTexture,
    pub y: Dmabuf,
    pub uv: Dmabuf,
    pub width: u32,
    pub height: u32,
    pub video_info: VideoInfoDmaDrm,
    shaders: Nv12Shaders,
    gst_allocator: DmaBufAllocator,
}

impl Nv12Target {
    pub fn new(
        renderer: &mut GlesRenderer,
        render_node: DrmNode,
        video_info: VideoInfoDmaDrm,
    ) -> Option<Self> {
        use smithay::backend::renderer::Offscreen;
        let w = video_info.width();
        let h = video_info.height();
        let rgb: GlesTexture = renderer
            .create_buffer(Fourcc::Abgr8888, (w as i32, h as i32).into())
            .ok()?;
        let y = alloc_plane(render_node, DrmFourcc::R8, w, h)?;
        let uv = alloc_plane(render_node, DrmFourcc::Gr88, w / 2, h / 2)?;
        let shaders = Nv12Shaders::compile(renderer).ok()?;
        Some(Nv12Target {
            rgb,
            y,
            uv,
            width: w,
            height: h,
            video_info,
            shaders,
            gst_allocator: DmaBufAllocator::new(),
        })
    }

    /// Convert the already-rendered RGB texture into the Y and UV plane buffers.
    /// Takes `&self` (the cheap `Dmabuf` handles are cloned to bind) so it fits
    /// the `GsBuffer::to_gs_buffer(&self, ...)` contract.
    pub fn convert(&self, renderer: &mut GlesRenderer) -> Result<(), Box<dyn std::error::Error>> {
        let (w, h) = (self.width as i32, self.height as i32);
        let rgb = self.rgb.clone();
        let shaders = &self.shaders;
        let mut y_plane = self.y.clone();
        let mut uv_plane = self.uv.clone();

        // Y plane: full res.
        {
            let mut target = renderer.bind(&mut y_plane)?;
            let mut frame = renderer.render(&mut target, (w, h).into(), Transform::Normal)?;
            frame.render_texture_from_to(
                &rgb,
                Rectangle::from_size((self.width as f64, self.height as f64).into()),
                Rectangle::from_size((w, h).into()),
                &[Rectangle::from_size((w, h).into())],
                &[],
                Transform::Normal,
                1.0,
                Some(&shaders.y),
                &[],
            )?;
            frame.finish()?.wait()?;
        }
        // UV plane: half res (bilinear downscale subsamples chroma).
        {
            let (uw, uh) = (w / 2, h / 2);
            let mut target = renderer.bind(&mut uv_plane)?;
            let mut frame = renderer.render(&mut target, (uw, uh).into(), Transform::Normal)?;
            frame.render_texture_from_to(
                &rgb,
                Rectangle::from_size((self.width as f64, self.height as f64).into()),
                Rectangle::from_size((uw, uh).into()),
                &[Rectangle::from_size((uw, uh).into())],
                &[],
                Transform::Normal,
                1.0,
                Some(&shaders.uv),
                &[],
            )?;
            frame.finish()?.wait()?;
        }
        Ok(())
    }

    /// Export the two planes as a single NV12 gst buffer (2 memories).
    pub fn to_gst_buffer(&self) -> Result<GstBuffer, Box<dyn std::error::Error>> {
        let mut buf = GstBuffer::new();
        let y_stride = self.y.strides().next().unwrap_or(self.width) as i32;
        let uv_stride = self.uv.strides().next().unwrap_or(self.width) as i32;
        let y_size = (y_stride as usize) * (self.height as usize);

        {
            let b = buf.get_mut().unwrap();
            for plane in [&self.y, &self.uv] {
                plane.handles().for_each(|handle| {
                    let fd = handle.as_raw_fd();
                    let size = smithay::reexports::rustix::fs::seek(
                        &handle.as_fd(),
                        smithay::reexports::rustix::fs::SeekFrom::End(0),
                    )
                    .unwrap() as usize;
                    let mem = unsafe {
                        self.gst_allocator
                            .alloc_with_flags(fd, size, FdMemoryFlags::DONT_CLOSE)
                            .expect("alloc dmabuf memory")
                    };
                    b.append_memory(mem);
                });
            }
            VideoMeta::add_full(
                b,
                gst_video::VideoFrameFlags::empty(),
                VideoFormat::Nv12,
                self.width,
                self.height,
                &[0usize, y_size],
                &[y_stride, uv_stride],
            )?;
        }
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::renderer::setup_renderer;
    use crate::utils::tests::test_init;
    use smithay::backend::renderer::{Bind, Frame};
    use smithay::utils::Rectangle;

    #[test]
    fn test_nv12() {
        test_init();
        let render_node =
            DrmNode::from_path("/dev/dri/renderD129").expect("render node"); // Intel on pve
        let mut renderer = setup_renderer(Some(render_node));
        let (w, h) = (64u32, 64u32);
        let caps = gst_video::VideoCapsBuilder::new()
            .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .format(VideoFormat::DmaDrm)
            .field("drm-format", "NV12")
            .width(w as i32)
            .height(h as i32)
            .pixel_aspect_ratio(1.into())
            .framerate(gst::Fraction::new(30, 1))
            .build();
        let video_info = VideoInfoDmaDrm::from_caps(&caps).expect("video info");
        let tgt = Nv12Target::new(&mut renderer, render_node, video_info).expect("nv12 target");

        // Render a solid colour into the RGB intermediate: R=48 G=96 B=192 (the
        // value used in the va/cuda crossing tests -> expected Y ~= 96).
        {
            let mut rgb = tgt.rgb.clone();
            let mut target = renderer.bind(&mut rgb).expect("bind rgb");
            let mut frame = renderer
                .render(&mut target, (w as i32, h as i32).into(), Transform::Normal)
                .expect("render");
            frame
                .clear(
                    [48.0 / 255.0, 96.0 / 255.0, 192.0 / 255.0, 1.0].into(),
                    &[Rectangle::from_size((w as i32, h as i32).into())],
                )
                .expect("clear");
            frame.finish().expect("finish").wait().expect("wait");
        }

        tgt.convert(&mut renderer).expect("convert");
        let buf = tgt.to_gst_buffer().expect("gst buffer");

        // Read back the Y plane (memory 0) and check luma ~= 96.
        let mem0 = buf.peek_memory(0);
        let mapped = mem0.map_readable().expect("map Y");
        let data = mapped.as_slice();
        let y_stride = tgt.y.strides().next().unwrap() as usize;
        let mut sum = 0u64;
        let n = 16usize;
        for i in 0..n {
            sum += data[i * y_stride + 10] as u64;
        }
        let avg = (sum / n as u64) as i32;
        println!("nv12 Y avg = {avg} (expected ~96)");
        assert!((avg - 96).abs() <= 8, "Y luma off: {avg}");
        drop(mapped);

        // UV plane (memory 1): interleaved Cb,Cr. Expected for R48/G96/B192 (BT.601):
        // U(Cb) ~= 177, V(Cr) ~= 100. Check both bytes are present (order-agnostic).
        let uv = buf.peek_memory(1).map_readable().expect("map UV");
        let uvd = uv.as_slice();
        let (b0, b1) = (uvd[0] as i32, uvd[1] as i32);
        println!("nv12 UV[0]={b0} UV[1]={b1} (expected Cb~177, Cr~100)");
        let near = |a: i32, t: i32| (a - t).abs() <= 10;
        let cb_then_cr = near(b0, 177) && near(b1, 100);
        let cr_then_cb = near(b0, 100) && near(b1, 177);
        assert!(
            cb_then_cr || cr_then_cb,
            "UV chroma off: [{b0}, {b1}]"
        );
        // NV12 byte order must be Cb (U) then Cr (V).
        assert!(cb_then_cr, "UV channel order is Cr,Cb — flip shader .r/.g");
    }
}
