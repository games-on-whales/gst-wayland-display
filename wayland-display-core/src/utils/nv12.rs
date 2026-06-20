//! GLES RGB->NV12 conversion.
//!
//! GLES can only render RGBA, so to emit NV12 we render the scene to an RGB
//! texture and then run two passes that write the luma (Y) and chroma (UV)
//! planes of an NV12 buffer:
//!   - Y plane: full resolution, single channel (R8), luma per pixel.
//!   - UV plane: half resolution, two channels (GR88), interleaved Cb/Cr; the
//!     half-res render with bilinear sampling does the 4:2:0 chroma subsample.
//!
//! The draw uses our own raw GLES2 program + fullscreen quad (via
//! `GlesFrame::with_context`) rather than smithay's `render_texture_from_to`
//! with a custom `GlesTexProgram` -- the latter hangs the draw on radv (AMD),
//! while the default-shader render path (used elsewhere) works there. Doing the
//! draw ourselves is portable across anv/radv.
//!
//! Each plane is a single-plane DMABuf bound as a normal GLES render target via
//! smithay's `Bind<Dmabuf>`; the two are exported as a 2-memory NV12 gst buffer.

use gst::Buffer as GstBuffer;
use gst_video::{VideoFormat, VideoInfoDmaDrm, VideoMeta};
use gstreamer_allocators::{DmaBufAllocator, FdMemoryFlags};
use smithay::backend::allocator::dmabuf::{Dmabuf, DmabufAllocator};
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags};
use smithay::backend::allocator::{Allocator, Buffer};
use smithay::backend::drm::DrmNode;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexture, ffi};
use smithay::backend::renderer::{Bind, Frame, Renderer};
use smithay::reexports::drm::buffer::DrmFourcc;
use smithay::reexports::gbm::Modifier;
use smithay::utils::{Rectangle, Transform};
use std::ffi::CString;
use std::os::fd::{AsFd, AsRawFd};

use crate::utils::allocator::new_gbm_device;

const VERT: &str = "\
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() { v_uv = a_uv; gl_Position = vec4(a_pos, 0.0, 1.0); }
";

// BT.601 limited-range RGB->YCbCr. Mirrors the matrix VA/CUDA converters use.
const FRAG_Y: &str = "\
precision mediump float;
uniform sampler2D tex;
varying vec2 v_uv;
void main() {
    vec3 c = texture2D(tex, v_uv).rgb;
    float y = 0.257 * c.r + 0.504 * c.g + 0.098 * c.b + 0.0625;
    gl_FragColor = vec4(y, y, y, 1.0);
}
";

// The UV plane is rendered as a single-channel R8 target W wide x H/2 tall, with
// the interleaved NV12 layout produced in the shader: even columns hold Cb, odd
// columns hold Cr. This avoids GR88, which Nvidia cannot render to (it exposes no
// GR88 render modifier), so all planes are R8 and importable on every vendor.
// u_w is the plane width in pixels; highp is required so floor(v_uv.x*u_w) is
// exact up to 4K widths.
const FRAG_UV: &str = "\
precision highp float;
uniform sampler2D tex;
uniform float u_w;
varying vec2 v_uv;
void main() {
    float xi = floor(gl_FragCoord.x);      // output column 0..W-1 (window coords)
    float parity = mod(xi, 2.0);           // 0 -> Cb, 1 -> Cr
    float cx = floor(xi * 0.5);            // chroma column 0..W/2-1
    float sx = (2.0 * cx + 1.0) / u_w;     // luma center: averages 2 luma texels
    vec3 c = texture2D(tex, vec2(sx, v_uv.y)).rgb;
    float cb = -0.148 * c.r - 0.291 * c.g + 0.439 * c.b + 0.5;
    float cr =  0.439 * c.r - 0.368 * c.g - 0.071 * c.b + 0.5;
    float o = (parity < 0.5) ? cb : cr;
    gl_FragColor = vec4(o, o, o, 1.0);
}
";

// Fullscreen quad: clip-space pos (-1..1) + uv (0..1). uv.y flipped so the
// sampled texture is upright in the output plane.
#[rustfmt::skip]
const QUAD: [f32; 24] = [
    -1.0, -1.0, 0.0, 1.0,
     1.0, -1.0, 1.0, 1.0,
    -1.0,  1.0, 0.0, 0.0,
    -1.0,  1.0, 0.0, 0.0,
     1.0, -1.0, 1.0, 1.0,
     1.0,  1.0, 1.0, 0.0,
];

#[derive(Debug, Clone, Copy)]
struct GlProgram {
    program: u32,
    a_pos: u32,
    a_uv: u32,
    u_tex: i32,
    u_w: i32,
}

#[derive(Debug, Clone, Copy)]
struct GlState {
    y: GlProgram,
    uv: GlProgram,
    vbo: u32,
}

/// Read a GLES info log (shader compile or program link) into a String. `get` is
/// the matching `glGet{Shader,Program}InfoLog` call, already bound to its object.
unsafe fn info_log(get: impl FnOnce(i32, *mut i32, *mut u8)) -> String {
    let mut buf = [0u8; 512];
    let mut len = 0i32;
    get(512, &mut len as *mut i32, buf.as_mut_ptr());
    String::from_utf8_lossy(&buf[..len.max(0) as usize]).into_owned()
}

unsafe fn compile(gl: &ffi::Gles2, frag: &str) -> GlProgram {
    let mk = |ty: u32, src: &str| -> u32 {
        let s = gl.CreateShader(ty);
        let csrc = CString::new(src).unwrap();
        let ptr = csrc.as_ptr();
        gl.ShaderSource(s, 1, &ptr, std::ptr::null());
        gl.CompileShader(s);
        s
    };
    let vs = mk(ffi::VERTEX_SHADER, VERT);
    let fs = mk(ffi::FRAGMENT_SHADER, frag);
    for (label, sh) in [("vert", vs), ("frag", fs)] {
        let mut ok = 0i32;
        gl.GetShaderiv(sh, ffi::COMPILE_STATUS, &mut ok);
        if ok == 0 {
            let log = info_log(|n, len, ptr| gl.GetShaderInfoLog(sh, n, len, ptr as *mut _));
            tracing::error!("nv12 {label} shader compile failed: {log}");
        }
    }
    let program = gl.CreateProgram();
    gl.AttachShader(program, vs);
    gl.AttachShader(program, fs);
    gl.LinkProgram(program);
    let mut linked = 0i32;
    gl.GetProgramiv(program, ffi::LINK_STATUS, &mut linked);
    gl.DeleteShader(vs);
    gl.DeleteShader(fs);
    let cpos = CString::new("a_pos").unwrap();
    let cuv = CString::new("a_uv").unwrap();
    if linked == 0 {
        let log = info_log(|n, len, ptr| gl.GetProgramInfoLog(program, n, len, ptr as *mut _));
        tracing::error!("nv12 program link failed: {log}");
    }
    let ctex = CString::new("tex").unwrap();
    let cw = CString::new("u_w").unwrap();
    GlProgram {
        program,
        a_pos: gl.GetAttribLocation(program, cpos.as_ptr()) as u32,
        a_uv: gl.GetAttribLocation(program, cuv.as_ptr()) as u32,
        u_tex: gl.GetUniformLocation(program, ctex.as_ptr()),
        u_w: gl.GetUniformLocation(program, cw.as_ptr()),
    }
}

unsafe fn draw(gl: &ffi::Gles2, prog: &GlProgram, vbo: u32, tex: u32, w: i32, h: i32) {
    gl.Viewport(0, 0, w, h);
    gl.Disable(ffi::BLEND);
    gl.UseProgram(prog.program);
    gl.BindBuffer(ffi::ARRAY_BUFFER, vbo);
    let stride = 4 * std::mem::size_of::<f32>() as i32;
    gl.EnableVertexAttribArray(prog.a_pos);
    gl.VertexAttribPointer(
        prog.a_pos,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        stride,
        std::ptr::null(),
    );
    gl.EnableVertexAttribArray(prog.a_uv);
    gl.VertexAttribPointer(
        prog.a_uv,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        stride,
        (2 * std::mem::size_of::<f32>()) as *const _,
    );
    gl.ActiveTexture(ffi::TEXTURE0);
    gl.BindTexture(ffi::TEXTURE_2D, tex);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
    gl.Uniform1i(prog.u_tex, 0);
    if prog.u_w >= 0 {
        gl.Uniform1f(prog.u_w, w as f32);
    }
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
}

fn alloc_plane(
    render_node: DrmNode,
    fourcc: DrmFourcc,
    w: u32,
    h: u32,
    mods: &[Modifier],
) -> Option<Dmabuf> {
    let gbm = new_gbm_device(render_node)?;
    let allocator = GbmAllocator::new(gbm, GbmBufferFlags::RENDERING);
    let mut dma = DmabufAllocator(allocator);
    // Try the GPU's supported render modifiers in order; Nvidia rejects EGLImage
    // import of INVALID/LINEAR planes and needs an explicit block-linear modifier,
    // so we allocate only with what the renderer reports it can render to. For the
    // i915 Y-tiled modifier GBM often needs the 4-tiled code (same workaround as the
    // RGB path). LINEAR is only an empty-list safety net.
    let mut tries: Vec<Modifier> = Vec::new();
    for m in mods {
        tries.push(*m);
        if *m == Modifier::I915_y_tiled {
            tries.push(Modifier::from(0x0100000000000009));
        }
    }
    if tries.is_empty() {
        tries.push(Modifier::Linear);
    }
    for m in tries {
        if let Ok(buf) = dma.create_buffer(w, h, fourcc, &[m]) {
            return Some(buf);
        }
    }
    None
}

/// Holds the intermediate RGB target, the two NV12 plane buffers, the compiled
/// GL programs, and the negotiated NV12 video info.
#[derive(Debug, Clone)]
pub struct Nv12Target {
    pub rgb: GlesTexture,
    pub y: Dmabuf,
    pub uv: Dmabuf,
    pub width: u32,
    pub height: u32,
    pub video_info: VideoInfoDmaDrm,
    gl: GlState,
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
            .create_buffer(
                smithay::backend::allocator::Fourcc::Abgr8888,
                (w as i32, h as i32).into(),
            )
            .ok()?;
        // Allocate the planes with a modifier the renderer can actually render to
        // on this GPU. Nvidia rejects EGLImage import of INVALID/LINEAR dmabufs and
        // needs an explicit block-linear modifier; Intel needs its tiled modifier;
        // AMD accepts LINEAR. Query the supported render formats and pick from them.
        let formats =
            <GlesRenderer as Bind<Dmabuf>>::supported_formats(renderer).unwrap_or_default();
        let mods_for = |fourcc: DrmFourcc| -> Vec<Modifier> {
            let mut v: Vec<Modifier> = formats
                .iter()
                .filter(|f| f.code == fourcc)
                .map(|f| f.modifier)
                .collect();
            // Prefer explicit modifiers; push INVALID to the back (Nvidia rejects it).
            v.sort_by_key(|m| *m == Modifier::Invalid);
            v
        };
        // Both planes are R8: Y is W x H, UV is W x H/2 holding interleaved Cb/Cr.
        // Allocate only with modifiers the renderer can render to. If the negotiated
        // modifier is one of them, honor it first so the produced buffer matches the
        // advertised caps; otherwise ignore it (e.g. a LINEAR-negotiated buffer on
        // Nvidia, which GBM allocates but the EGL cannot render to) and use a
        // render-capable modifier.
        let mut r8_mods = mods_for(DrmFourcc::R8);
        let negotiated = Modifier::from(video_info.modifier());
        if r8_mods.contains(&negotiated) {
            r8_mods.retain(|m| *m != negotiated);
            r8_mods.insert(0, negotiated);
        }
        let y = alloc_plane(render_node, DrmFourcc::R8, w, h, &r8_mods)?;
        let uv = alloc_plane(render_node, DrmFourcc::R8, w, h / 2, &r8_mods)?;

        let gl = renderer
            .with_context(|gl| unsafe {
                let mut vbo = 0u32;
                gl.GenBuffers(1, &mut vbo);
                gl.BindBuffer(ffi::ARRAY_BUFFER, vbo);
                gl.BufferData(
                    ffi::ARRAY_BUFFER,
                    std::mem::size_of_val(&QUAD) as isize,
                    QUAD.as_ptr() as *const _,
                    ffi::STATIC_DRAW,
                );
                GlState {
                    y: compile(gl, FRAG_Y),
                    uv: compile(gl, FRAG_UV),
                    vbo,
                }
            })
            .ok()?;

        Some(Nv12Target {
            rgb,
            y,
            uv,
            width: w,
            height: h,
            video_info,
            gl,
            gst_allocator: DmaBufAllocator::new(),
        })
    }

    /// Convert the already-rendered RGB texture into the Y and UV plane buffers.
    pub fn convert(&self, renderer: &mut GlesRenderer) -> Result<(), Box<dyn std::error::Error>> {
        let (w, h) = (self.width as i32, self.height as i32);
        let tex = self.rgb.tex_id();
        let gl_state = self.gl;
        let mut y_plane = self.y.clone();
        let mut uv_plane = self.uv.clone();

        {
            let mut target = renderer.bind(&mut y_plane)?;
            let mut frame = renderer.render(&mut target, (w, h).into(), Transform::Normal)?;
            frame.with_context(|gl| unsafe { draw(gl, &gl_state.y, gl_state.vbo, tex, w, h) })?;
            frame.finish()?.wait()?;
        }
        {
            let (uw, uh) = (w, h / 2);
            let mut target = renderer.bind(&mut uv_plane)?;
            let mut frame = renderer.render(&mut target, (uw, uh).into(), Transform::Normal)?;
            frame
                .with_context(|gl| unsafe { draw(gl, &gl_state.uv, gl_state.vbo, tex, uw, uh) })?;
            frame.finish()?.wait()?;
        }
        Ok(())
    }

    /// Reconstruct the two R8 planes into a single 2-plane NV12 `Dmabuf` (Y as
    /// plane 0, UV as plane 1, each keeping its own fd). Test-only: it stands in
    /// for the production path (`to_gst_buffer` followed by the late dmabuf->CUDA
    /// element's `reconstruct_nv12_dmabuf`) so a unit test can feed the CUDA
    /// converter a 2-plane NV12 dmabuf without a running pipeline.
    #[cfg(all(test, feature = "cuda"))]
    pub fn as_nv12_dmabuf(&self) -> Option<Dmabuf> {
        use smithay::backend::allocator::dmabuf::DmabufFlags;
        let modifier = self.y.format().modifier;
        let y_stride = self.y.strides().next()?;
        let uv_stride = self.uv.strides().next()?;
        let y_fd = self.y.handles().next()?.as_fd().try_clone_to_owned().ok()?;
        let uv_fd = self
            .uv
            .handles()
            .next()?
            .as_fd()
            .try_clone_to_owned()
            .ok()?;
        let mut builder = Dmabuf::builder(
            (self.width as i32, self.height as i32),
            DrmFourcc::Nv12,
            modifier,
            DmabufFlags::empty(),
        );
        builder.add_plane(y_fd, 0, 0, y_stride);
        builder.add_plane(uv_fd, 1, 0, uv_stride);
        builder.build()
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
        let node_path =
            std::env::var("NV12_TEST_NODE").unwrap_or_else(|_| "/dev/dri/renderD129".into());
        let render_node = DrmNode::from_path(&node_path).expect("render node");
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
        assert_eq!(buf.n_memory(), 2, "NV12 buffer should have Y + UV memories");

        // CPU readback is only meaningful for a LINEAR layout; tiled/block-linear
        // planes (Intel, Nvidia) read back as raw tiles, so byte-exact assertions
        // only run on LINEAR. Correctness on tiled vendors is covered by the e2e
        // encode test. Reaching here without an EGLImage error already proves the
        // converter renders to importable planes on this GPU.
        let linear = u64::from(tgt.y.format().modifier) == u64::from(Modifier::Linear);

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
        println!("nv12 Y avg = {avg} (expected ~96), linear={linear}");
        drop(mapped);

        let uv = buf.peek_memory(1).map_readable().expect("map UV");
        let uvd = uv.as_slice();
        let (b0, b1) = (uvd[0] as i32, uvd[1] as i32);
        println!("nv12 UV[0]={b0} UV[1]={b1} (expected Cb~177, Cr~100), linear={linear}");
        let near = |a: i32, t: i32| (a - t).abs() <= 12;
        if linear {
            assert!((avg - 96).abs() <= 8, "Y luma off: {avg}");
            assert!(
                near(b0, 177) && near(b1, 100),
                "UV chroma off: [{b0}, {b1}]"
            );
        }
    }
}
