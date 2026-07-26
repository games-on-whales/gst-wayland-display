use std::time::{Duration, Instant};

use super::State;
use crate::utils::allocator::GsBuffer;
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::gles::{GlesError, GlesRenderer, GlesTarget};
use smithay::{
    backend::renderer::{
        Color32F, ExportMem, ImportAll, ImportMem, Renderer,
        damage::{Error as OutputDamageTrackerError, RenderOutputResult},
        element::{
            Id, Kind, memory::MemoryRenderBufferRenderElement, solid::SolidColorRenderElement,
            surface::WaylandSurfaceRenderElement,
        },
        utils::CommitCounter,
    },
    desktop::space::render_output,
    input::pointer::CursorImageStatus,
    render_elements,
    utils::{Physical, Point, Rectangle, Size},
};
use std::sync::atomic::{AtomicBool, Ordering};

pub const CURSOR_DATA_BYTES: &[u8] = include_bytes!("../../resources/cursor.rgba");

render_elements! {
    CursorElement<R> where R: Renderer + ImportAll + ImportMem;
    Surface=WaylandSurfaceRenderElement<R>,
    Memory=MemoryRenderBufferRenderElement<R>,
    // HDR render-path spike (WOLF_HDR_SPIKE): synthetic >1.0 brightness bars.
    Solid=SolidColorRenderElement
}

/// HDR render-path spike: linear brightness levels (multiples of SDR reference white) for the
/// synthetic test bars. The >1.0 levels are the highlights the spike is trying to carry through
/// the fp16 render target. Shared by the bar geometry ([`hdr_spike_bars`]) and the GLES readback
/// ([`hdr_spike_gles_readback`]) so they sample the same bars.
const HDR_SPIKE_LEVELS: [f32; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];

/// HDR render-path spike: bars occupy the top ~15% of the output height.
const HDR_SPIKE_BAR_HEIGHT_PCT: i32 = 15;

/// HDR render-path spike: the GLES readback is logged only once (a few-pixel probe, not every
/// frame). Set the first time a frame is rendered into an fp16 target under WOLF_HDR_SPIKE.
static HDR_READBACK_DONE: AtomicBool = AtomicBool::new(false);

/// HDR render-path spike: build a row of full-height-ish vertical bars across the top ~15% of
/// the `width`x`height` output, each filled with a LINEAR color that is a multiple of SDR
/// reference white ([`HDR_SPIKE_LEVELS`]). `Color32F` is unclamped f32, so the >1.0 bars carry
/// true HDR highlights into the fp16 render target -> Vulkan converter -> P010 PQ. Returns the
/// bars as `CursorElement::Solid` so they can be prepended as the topmost elements. Only ever
/// called when `WOLF_HDR_SPIKE` is set.
fn hdr_spike_bars<R: Renderer + ImportAll + ImportMem>(
    width: i32,
    height: i32,
) -> Vec<CursorElement<R>> {
    let bar_h = (height * HDR_SPIKE_BAR_HEIGHT_PCT / 100).max(1);
    let n = HDR_SPIKE_LEVELS.len() as i32;
    let bar_w = (width / n).max(1);
    HDR_SPIKE_LEVELS
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = i as i32 * bar_w;
            let geo = Rectangle::new(
                Point::<i32, Physical>::from((x, 0)),
                Size::<i32, Physical>::from((bar_w, bar_h)),
            );
            CursorElement::Solid(SolidColorRenderElement::new(
                Id::new(),
                geo,
                CommitCounter::default(),
                Color32F::new(v, v, v, 1.0),
                Kind::Unspecified,
            ))
        })
        .collect()
}

/// Decode an IEEE-754 binary16 (half) bit pattern to `f32`. Used to interpret the fp16 RGBA
/// render-target pixels read back by [`hdr_spike_gles_readback`] (no `half` crate dependency).
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let mag = if exp == 0 {
        // subnormal: 2^-14 * (mant/1024)
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        // normal: 2^(exp-15) * (1 + mant/1024)
        (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -mag } else { mag }
}

/// HDR render-path spike: ISOLATION probe -- after the GLES renderer has drawn the scene
/// (incl. the >1.0 brightness bars) into the fp16 RGBA dmabuf but BEFORE the Vulkan converter
/// runs, read back one pixel at the center of each bar straight off the GLES framebuffer and
/// log its float RGB. This answers "did GlesRenderer store >1.0, or clamp to 1.0?" independent
/// of the downstream Vulkan/P010 path.
///
/// Readback uses smithay's `ExportMem` (`copy_framebuffer` + `map_texture`) with the fp16
/// `Abgr16161616f` fourcc -> GL `(RGBA16F, RGBA, HALF_FLOAT)`, which `glReadPixels` finishes
/// synchronously (no manual GL sync, no dmabuf mmap). NOTE: smithay's `map_texture` maps only
/// `w*h*4` bytes, but an fp16 pixel is 8 bytes, so a 1x1 region would return just R,G. We read a
/// **2x1** region: the PBO holds the left pixel's full RGBA in bytes 0..8, and `map_texture`
/// maps exactly those 8 bytes -- the full half-float RGBA of the bar-center pixel.
fn hdr_spike_gles_readback(
    renderer: &mut GlesRenderer,
    target: &GlesTarget<'_>,
    width: i32,
    height: i32,
) {
    let bar_h = (height * HDR_SPIKE_BAR_HEIGHT_PCT / 100).max(1);
    let n = HDR_SPIKE_LEVELS.len() as i32;
    let bar_w = (width / n).max(1);
    let cy = (bar_h / 2).clamp(0, (height - 1).max(0));
    for (i, &v) in HDR_SPIKE_LEVELS.iter().enumerate() {
        // Left pixel of a 2x1 probe at the bar center; clamp so the 2-wide region stays in bounds.
        let cx = (i as i32 * bar_w + bar_w / 2).clamp(0, (width - 2).max(0));
        let region = Rectangle::new(Point::from((cx, cy)), Size::from((2, 1)));
        match renderer.copy_framebuffer(target, region, Fourcc::Abgr16161616f) {
            Ok(mapping) => match renderer.map_texture(&mapping) {
                Ok(bytes) if bytes.len() >= 8 => {
                    let r = half_to_f32(u16::from_ne_bytes([bytes[0], bytes[1]]));
                    let g = half_to_f32(u16::from_ne_bytes([bytes[2], bytes[3]]));
                    let b = half_to_f32(u16::from_ne_bytes([bytes[4], bytes[5]]));
                    tracing::info!(
                        "HDR_SPIKE gles_readback bar[{v}x] expected=({v},{v},{v}) got=({r:.4},{g:.4},{b:.4})"
                    );
                }
                Ok(bytes) => tracing::warn!(
                    "HDR_SPIKE gles_readback bar[{v}x]: short mapping ({} bytes)",
                    bytes.len()
                ),
                Err(e) => {
                    tracing::warn!("HDR_SPIKE gles_readback bar[{v}x]: map_texture failed: {e:?}")
                }
            },
            Err(e) => {
                tracing::warn!("HDR_SPIKE gles_readback bar[{v}x]: copy_framebuffer failed: {e:?}")
            }
        }
    }
}

impl State {
    pub fn create_frame(
        &mut self,
    ) -> Result<(gst::Buffer, RenderOutputResult), OutputDamageTrackerError<GlesError>> {
        assert!(self.output.is_some());
        assert!(self.dtr.is_some());
        assert!(self.video_info.is_some());
        assert!(self.output_buffer.is_some());

        let mut elements =
            if Instant::now().duration_since(self.last_pointer_movement) < Duration::from_secs(5) {
                match &self.cursor_state {
                CursorImageStatus::Named(_cursor_icon) => vec![CursorElement::Memory(
                    // TODO: icon?
                    MemoryRenderBufferRenderElement::from_buffer(
                        &mut self.renderer,
                        self.pointer_location.to_physical_precise_round(1),
                        &self.cursor_element,
                        None,
                        None,
                        None,
                        Kind::Cursor,
                    )
                    .map_err(OutputDamageTrackerError::Rendering)?,
                )],
                CursorImageStatus::Surface(wl_surface) => {
                    smithay::backend::renderer::element::surface::render_elements_from_surface_tree(
                        &mut self.renderer,
                        wl_surface,
                        self.pointer_location.to_physical_precise_round(1),
                        1.,
                        1.,
                        Kind::Cursor,
                    )
                }
                CursorImageStatus::Hidden => vec![],
            }
            } else {
                vec![]
            };

        // HDR render-path spike: prepend synthetic >1.0 brightness bars across the top of the
        // output as the topmost elements (so client surfaces never occlude them). Gated behind
        // WOLF_HDR_SPIKE -- unset = exactly the elements built above (no bars, no behavior change).
        if std::env::var("WOLF_HDR_SPIKE").is_ok() {
            if let Some(vi) = self.video_info.as_ref() {
                let mut bars = hdr_spike_bars(vi.width() as i32, vi.height() as i32);
                bars.append(&mut elements);
                elements = bars;
            }
        }

        let mut output_buffer = self.output_buffer.clone().expect("Output buffer not set");

        // Capture the RGBA render-target fourcc BEFORE bind() borrows `output_buffer` (the
        // returned GlesTarget holds a mutable borrow of the buffer's dmabuf for the rest of the
        // frame). Used by the HDR-spike readback below to confirm the target is fp16.
        let render_rgba_fourcc = output_buffer.render_rgba_fourcc();

        let mut target = output_buffer
            .bind(&mut self.renderer)
            .map_err(OutputDamageTrackerError::Rendering)?;

        let render_output_result = render_output(
            self.output.as_ref().unwrap(),
            &mut self.renderer,
            &mut target,
            1.0,
            0,
            [&self.space],
            &*elements,
            self.dtr.as_mut().unwrap(),
            [0.0, 0.0, 0.0, 1.0],
        )?;

        // The NV12/VulkanImage paths import this GLES render target as external Vulkan
        // memory and sample it during `to_gs_buffer`. Waiting in the caller after
        // `create_frame` returns is too late: the converter has already submitted its
        // Vulkan read by then, so the GLES write and the Vulkan read are only ordered if
        // the driver happens to serialize the two APIs -- which nothing guarantees.
        // Complete the GLES render here, before any hardware conversion or hand-off
        // consumes the target.
        render_output_result
            .sync
            .wait()
            .expect("Error during render_result.sync");
        // HDR render-path spike ISOLATION probe: read the fp16 RGBA dmabuf straight off the GLES
        // framebuffer (after render_output, before the Vulkan converter in to_gs_buffer) and log
        // the bar-center float values -- proving whether GLES kept >1.0 or clamped. Gated behind
        // WOLF_HDR_SPIKE, only on an fp16 target, and logged once (cheap, a handful of pixels).
        if std::env::var("WOLF_HDR_SPIKE").is_ok()
            && render_rgba_fourcc == Some(Fourcc::Abgr16161616f)
        {
            if let Some(vi) = self.video_info.as_ref() {
                let (w, h) = (vi.width() as i32, vi.height() as i32);
                if !HDR_READBACK_DONE.swap(true, Ordering::Relaxed) {
                    hdr_spike_gles_readback(&mut self.renderer, &target, w, h);
                }
            }
        }

        // WOLF_HDR_CM: per-frame PQ-passthrough flag (true when the active surface's last
        // committed buffer was a 10-bit, already-PQ BT.2020 client buffer). Always false unless
        // WOLF_HDR_CM is set, so the converter behaves identically to before by default.
        let pq_passthrough = self.current_input_is_pq;
        match self.output_buffer.clone().unwrap().to_gs_buffer(
            &mut target,
            &mut self.renderer,
            pq_passthrough,
        ) {
            Ok(buffer) => Ok((buffer, render_output_result)),
            Err(e) => {
                tracing::warn!("Failed to convert buffer to gst buffer: {:?}", e);
                Err(OutputDamageTrackerError::Rendering(GlesError::MappingError))
            }
        }
    }
}
