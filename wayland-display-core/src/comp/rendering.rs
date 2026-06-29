use std::time::{Duration, Instant};

use super::State;
use crate::utils::allocator::GsBuffer;
use smithay::backend::renderer::gles::GlesError;
use smithay::{
    backend::renderer::{
        Color32F, ImportAll, ImportMem, Renderer,
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

pub const CURSOR_DATA_BYTES: &[u8] = include_bytes!("../../resources/cursor.rgba");

render_elements! {
    CursorElement<R> where R: Renderer + ImportAll + ImportMem;
    Surface=WaylandSurfaceRenderElement<R>,
    Memory=MemoryRenderBufferRenderElement<R>,
    // HDR render-path spike (WOLF_HDR_SPIKE): synthetic >1.0 brightness bars.
    Solid=SolidColorRenderElement
}

/// HDR render-path spike: build a row of full-height-ish vertical bars across the top ~15% of
/// the `width`x`height` output, each filled with a LINEAR color that is a multiple of SDR
/// reference white (0.5, 1.0, 2.0, 4.0, 8.0, 12.0). `Color32F` is unclamped f32, so the >1.0
/// bars carry true HDR highlights into the fp16 render target -> Vulkan converter -> P010 PQ.
/// Returns the bars as `CursorElement::Solid` so they can be prepended as the topmost elements.
/// Only ever called when `WOLF_HDR_SPIKE` is set.
fn hdr_spike_bars<R: Renderer + ImportAll + ImportMem>(
    width: i32,
    height: i32,
) -> Vec<CursorElement<R>> {
    const LEVELS: [f32; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
    let bar_h = (height * 15 / 100).max(1);
    let n = LEVELS.len() as i32;
    let bar_w = (width / n).max(1);
    LEVELS
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

        match self
            .output_buffer
            .clone()
            .unwrap()
            .to_gs_buffer(&mut target, &mut self.renderer)
        {
            Ok(buffer) => Ok((buffer, render_output_result)),
            Err(e) => {
                tracing::warn!("Failed to convert buffer to gst buffer: {:?}", e);
                Err(OutputDamageTrackerError::Rendering(GlesError::MappingError))
            }
        }
    }
}
