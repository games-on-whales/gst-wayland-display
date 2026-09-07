use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::State;
use crate::utils::allocator::GsBuffer;
use smithay::backend::renderer::gles::GlesError;
use smithay::{
    backend::renderer::{
        ImportAll, ImportMem, Renderer,
        damage::{Error as OutputDamageTrackerError, RenderOutputResult},
        element::{
            Kind, memory::MemoryRenderBufferRenderElement, surface::WaylandSurfaceRenderElement,
        },
    },
    desktop::space::render_output,
    input::pointer::{CursorImageAttributes, CursorImageStatus},
    render_elements,
    wayland::compositor::with_states,
};

pub const CURSOR_DATA_BYTES: &[u8] = include_bytes!("../../resources/cursor.rgba");

render_elements! {
    CursorElement<R> where R: Renderer + ImportAll + ImportMem;
    Surface=WaylandSurfaceRenderElement<R>,
    Memory=MemoryRenderBufferRenderElement<R>
}

impl State {
    pub fn create_frame(
        &mut self,
    ) -> Result<(gst::Buffer, RenderOutputResult), OutputDamageTrackerError<GlesError>> {
        assert!(self.output.is_some());
        assert!(self.dtr.is_some());
        assert!(self.video_info.is_some());
        assert!(self.output_buffer.is_some());

        let elements = if Instant::now().duration_since(self.last_pointer_movement)
            < Duration::from_secs(5)
        {
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
                    // A client submits its cursor image together with a HOTSPOT:
                    // the point inside the image that must sit on the pointer
                    // (an arrow's tip, a crosshair's centre). Render at
                    // pointer - hotspot, exactly as smithay's own anvil example
                    // does. Without this, every client whose hotspot is not
                    // (0,0) has its cursor drawn offset by the hotspot —
                    // invisible for conventional arrows (hotspot ~0,0), glaring
                    // for centre-hotspot cursors (Mindustry ships 64x64 cursors
                    // with a (32,32) hotspot: drawn 32px down-right of the real
                    // pointer, so clicks land up-left of the visible cursor).
                    let hotspot = with_states(wl_surface, |states| {
                        states
                            .data_map
                            .get::<Mutex<CursorImageAttributes>>()
                            .map(|attrs| attrs.lock().unwrap().hotspot)
                            .unwrap_or_else(|| (0, 0).into())
                    });
                    smithay::backend::renderer::element::surface::render_elements_from_surface_tree(
                        &mut self.renderer,
                        wl_surface,
                        (self.pointer_location - hotspot.to_f64()).to_physical_precise_round(1),
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
