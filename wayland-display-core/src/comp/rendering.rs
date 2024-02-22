use std::time::{Duration, Instant};

use smithay::{
    desktop::space::render_output,
    backend::renderer::{
        damage::Error as DTRError,
        element::{
            memory::MemoryRenderBufferRenderElement, surface::WaylandSurfaceRenderElement,
            RenderElementStates,
        },
        gles::GlesRenderer,
        Bind, ExportMem, ImportAll, ImportMem, Renderer, Unbind,
    },
    input::pointer::CursorImageStatus,
    render_elements,
    utils::{Physical, Rectangle},
};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::damage::RenderOutputResult;
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::element::memory::MemoryBuffer;
use smithay::reexports::drm::buffer::DrmFourcc;
use smithay::utils::Size;
use tracing::debug;

use super::State;

pub const CURSOR_DATA_BYTES: &[u8] = include_bytes!("../../resources/cursor.rgba");

render_elements! {
    CursorElement<R> where R: Renderer + ImportAll + ImportMem;
    Surface=WaylandSurfaceRenderElement<R>,
    Memory=MemoryRenderBufferRenderElement<R>
}

impl State {
    pub fn create_frame(
        &mut self,
    ) -> Result<
        (
            gst::Buffer,
            RenderOutputResult,
        ),
        DTRError<GlesRenderer>,
    > {
        assert!(self.output.is_some());
        assert!(self.dtr.is_some());
        assert!(self.video_info.is_some());
        assert!(self.renderbuffer.is_some());

        let elements =
            if Instant::now().duration_since(self.last_pointer_movement) < Duration::from_secs(5) {
                match &self.cursor_state {
                    CursorImageStatus::Named(cursor_icon) => vec![CursorElement::Memory(
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
                            .map_err(DTRError::Rendering)?,
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

        self.renderer
            .bind(self.renderbuffer.clone().unwrap())
            .map_err(DTRError::Rendering)?;
        let render_output_result = render_output(
            self.output.as_ref().unwrap(),
            &mut self.renderer,
            1.0,
            0,
            [&self.space],
            &*elements,
            self.dtr.as_mut().unwrap(),
            [0.0, 0.0, 0.0, 1.0],
        )?;

        let mapping = self
            .renderer
            .copy_framebuffer(Rectangle::from_loc_and_size(
                (0, 0),
                (
                    self.video_info.as_ref().unwrap().width() as i32,
                    self.video_info.as_ref().unwrap().height() as i32,
                ),
            ), Fourcc::try_from(self.video_info.as_ref().unwrap().format().to_fourcc()).unwrap_or(Fourcc::Abgr8888))
            .expect("Failed to export framebuffer");
        let map = self
            .renderer
            .map_texture(&mapping)
            .expect("Failed to download framebuffer");

        let buffer = {
            let mut buffer = gst::Buffer::with_size(map.len()).expect("failed to create buffer");
            {
                let buffer = buffer.get_mut().unwrap();

                let mut vframe = gst_video::VideoFrameRef::from_buffer_ref_writable(
                    buffer,
                    self.video_info.as_ref().unwrap(),
                )
                    .unwrap();
                let plane_data = vframe.plane_data_mut(0).unwrap();
                plane_data.clone_from_slice(map);
            }

            buffer
        };
        self.renderer.unbind().map_err(DTRError::Rendering)?;
        Ok((buffer, render_output_result))
    }
}
