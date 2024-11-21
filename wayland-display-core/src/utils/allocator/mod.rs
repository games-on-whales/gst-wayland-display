use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::gles::{GlesRenderbuffer, GlesRenderer};
use smithay::backend::renderer::Offscreen;

#[derive(Debug, Default, Clone)]
pub struct GLESAllocator {
    buffer: Option<GlesRenderbuffer>,
}

#[derive(Debug, Clone)]
pub enum AllocatorType {
    GLES(GLESAllocator),
}

pub trait Allocator<Renderer, Target> {
    fn alloc_buffer(&mut self, renderer: &mut Renderer, format: Fourcc, width: i32, height: i32);

    fn get_buffer(&self) -> Option<Target>;
}

impl Allocator<GlesRenderer, GlesRenderbuffer> for AllocatorType {
    fn alloc_buffer(
        &mut self,
        renderer: &mut GlesRenderer,
        format: Fourcc,
        width: i32,
        height: i32,
    ) {
        match self {
            AllocatorType::GLES(allocator) => {
                let result = renderer.create_buffer(format, (width, height).into());
                match result {
                    Ok(buffer) => allocator.buffer = Some(buffer),
                    Err(_) => allocator.buffer = None,
                }
            }
        }
    }

    fn get_buffer(&self) -> Option<GlesRenderbuffer> {
        match self {
            AllocatorType::GLES(allocator) => allocator.buffer.clone(),
        }
    }
}
