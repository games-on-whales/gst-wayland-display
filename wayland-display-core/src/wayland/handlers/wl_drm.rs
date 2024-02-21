use crate::{comp::State, wayland::protocols::wl_drm::{delegate_wl_drm, DrmHandler, ImportError}};
use smithay::{
    backend::{allocator::dmabuf::Dmabuf, drm::DrmNode},
    reexports::wayland_server::{protocol::wl_buffer::WlBuffer, Resource},
    wayland::dmabuf::DmabufGlobal,
};
use smithay::backend::renderer::ImportDma;

impl<D: 'static> DrmHandler<Option<D>> for State {
    fn dmabuf_imported(
        &mut self,
        global: &DmabufGlobal,
        dmabuf: Dmabuf,
    ) -> Result<Option<D>, ImportError> {
        todo!("self.renderer.import_dmabuf(&dmabuf, None) ???");
        Err(ImportError::Failed)
    }

    fn buffer_created(&mut self, buffer: WlBuffer, result: Option<D>) {
        todo!()
    }
}

delegate_wl_drm!(State);
