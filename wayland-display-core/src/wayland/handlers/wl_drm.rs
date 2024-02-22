use crate::{comp::State, wayland::protocols::wl_drm::{delegate_wl_drm, DrmHandler, ImportError}};
use smithay::{
    backend::{allocator::dmabuf::Dmabuf, drm::DrmNode},
    reexports::wayland_server::{protocol::wl_buffer::WlBuffer, Resource},
    wayland::dmabuf::DmabufGlobal,
};
use smithay::backend::renderer::ImportDma;

impl DrmHandler<()> for State {
    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: Dmabuf,
    ) -> Result<(), ImportError> {
        self.renderer.import_dmabuf(&dmabuf, None).map(|_| ()).map_err(|_| ImportError::Failed)
    }

    fn buffer_created(&mut self, buffer: WlBuffer, result: ()) {}
}

delegate_wl_drm!(State);
