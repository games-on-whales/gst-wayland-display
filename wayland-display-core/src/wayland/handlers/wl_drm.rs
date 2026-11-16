use crate::{
    comp::{State, debug_fail_dmabuf_import},
    wayland::protocols::wl_drm::{DrmHandler, ImportError, delegate_wl_drm},
};
use smithay::backend::renderer::ImportDma;
use smithay::{
    backend::allocator::dmabuf::Dmabuf, reexports::wayland_server::protocol::wl_buffer::WlBuffer,
    wayland::dmabuf::DmabufGlobal,
};

impl DrmHandler<()> for State {
    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: Dmabuf,
    ) -> Result<(), ImportError> {
        // Fault-injection hook: see `debug_fail_dmabuf_import` doc comment
        // (comp/mod.rs). Test-only; never set in production.
        if debug_fail_dmabuf_import() {
            self.note_renderer_degraded(
                "wl_drm dmabuf import failed: WOLF_DEBUG_FAIL_DMABUF_IMPORT injected failure",
            );
            return Err(ImportError::Failed);
        }

        match self.renderer.import_dmabuf(&dmabuf, None) {
            Ok(_) => {
                // A successful import proves the client's GPU path is alive -- clear any
                // active degradation condition.
                self.clear_renderer_degraded();
                Ok(())
            }
            Err(err) => {
                // wl_drm buffers are dmabuf-backed (mesa's protocol); a failed import here is
                // the same GPU-renderer degradation as the dmabuf handler. Enters/refreshes
                // the degradation condition for a downstream fail-closed hook.
                self.note_renderer_degraded(&format!("wl_drm dmabuf import failed: {err:?}"));
                Err(ImportError::Failed)
            }
        }
    }

    fn buffer_created(&mut self, _buffer: WlBuffer, _result: ()) {}
}

delegate_wl_drm!(State);
