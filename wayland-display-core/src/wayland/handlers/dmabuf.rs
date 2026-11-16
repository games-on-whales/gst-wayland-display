use smithay::{
    backend::{allocator::dmabuf::Dmabuf, renderer::ImportDma},
    wayland::dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier},
};

use crate::comp::{State, debug_fail_dmabuf_import};

impl DmabufHandler for State {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: Dmabuf,
        notifier: ImportNotifier,
    ) {
        // Fault-injection hook: see `debug_fail_dmabuf_import` doc comment
        // (comp/mod.rs). Test-only; never set in production.
        if debug_fail_dmabuf_import() {
            self.note_renderer_degraded(
                "dmabuf import failed: WOLF_DEBUG_FAIL_DMABUF_IMPORT injected failure",
            );
            notifier.failed();
            return;
        }

        match self.renderer.import_dmabuf(&dmabuf, None) {
            Ok(_) => {
                // A successful import proves the client's GPU path is alive -- clear any
                // active degradation condition.
                self.clear_renderer_degraded();
                let _ = notifier.successful::<State>();
            }
            Err(err) => {
                // A client's dmabuf failed to import on the GPU renderer -- the frame it
                // backs will not composite. Enters/refreshes the degradation condition so
                // a downstream consumer can fail a session that requires hardware rendering.
                self.note_renderer_degraded(&format!("dmabuf import failed: {err:?}"));
                notifier.failed();
            }
        }
    }
}
