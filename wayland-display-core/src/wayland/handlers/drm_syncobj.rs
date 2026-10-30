use smithay::wayland::drm_syncobj::{DrmSyncobjHandler, DrmSyncobjState};

use crate::comp::State;

impl DrmSyncobjHandler for State {
    fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState> {
        self.drm_syncobj_state.as_mut()
    }
}
