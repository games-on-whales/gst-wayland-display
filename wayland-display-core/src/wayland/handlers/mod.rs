pub mod color_management;
pub mod compositor;
pub mod data_device;
pub mod dmabuf;
pub mod drm_syncobj;
pub mod frog_color_management;
pub mod pointer_constraints;
pub mod relative_pointer;
pub mod seat;
pub mod shm;
pub mod wl_drm;
pub mod xdg;

// Dispatch for every smithay protocol, through each one's `Dispatch2` user data.
smithay::delegate_dispatch2!(crate::comp::State);
