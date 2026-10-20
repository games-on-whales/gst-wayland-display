//! Minimal `wp_color_management_v1` client for the dynamic-HDR GPU test: maps one
//! xdg toplevel with an SHM buffer whose surface declares ST 2084 PQ / BT.2020.

use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use wayland_client::protocol::{
    wl_buffer, wl_compositor, wl_registry, wl_shm, wl_shm_pool, wl_surface,
};
use wayland_client::{Connection, Dispatch, QueueHandle, delegate_noop};
use wayland_protocols::wp::color_management::v1::client::{
    wp_color_management_surface_v1, wp_color_manager_v1, wp_image_description_creator_params_v1,
    wp_image_description_v1,
};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

#[derive(Default)]
struct State {
    compositor: Option<wl_compositor::WlCompositor>,
    shm: Option<wl_shm::WlShm>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    color: Option<wp_color_manager_v1::WpColorManagerV1>,
    configured: Option<u32>,
}

impl Dispatch<wl_registry::WlRegistry, ()> for State {
    fn event(
        s: &mut Self,
        r: &wl_registry::WlRegistry,
        e: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = e
        {
            match interface.as_str() {
                "wl_compositor" => s.compositor = Some(r.bind(name, version.min(4), qh, ())),
                "wl_shm" => s.shm = Some(r.bind(name, 1, qh, ())),
                "xdg_wm_base" => s.wm_base = Some(r.bind(name, 1, qh, ())),
                "wp_color_manager_v1" => s.color = Some(r.bind(name, 1, qh, ())),
                _ => {}
            }
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for State {
    fn event(
        _: &mut Self,
        b: &xdg_wm_base::XdgWmBase,
        e: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = e {
            b.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for State {
    fn event(
        s: &mut Self,
        _: &xdg_surface::XdgSurface,
        e: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = e {
            s.configured = Some(serial);
        }
    }
}

delegate_noop!(State: ignore wl_compositor::WlCompositor);
delegate_noop!(State: ignore wl_surface::WlSurface);
delegate_noop!(State: ignore wl_shm::WlShm);
delegate_noop!(State: ignore wl_shm_pool::WlShmPool);
delegate_noop!(State: ignore wl_buffer::WlBuffer);
delegate_noop!(State: ignore xdg_toplevel::XdgToplevel);
delegate_noop!(State: ignore wp_color_manager_v1::WpColorManagerV1);
delegate_noop!(State: wp_image_description_creator_params_v1::WpImageDescriptionCreatorParamsV1);
delegate_noop!(State: ignore wp_image_description_v1::WpImageDescriptionV1);
delegate_noop!(State: wp_color_management_surface_v1::WpColorManagementSurfaceV1);

/// Connect to `socket`, map a PQ-declaring toplevel, and keep the connection alive
/// (dispatching) until `stop` is set.
pub fn run_pq_client(
    socket: std::path::PathBuf,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::Ordering;
    let conn = Connection::from_socket(UnixStream::connect(socket).expect("connect")).unwrap();
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();
    conn.display().get_registry(&qh, ());
    let mut state = State::default();
    queue.roundtrip(&mut state).unwrap();
    let compositor = state.compositor.clone().expect("wl_compositor");
    let wm_base = state.wm_base.clone().expect("xdg_wm_base");
    let color = state
        .color
        .clone()
        .expect("wp_color_manager_v1 (dynamic-hdr off?)");
    let shm = state.shm.clone().expect("wl_shm");

    let surface = compositor.create_surface(&qh, ());
    let xdg = wm_base.get_xdg_surface(&surface, &qh, ());
    let _toplevel = xdg.get_toplevel(&qh, ());
    surface.commit();
    while state.configured.is_none() {
        queue.blocking_dispatch(&mut state).unwrap();
    }
    xdg.ack_configure(state.configured.unwrap());

    let (w, h) = (320, 240);
    let file = tempfile::tempfile().unwrap();
    file.set_len((w * h * 4) as u64).unwrap();
    let pool = shm.create_pool(file.as_fd(), w * h * 4, &qh, ());
    let buffer = pool.create_buffer(0, w, h, w * 4, wl_shm::Format::Xrgb8888, &qh, ());

    let params = color.create_parametric_creator(&qh, ());
    params.set_tf_named(wp_color_manager_v1::TransferFunction::St2084Pq);
    params.set_primaries_named(wp_color_manager_v1::Primaries::Bt2020);
    let desc = params.create(&qh, ());
    let cm_surface = color.get_surface(&surface, &qh, ());
    cm_surface.set_image_description(&desc, wp_color_manager_v1::RenderIntent::Perceptual);
    surface.attach(Some(&buffer), 0, 0);
    surface.damage_buffer(0, 0, w, h);
    surface.commit();
    conn.flush().unwrap();

    while !stop.load(Ordering::Relaxed) {
        let _ = queue.dispatch_pending(&mut state);
        let _ = conn.flush();
        if let Some(guard) = conn.prepare_read() {
            let _ = guard.read();
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
