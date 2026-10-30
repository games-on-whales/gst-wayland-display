//! Minimal Wayland client for the GPU tests: maps one xdg toplevel with a solid-colour SHM
//! buffer, optionally declaring ST 2084 PQ / BT.2020 through `wp_color_management_v1`.

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
#[allow(dead_code)]
pub fn run_pq_client(
    socket: std::path::PathBuf,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    run_client(socket, stop, [0, 0, 0], true)
}

/// Connect to `socket`, map a toplevel filled with `rgb` (sRGB, 8-bit), optionally declaring
/// PQ, and keep the connection alive until `stop` is set.
pub fn run_client(
    socket: std::path::PathBuf,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    rgb: [u8; 3],
    pq: bool,
) {
    run_switching_client(
        socket,
        stop,
        rgb,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(pq)),
    )
}

/// Like [`run_client`], but follows `pq` while running: each change declares ST 2084 PQ /
/// BT.2020 or unsets the image description (back to undeclared sRGB) and commits, as a
/// game toggling its HDR setting does.
pub fn run_switching_client(
    socket: std::path::PathBuf,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    rgb: [u8; 3],
    pq: std::sync::Arc<std::sync::atomic::AtomicBool>,
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
    let color = state.color.clone();
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
    let mut file = tempfile::tempfile().unwrap();
    {
        use std::io::Write;
        // XRGB8888 is B, G, R, X in memory.
        let pixel = [rgb[2], rgb[1], rgb[0], 0xff];
        let row: Vec<u8> = pixel
            .iter()
            .copied()
            .cycle()
            .take((w * 4) as usize)
            .collect();
        for _ in 0..h {
            file.write_all(&row).unwrap();
        }
    }
    let pool = shm.create_pool(file.as_fd(), w * h * 4, &qh, ());
    let buffer = pool.create_buffer(0, w, h, w * 4, wl_shm::Format::Xrgb8888, &qh, ());

    let mut cm_surface = None;
    let mut declared = None;
    while !stop.load(Ordering::Relaxed) {
        let want = pq.load(Ordering::Relaxed);
        if declared != Some(want) {
            if want {
                let color = color
                    .as_ref()
                    .expect("wp_color_manager_v1 (dynamic-hdr off?)");
                let params = color.create_parametric_creator(&qh, ());
                params.set_tf_named(wp_color_manager_v1::TransferFunction::St2084Pq);
                params.set_primaries_named(wp_color_manager_v1::Primaries::Bt2020);
                let desc = params.create(&qh, ());
                cm_surface
                    .get_or_insert_with(|| color.get_surface(&surface, &qh, ()))
                    .set_image_description(&desc, wp_color_manager_v1::RenderIntent::Perceptual);
            } else if let Some(cm_surface) = &cm_surface {
                cm_surface.unset_image_description();
            }
            // The declaration latches with the next commit.
            surface.attach(Some(&buffer), 0, 0);
            surface.damage_buffer(0, 0, w, h);
            surface.commit();
            declared = Some(want);
        }
        let _ = queue.dispatch_pending(&mut state);
        let _ = conn.flush();
        if let Some(guard) = conn.prepare_read() {
            let _ = guard.read();
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
