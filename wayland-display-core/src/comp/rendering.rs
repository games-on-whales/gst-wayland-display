use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{State, effective_render_size, fit_offset_physical, window_fullscreen_fit};
use crate::utils::allocator::GsBuffer;
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::gles::{
    GlesError, GlesFrame, GlesRenderer, GlesTarget, GlesTexProgram, Uniform, UniformName,
    UniformType,
};
use smithay::utils::user_data::UserDataMap;
use smithay::{
    backend::renderer::{
        Color32F, ExportMem, ImportAll, ImportMem, Renderer,
        damage::{Error as OutputDamageTrackerError, RenderOutputResult},
        element::{
            AsRenderElements, Element, Id, Kind, RenderElement, UnderlyingStorage,
            memory::MemoryRenderBufferRenderElement,
            solid::SolidColorRenderElement,
            surface::WaylandSurfaceRenderElement,
            utils::{Relocate, RelocateRenderElement, RescaleRenderElement},
        },
        utils::{CommitCounter, DamageSet, OpaqueRegions},
    },
    desktop::Window,
    input::pointer::{CursorImageAttributes, CursorImageStatus},
    render_elements,
    utils::{Buffer as BufferCoords, Logical, Physical, Point, Rectangle, Scale, Size, Transform},
    wayland::compositor::with_states,
};
use std::sync::atomic::{AtomicBool, Ordering};

pub const CURSOR_DATA_BYTES: &[u8] = include_bytes!("../../resources/cursor.rgba");

render_elements! {
    CursorElement<R> where R: Renderer + ImportAll + ImportMem;
    Surface=WaylandSurfaceRenderElement<R>,
    Memory=MemoryRenderBufferRenderElement<R>,
    // HDR render-path spike (WOLF_HDR_SPIKE): synthetic >1.0 brightness bars.
    Solid=SolidColorRenderElement
}

/// A window's surface elements, through its fullscreen fit (see [`WindowElements`]).
type SpaceElement = WindowElements;

render_elements! {
    FrameElement<=GlesRenderer>;
    Plain=CursorElement<GlesRenderer>,
    Space=SpaceElement,
    // Content drawn in the other transfer than the frame's working space, converted.
    Converted=Converted<CursorElement<GlesRenderer>>,
    ConvertedSpace=Converted<SpaceElement>,
}

/// Shared head of the HDR texture shaders below (smithay's texture-program contract: the
/// `//_DEFINES_` line, `tex`, `alpha`, `v_coords`, optional `tint`), plus the transfer
/// helpers. `ref_white_scale` is the SDR reference white as a fraction of 10,000 nits.
macro_rules! hdr_texture_shader {
    ($main:literal) => {
        concat!(
            r#"#version 100

//_DEFINES_

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

uniform float alpha;
uniform float ref_white_scale;
varying vec2 v_coords;

#if defined(DEBUG_FLAGS)
uniform float tint;
#endif

const float m1 = 0.1593017578125;
const float m2 = 78.84375;
const float c1 = 0.8359375;
const float c2 = 18.8515625;
const float c3 = 18.6875;

vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 linear_to_srgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

// ST 2084: normalised linear light (1.0 = 10,000 nits) <-> PQ code values.
vec3 pq_oetf(vec3 y) {
    vec3 p = pow(clamp(y, 0.0, 1.0), vec3(m1));
    return pow((c1 + c2 * p) / (1.0 + c3 * p), vec3(m2));
}

vec3 pq_eotf(vec3 e) {
    vec3 ep = pow(clamp(e, 0.0, 1.0), vec3(1.0 / m2));
    return pow(max(ep - c1, 0.0) / (c2 - c3 * ep), vec3(1.0 / m1));
}

vec3 convert(vec3 rgb);

void main() {
    vec4 color = texture2D(tex, v_coords);
#if defined(NO_ALPHA)
    color.a = 1.0;
#endif
    // Premultiplied: convert the straight colour, then premultiply again.
    vec3 rgb = color.a > 0.0 ? color.rgb / color.a : vec3(0.0);
    color = vec4(convert(clamp(rgb, 0.0, 1.0)) * color.a, color.a) * alpha;

#if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
#endif

    gl_FragColor = color;
}
"#,
            $main
        )
    };
}

/// SDR content into a PQ frame: sRGB EOTF, BT.709 -> BT.2020 primaries, SDR white to the
/// reference white, then the ST 2084 inverse EOTF.
const PQ_ENCODE_SRGB_FRAG: &str = hdr_texture_shader!(
    r#"
vec3 convert(vec3 rgb) {
    // Column-major: each vec3 is a column of the row-major BT.709 -> BT.2020 matrix.
    mat3 bt709_to_bt2020 = mat3(
        0.627404, 0.069097, 0.016391,
        0.329283, 0.919540, 0.088013,
        0.043313, 0.011362, 0.895595);
    return pq_oetf(bt709_to_bt2020 * srgb_to_linear(rgb) * ref_white_scale);
}
"#
);

/// PQ content into an SDR frame, the tone map of the converter's `rgba_hdr_to_*_sdr`
/// shaders: PQ EOTF, BT.2020 -> BT.709 primaries, reference white to 1.0, luminance roll-off
/// of everything above 0.8 into [0.8, 1.0), sRGB OETF.
const PQ_TONEMAP_SRGB_FRAG: &str = hdr_texture_shader!(
    r#"
float roll_off(float x) {
    return x <= 0.8 ? x : 0.8 + 0.2 * (1.0 - exp(-(x - 0.8) / 0.2));
}

vec3 convert(vec3 rgb) {
    // Column-major: each vec3 is a column of the row-major BT.2020 -> BT.709 matrix.
    mat3 bt2020_to_bt709 = mat3(
         1.660491, -0.124550, -0.018151,
        -0.587641,  1.132900, -0.100579,
        -0.072850, -0.008349,  1.118730);
    vec3 rel = max(bt2020_to_bt709 * pq_eotf(rgb) / ref_white_scale, vec3(0.0));
    float y = dot(rel, vec3(0.2126, 0.7152, 0.0722));
    rel *= y > 0.0 ? roll_off(y) / y : 0.0;
    return linear_to_srgb(clamp(rel, 0.0, 1.0));
}
"#
);

/// The CPU twin of [`PQ_ENCODE_SRGB_FRAG`] / [`PQ_TONEMAP_SRGB_FRAG`] for single-pixel-buffer
/// surfaces, which smithay draws as solid colours rather than through a texture shader.
/// `color` is premultiplied, like the buffer's.
pub(crate) fn convert_solid(color: [f32; 4], to_pq: bool, ref_white_scale: f32) -> [f32; 4] {
    const M1: f32 = 0.159_301_76;
    const M2: f32 = 78.843_75;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_563;
    const C3: f32 = 18.687_5;
    let mat = |m: [[f32; 3]; 3], v: [f32; 3]| {
        [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
    };
    let a = color[3];
    if a <= 0.0 {
        return [0.0; 4];
    }
    let rgb = [0, 1, 2].map(|i| (color[i] / a).clamp(0.0, 1.0));
    let out = if to_pq {
        let lin = rgb.map(|c| {
            if c < 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        });
        let bt2020 = mat(
            [
                [0.627404, 0.329283, 0.043313],
                [0.069097, 0.919540, 0.011362],
                [0.016391, 0.088013, 0.895595],
            ],
            lin,
        );
        bt2020.map(|y| {
            let p = (y * ref_white_scale).clamp(0.0, 1.0).powf(M1);
            ((C1 + C2 * p) / (1.0 + C3 * p)).powf(M2)
        })
    } else {
        let lin = rgb.map(|e| {
            let ep = e.powf(1.0 / M2);
            ((ep - C1).max(0.0) / (C2 - C3 * ep)).powf(1.0 / M1)
        });
        let mut rel = mat(
            [
                [1.660491, -0.587641, -0.072850],
                [-0.124550, 1.132_9, -0.008349],
                [-0.018151, -0.100579, 1.118_73],
            ],
            lin,
        )
        .map(|c| (c / ref_white_scale).max(0.0));
        let y = 0.2126 * rel[0] + 0.7152 * rel[1] + 0.0722 * rel[2];
        let rolled = if y <= 0.8 {
            y
        } else {
            0.8 + 0.2 * (1.0 - (-(y - 0.8) / 0.2).exp())
        };
        if y > 0.0 {
            rel = rel.map(|c| c * rolled / y);
        }
        rel.map(|c| {
            let c = c.clamp(0.0, 1.0);
            if c < 0.003_130_8 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        })
    };
    [out[0] * a, out[1] * a, out[2] * a, a]
}

/// A render element drawn through one of the HDR texture shaders into a frame whose working
/// space is the other transfer: an sRGB surface or cursor in a PQ frame (where it would read
/// as up to 10,000 nits) is PQ-encoded at the SDR reference white, and a PQ surface in an SDR
/// frame (where it would show as washed-out gamma) is tone-mapped to SDR.
pub struct Converted<E> {
    inner: E,
    program: GlesTexProgram,
    ref_white_scale: f32,
}

impl<E: Element> Element for Converted<E> {
    fn id(&self) -> &Id {
        self.inner.id()
    }
    fn current_commit(&self) -> CommitCounter {
        self.inner.current_commit()
    }
    fn location(&self, scale: Scale<f64>) -> Point<i32, Physical> {
        self.inner.location(scale)
    }
    fn src(&self) -> Rectangle<f64, BufferCoords> {
        self.inner.src()
    }
    fn transform(&self) -> Transform {
        self.inner.transform()
    }
    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.inner.geometry(scale)
    }
    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        self.inner.damage_since(scale, commit)
    }
    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        self.inner.opaque_regions(scale)
    }
    fn alpha(&self) -> f32 {
        self.inner.alpha()
    }
    fn kind(&self) -> Kind {
        self.inner.kind()
    }
    fn is_framebuffer_effect(&self) -> bool {
        self.inner.is_framebuffer_effect()
    }
}

impl<E: RenderElement<GlesRenderer>> RenderElement<GlesRenderer> for Converted<E> {
    fn draw(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), GlesError> {
        frame.override_default_tex_program(
            self.program.clone(),
            vec![Uniform::new("ref_white_scale", self.ref_white_scale)],
        );
        let result = self
            .inner
            .draw(frame, src, dst, damage, opaque_regions, cache);
        frame.clear_tex_program_override();
        result
    }

    fn capture_framebuffer(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        cache: &UserDataMap,
    ) -> Result<(), GlesError> {
        self.inner.capture_framebuffer(frame, src, dst, cache)
    }

    fn underlying_storage(&self, _renderer: &mut GlesRenderer) -> Option<UnderlyingStorage<'_>> {
        // Scanning the buffer out directly would skip the conversion.
        None
    }
}

/// [`FrameElement`]s live in RENDER space (the output's mode); this maps them into the
/// encode-sized framebuffer with one rescale/relocate pair, so the cursor stays inside the
/// scaled scene and what the user points at is what hit-testing (in render space) resolves.
type ScaledFrameElement = RelocateRenderElement<RescaleRenderElement<FrameElement>>;

render_elements! {
    /// The final element list handed to the damage tracker: the scaled scene plus the
    /// encode-space overlays (the HDR spike bars), which must NOT be scaled.
    OutputElement<=GlesRenderer>;
    Scene=ScaledFrameElement,
    Overlay=CursorElement<GlesRenderer>,
}

/// One mapped window's surface tree (plus its popups), mapped from the size the client
/// actually committed into the size it was configured at — see
/// [`super::window_fullscreen_fit`]. The wrappers are an exact identity for every window
/// that already fills its configure.
type WindowElements =
    RelocateRenderElement<RescaleRenderElement<WaylandSurfaceRenderElement<GlesRenderer>>>;

/// One mapped window, resolved for this frame: where its elements are built and how they
/// are mapped into its configured box. Snapshotted out of the space *before* the renderer
/// is borrowed mutably to build the elements.
struct FitWindow {
    window: Window,
    /// The window's render location, relative to the output's region (render space).
    loc: Point<i32, Logical>,
    /// Aspect-preserving fit scale and centring offset — see [`window_fit_physical`].
    scale: f64,
    offset: Point<i32, Physical>,
}

/// How to map a `render`-sized scene into an `encode`-sized framebuffer: an aspect-preserving
/// upscale plus the centring offset that turns the leftover into symmetric letterbox/pillarbox
/// bars (painted by the clear colour).
///
/// Returns `(1.0, (0,0))` -- an exact identity -- whenever the two sizes match, which is the
/// default configuration (no render size requested). `RescaleRenderElement` at scale 1.0 with
/// origin `(0,0)` and `RelocateRenderElement` at offset `(0,0)` are both pass-throughs on
/// integer geometry, so the default path composites exactly as it did before.
fn scene_transform(
    encode: Size<i32, Physical>,
    render: Size<i32, Physical>,
) -> (f64, Point<i32, Physical>) {
    if render.w <= 0 || render.h <= 0 || render == encode {
        return (1.0, Point::from((0, 0)));
    }
    let scale = f64::min(
        encode.w as f64 / render.w as f64,
        encode.h as f64 / render.h as f64,
    );
    let offset = Point::from((
        ((encode.w as f64 - render.w as f64 * scale) / 2.0).round() as i32,
        ((encode.h as f64 - render.h as f64 * scale) / 2.0).round() as i32,
    ));
    (scale, offset)
}

/// [`super::window_fullscreen_fit`] for one window, with the centring offset converted from
/// logical into the PHYSICAL pixels the render elements live in.
///
/// The problem the fit solves: a fullscreen toplevel that commits a surface SMALLER than its
/// configured size — a native Wayland game whose in-menu "resolution" setting picked, say,
/// 1280x720 on a 1920x1080 output — is composited by the space at that size at its map
/// location, i.e. 1:1 in the top-left corner with black everywhere else. Every other
/// compositor scales it to fill the output; gwd did not, which is what made an in-app
/// resolution choice look broken.
fn window_fit_physical(
    window: &Window,
    output_logical: Size<i32, Logical>,
    output_scale: f64,
) -> (f64, Point<i32, Physical>) {
    let (scale, offset) = window_fullscreen_fit(window, output_logical);
    (scale, fit_offset_physical(offset, output_scale))
}

/// HDR render-path spike: linear brightness levels (multiples of SDR reference white) for the
/// synthetic test bars. The >1.0 levels are the highlights the spike is trying to carry through
/// the fp16 render target. Shared by the bar geometry ([`hdr_spike_bars`]) and the GLES readback
/// ([`hdr_spike_gles_readback`]) so they sample the same bars.
const HDR_SPIKE_LEVELS: [f32; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];

/// HDR render-path spike: bars occupy the top ~15% of the output height.
const HDR_SPIKE_BAR_HEIGHT_PCT: i32 = 15;

/// HDR render-path spike: the GLES readback is logged only once (a few-pixel probe, not every
/// frame). Set the first time a frame is rendered into an fp16 target under WOLF_HDR_SPIKE.
static HDR_READBACK_DONE: AtomicBool = AtomicBool::new(false);

/// HDR render-path spike: build a row of full-height-ish vertical bars across the top ~15% of
/// the `width`x`height` output, each filled with a LINEAR color that is a multiple of SDR
/// reference white ([`HDR_SPIKE_LEVELS`]). `Color32F` is unclamped f32, so the >1.0 bars carry
/// true HDR highlights into the fp16 render target -> Vulkan converter -> P010 PQ. Returns the
/// bars as `CursorElement::Solid` so they can be prepended as the topmost elements. Only ever
/// called when `WOLF_HDR_SPIKE` is set.
fn hdr_spike_bars<R: Renderer + ImportAll + ImportMem>(
    width: i32,
    height: i32,
) -> Vec<CursorElement<R>> {
    let bar_h = (height * HDR_SPIKE_BAR_HEIGHT_PCT / 100).max(1);
    let n = HDR_SPIKE_LEVELS.len() as i32;
    let bar_w = (width / n).max(1);
    HDR_SPIKE_LEVELS
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = i as i32 * bar_w;
            let geo = Rectangle::new(
                Point::<i32, Physical>::from((x, 0)),
                Size::<i32, Physical>::from((bar_w, bar_h)),
            );
            CursorElement::Solid(SolidColorRenderElement::new(
                Id::new(),
                geo,
                CommitCounter::default(),
                Color32F::new(v, v, v, 1.0),
                Kind::Unspecified,
            ))
        })
        .collect()
}

/// Decode an IEEE-754 binary16 (half) bit pattern to `f32`. Used to interpret the fp16 RGBA
/// render-target pixels read back by [`hdr_spike_gles_readback`] (no `half` crate dependency).
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let mag = if exp == 0 {
        // subnormal: 2^-14 * (mant/1024)
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        // normal: 2^(exp-15) * (1 + mant/1024)
        (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -mag } else { mag }
}

/// HDR render-path spike: ISOLATION probe -- after the GLES renderer has drawn the scene
/// (incl. the >1.0 brightness bars) into the fp16 RGBA dmabuf but BEFORE the Vulkan converter
/// runs, read back one pixel at the center of each bar straight off the GLES framebuffer and
/// log its float RGB. This answers "did GlesRenderer store >1.0, or clamp to 1.0?" independent
/// of the downstream Vulkan/P010 path.
///
/// Readback uses smithay's `ExportMem` (`copy_framebuffer` + `map_texture`) with the fp16
/// `Abgr16161616f` fourcc -> GL `(RGBA16F, RGBA, HALF_FLOAT)`, which `glReadPixels` finishes
/// synchronously (no manual GL sync, no dmabuf mmap). NOTE: smithay's `map_texture` maps only
/// `w*h*4` bytes, but an fp16 pixel is 8 bytes, so a 1x1 region would return just R,G. We read a
/// **2x1** region: the PBO holds the left pixel's full RGBA in bytes 0..8, and `map_texture`
/// maps exactly those 8 bytes -- the full half-float RGBA of the bar-center pixel.
fn hdr_spike_gles_readback(
    renderer: &mut GlesRenderer,
    target: &GlesTarget<'_>,
    width: i32,
    height: i32,
) {
    let bar_h = (height * HDR_SPIKE_BAR_HEIGHT_PCT / 100).max(1);
    let n = HDR_SPIKE_LEVELS.len() as i32;
    let bar_w = (width / n).max(1);
    let cy = (bar_h / 2).clamp(0, (height - 1).max(0));
    for (i, &v) in HDR_SPIKE_LEVELS.iter().enumerate() {
        // Left pixel of a 2x1 probe at the bar center; clamp so the 2-wide region stays in bounds.
        let cx = (i as i32 * bar_w + bar_w / 2).clamp(0, (width - 2).max(0));
        let region = Rectangle::new(Point::from((cx, cy)), Size::from((2, 1)));
        match renderer.copy_framebuffer(target, region, Fourcc::Abgr16161616f) {
            Ok(mapping) => match renderer.map_texture(&mapping) {
                Ok(bytes) if bytes.len() >= 8 => {
                    let r = half_to_f32(u16::from_ne_bytes([bytes[0], bytes[1]]));
                    let g = half_to_f32(u16::from_ne_bytes([bytes[2], bytes[3]]));
                    let b = half_to_f32(u16::from_ne_bytes([bytes[4], bytes[5]]));
                    tracing::info!(
                        "HDR_SPIKE gles_readback bar[{v}x] expected=({v},{v},{v}) got=({r:.4},{g:.4},{b:.4})"
                    );
                }
                Ok(bytes) => tracing::warn!(
                    "HDR_SPIKE gles_readback bar[{v}x]: short mapping ({} bytes)",
                    bytes.len()
                ),
                Err(e) => {
                    tracing::warn!("HDR_SPIKE gles_readback bar[{v}x]: map_texture failed: {e:?}")
                }
            },
            Err(e) => {
                tracing::warn!("HDR_SPIKE gles_readback bar[{v}x]: copy_framebuffer failed: {e:?}")
            }
        }
    }
}

/// The compiled HDR texture shaders: SDR -> PQ and PQ -> SDR.
#[derive(Clone)]
pub(crate) struct HdrPrograms {
    encode: GlesTexProgram,
    tonemap: GlesTexProgram,
}

/// How a frame's surfaces are composited: in the frame's own working space, or with the
/// surfaces of the other transfer converted by `program` (`to_pq`: into a PQ frame).
struct Composition {
    /// `None`: the HDR shaders did not compile, so nothing is converted and the whole frame
    /// is handed over as PQ (the converter's whole-frame PQ pipelines), which is right for
    /// the PQ content at least.
    program: Option<GlesTexProgram>,
    to_pq: bool,
    ref_white_scale: f32,
    /// Surfaces holding PQ content (with a buffer), by render element id.
    pq_surfaces: std::collections::HashSet<Id>,
    /// Premultiplied colours of the single-pixel-buffer surfaces, by render element id.
    solids: std::collections::HashMap<Id, [f32; 4]>,
}

impl Composition {
    fn converts(&self, id: &Id) -> bool {
        self.program.is_some() && self.pq_surfaces.contains(id) != self.to_pq
    }

    /// The transfer of the composited frame.
    fn content(&self) -> crate::utils::color::ContentTransfer {
        if self.program.is_none() || self.to_pq {
            crate::utils::color::ContentTransfer::Pq
        } else {
            crate::utils::color::ContentTransfer::Sdr
        }
    }

    fn convert<E: Element>(
        &self,
        element: E,
        plain: impl FnOnce(E) -> FrameElement,
        converted: impl FnOnce(Converted<E>) -> FrameElement,
    ) -> FrameElement {
        if !self.converts(element.id()) {
            return plain(element);
        }
        match self.solids.get(element.id()) {
            Some(color) => {
                // Smithay draws a single-pixel buffer as a solid colour, which bypasses the
                // texture shader: convert the colour itself.
                let mut color = convert_solid(*color, self.to_pq, self.ref_white_scale);
                let alpha = element.alpha();
                color.iter_mut().for_each(|c| *c *= alpha);
                FrameElement::Plain(CursorElement::Solid(SolidColorRenderElement::new(
                    element.id().clone(),
                    element.geometry(Scale::from(1.0)),
                    element.current_commit(),
                    Color32F::from(color),
                    element.kind(),
                )))
            }
            None => converted(Converted {
                inner: element,
                program: self.program.clone().expect("converts() requires a program"),
                ref_white_scale: self.ref_white_scale,
            }),
        }
    }
}

/// Record `states`' surface in `composition`-to-be if it holds a buffer: PQ content, and the
/// colour of a single-pixel buffer.
#[allow(clippy::mutable_key_type)] // smithay's ObjectId has interior mutability that does not participate in Hash/Eq
fn scan_surface(
    surface: &smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
    states: &smithay::wayland::compositor::SurfaceData,
    pq_surfaces: &mut std::collections::HashSet<Id>,
    solids: &mut std::collections::HashMap<Id, [f32; 4]>,
) {
    use smithay::backend::renderer::utils::RendererSurfaceStateUserData;
    let Some(state) = states.data_map.get::<RendererSurfaceStateUserData>() else {
        return;
    };
    let id = Id::from_wayland_resource(surface);
    {
        // Released before `surface_data_visible_hdr`, which takes the same lock.
        let state = state.lock().unwrap();
        let Some(buffer) = state.buffer() else {
            return;
        };
        if let Ok(spb) = smithay::wayland::single_pixel_buffer::get_single_pixel_buffer(buffer) {
            solids.insert(id.clone(), spb.rgba32f());
        }
    }
    if crate::wayland::handlers::color_management::surface_data_visible_hdr(states).is_some() {
        pq_surfaces.insert(id);
    }
}

impl State {
    /// The HDR texture shaders, compiled on first use; `None` (surfaces drawn as-is) if the
    /// GLES driver rejects them.
    pub(crate) fn hdr_programs(&mut self) -> Option<HdrPrograms> {
        self.hdr_programs
            .get_or_insert_with(|| {
                let uniforms = [UniformName::new("ref_white_scale", UniformType::_1f)];
                let compile = |renderer: &mut GlesRenderer, src| {
                    renderer
                        .compile_custom_texture_shader(src, &uniforms)
                        .inspect_err(|err| tracing::warn!(?err, "HDR shader failed to compile"))
                        .ok()
                };
                Some(HdrPrograms {
                    encode: compile(&mut self.renderer, PQ_ENCODE_SRGB_FRAG)?,
                    tonemap: compile(&mut self.renderer, PQ_TONEMAP_SRGB_FRAG)?,
                })
            })
            .clone()
    }

    /// How to composite this frame. Under dynamic HDR the working space follows the output:
    /// with PQ content on an HDR output the frame is PQ and SDR surfaces (and the cursor) are
    /// PQ-encoded; on an SDR output it stays SDR and only the PQ surfaces are tone-mapped, so
    /// SDR content is untouched. `None`: no PQ content, or no conversion possible.
    #[allow(clippy::mutable_key_type)] // smithay's ObjectId has interior mutability that does not participate in Hash/Eq
    fn composition(
        &mut self,
        output_hdr: bool,
        convertible_target: bool,
        cursor_drawn: bool,
    ) -> Option<Composition> {
        if !self.hdr_cm || !convertible_target {
            return None;
        }
        let mut pq_surfaces = std::collections::HashSet::new();
        let mut solids = std::collections::HashMap::new();
        for window in self.space.elements() {
            window.with_surfaces(|surface, states| {
                scan_surface(surface, states, &mut pq_surfaces, &mut solids)
            });
        }
        // The legacy 10-bit PQ guess stands for the active window's root only.
        if let Some(root) = self.active_root_pq() {
            pq_surfaces.insert(Id::from_wayland_resource(&root));
        }
        if let (true, CursorImageStatus::Surface(cursor)) = (cursor_drawn, &self.cursor_state) {
            smithay::wayland::compositor::with_surface_tree_downward(
                cursor,
                (),
                |_, _, _| smithay::wayland::compositor::TraversalAction::DoChildren(()),
                |surface, states, _| scan_surface(surface, states, &mut pq_surfaces, &mut solids),
                |_, _, _| true,
            );
        }
        if pq_surfaces.is_empty() {
            return None;
        }
        let programs = self.hdr_programs();
        Some(Composition {
            program: programs.map(|p| if output_hdr { p.encode } else { p.tonemap }),
            to_pq: output_hdr,
            ref_white_scale: self.vulkan_share.conversion_options().sdr_reference_white / 10_000.0,
            pq_surfaces,
            solids,
        })
    }

    pub fn create_frame(
        &mut self,
    ) -> Result<(gst::Buffer, RenderOutputResult<'_>), OutputDamageTrackerError<GlesError>> {
        assert!(self.output.is_some());
        assert!(self.dtr.is_some());
        assert!(self.video_info.is_some());
        assert!(self.output_buffer.is_some());

        // The scene is composited at the output's scale (the UI scale): the output's LOGICAL
        // size is `mode / ui_scale`, so drawing at that scale puts the scene back at the
        // physical render size -- which is what the rescale below maps into the encode
        // framebuffer -- and a scale-aware client's high-density buffer is sampled 1:1 rather
        // than being squeezed into the logical size first. `space_render_elements` picks this
        // up from the Output on its own; the cursor is built by hand, so it needs it passed
        // in. 1.0 unless a UI scale was requested, i.e. the default path is unchanged.
        let output_scale = self
            .output
            .as_ref()
            .expect("output not set")
            .current_scale()
            .fractional_scale();

        // The cursor lives in RENDER space (its location is `pointer_location`, which input
        // clamps to the output's logical extent), so it is built at the same scale here and
        // scaled below together with the client surfaces.
        let cursor_drawn =
            Instant::now().duration_since(self.last_pointer_movement) < Duration::from_secs(5);
        let elements = if cursor_drawn {
            match &self.cursor_state {
                CursorImageStatus::Named(_cursor_icon) => vec![CursorElement::Memory(
                    // TODO: icon?
                    MemoryRenderBufferRenderElement::from_buffer(
                        &mut self.renderer,
                        self.pointer_location
                            .to_physical_precise_round(output_scale),
                        &self.cursor_element,
                        None,
                        None,
                        None,
                        Kind::Cursor,
                    )
                    .map_err(OutputDamageTrackerError::Rendering)?,
                )],
                CursorImageStatus::Surface(wl_surface) => {
                    // A client submits its cursor image together with a HOTSPOT:
                    // the point inside the image that must sit on the pointer
                    // (an arrow's tip, a crosshair's centre). Render at
                    // pointer - hotspot, exactly as smithay's own anvil example
                    // does. Without this, every client whose hotspot is not
                    // (0,0) has its cursor drawn offset by the hotspot —
                    // invisible for conventional arrows (hotspot ~0,0), glaring
                    // for centre-hotspot cursors (Mindustry ships 64x64 cursors
                    // with a (32,32) hotspot: drawn 32px down-right of the real
                    // pointer, so clicks land up-left of the visible cursor).
                    let hotspot = with_states(wl_surface, |states| {
                        states
                            .data_map
                            .get::<Mutex<CursorImageAttributes>>()
                            .map(|attrs| attrs.lock().unwrap().hotspot)
                            .unwrap_or_else(|| (0, 0).into())
                    });
                    smithay::backend::renderer::element::surface::render_elements_from_surface_tree(
                        &mut self.renderer,
                        wl_surface,
                        (self.pointer_location - hotspot.to_f64())
                            .to_physical_precise_round(output_scale),
                        output_scale,
                        1.,
                        Kind::Cursor,
                    )
                }
                CursorImageStatus::Hidden => vec![],
            }
        } else {
            vec![]
        };

        // Dynamic HDR: the committed output state picks the frame's working space (see
        // `composition`); the converter is told what was actually composited.
        // Read before bind(), whose target borrows the state for the rest of the frame.
        let output_hdr = self.output_hdr();

        let mut output_buffer = self.output_buffer.clone().expect("Output buffer not set");

        // Capture the RGBA render-target fourcc BEFORE bind() borrows `output_buffer` (the
        // returned GlesTarget holds a mutable borrow of the buffer's dmabuf for the rest of the
        // frame). Used by the HDR-spike readback below to confirm the target is fp16.
        let render_rgba_fourcc = output_buffer.render_rgba_fourcc();

        // The fp16 target of a converter carries PQ code values unclamped; raw RGB output
        // carries the frame's transfer as metadata (`utils::source_color`).
        let composition = self.composition(
            output_hdr,
            render_rgba_fourcc.is_none_or(|fourcc| fourcc == Fourcc::Abgr16161616f),
            cursor_drawn,
        );
        let color = crate::utils::color::FrameColor {
            output_hdr,
            content: composition.as_ref().map_or(
                crate::utils::color::ContentTransfer::Sdr,
                Composition::content,
            ),
        };
        self.last_frame_color = color;
        let mut frame_elements: Vec<FrameElement> = elements
            .into_iter()
            .map(|element| match (&element, &composition) {
                // The spike's bars are linear values, not sRGB.
                (CursorElement::Solid(_), _) | (_, None) => FrameElement::Plain(element),
                (_, Some(composition)) => {
                    composition.convert(element, FrameElement::Plain, FrameElement::Converted)
                }
            })
            .collect();

        let mut target = output_buffer
            .bind(&mut self.renderer)
            .map_err(OutputDamageTrackerError::Rendering)?;

        // Client surfaces, in render space. Built per window (rather than via
        // `desktop::space::space_render_elements` / `render_output`) so each window's
        // elements can be wrapped in its OWN fit transform before the whole scene is wrapped
        // in the render->encode one, and so they can be grouped with the cursor.
        //
        // The traversal mirrors `Space::render_elements_for_region`, which this replaces:
        // z-order back-to-front from `space.elements()`, reversed so the topmost window's
        // elements come first (render-element lists are front-to-back), each window's
        // location taken relative to the output's region. gwd maps exactly one output at
        // (0,0) and uses no layer-shell surfaces, so nothing else `space_render_elements`
        // does applies here.
        let output = self.output.clone().expect("output not set");
        let region = self.space.output_geometry(&output);
        let windows: Vec<FitWindow> = match region {
            Some(region) => self
                .space
                .elements()
                .rev()
                .filter_map(|window| {
                    let bbox = self.space.element_bbox(window)?;
                    if !region.overlaps(bbox) {
                        return None;
                    }
                    let loc = self.space.element_location(window)?
                        - smithay::desktop::space::SpaceElement::geometry(window).loc
                        - region.loc;
                    let (scale, offset) = window_fit_physical(window, region.size, output_scale);
                    Some(FitWindow {
                        window: window.clone(),
                        loc,
                        scale,
                        offset,
                    })
                })
                .collect(),
            None => Vec::new(),
        };

        let mut window_elements: Vec<WindowElements> = Vec::new();
        for fit in windows {
            let origin: Point<i32, Physical> = fit.loc.to_physical_precise_round(output_scale);
            let elements = fit
                .window
                .render_elements::<WaylandSurfaceRenderElement<GlesRenderer>>(
                    &mut self.renderer,
                    origin,
                    Scale::from(output_scale),
                    // alpha
                    1.0,
                );
            window_elements.extend(elements.into_iter().map(|e| {
                RelocateRenderElement::from_element(
                    RescaleRenderElement::from_element(e, origin, fit.scale),
                    fit.offset,
                    Relocate::Relative,
                )
            }));
        }

        frame_elements.extend(
            window_elements
                .into_iter()
                .map(|element| match &composition {
                    None => FrameElement::Space(element),
                    Some(composition) => composition.convert(
                        element,
                        FrameElement::Space,
                        FrameElement::ConvertedSpace,
                    ),
                }),
        );

        // The framebuffer is always ENCODE-sized; the scene is composited at the RENDER size and
        // upscaled into it (aspect-preserving, centred). Both are the same size unless a render
        // size was requested, in which case `scene_transform` is an exact identity.
        let vi = self.video_info.as_ref().expect("video_info not set");
        let encode_size: Size<i32, Physical> = (vi.width() as i32, vi.height() as i32).into();
        let (scale, offset) = scene_transform(encode_size, effective_render_size(self));
        let mut elements: Vec<OutputElement> = Vec::with_capacity(frame_elements.len());
        // HDR render-path spike: synthetic >1.0 brightness bars across the top of the output
        // as the topmost elements (so client surfaces never occlude them). Gated behind
        // WOLF_HDR_SPIKE -- unset = no bars, no behavior change. They are authored in ENCODE
        // space, so they are the one thing NOT scaled.
        if crate::utils::allocator::hdr_spike() {
            elements.extend(
                hdr_spike_bars(encode_size.w, encode_size.h)
                    .into_iter()
                    .map(OutputElement::Overlay),
            );
        }
        elements.extend(frame_elements.into_iter().map(|e| {
            OutputElement::Scene(RelocateRenderElement::from_element(
                RescaleRenderElement::from_element(e, Point::from((0, 0)), scale),
                offset,
                Relocate::Relative,
            ))
        }));

        // The damage tracker is Static at the encode size (see `apply_output_mode`), which is
        // what sizes the GL viewport/projection here -- the letterbox bars are this clear
        // colour showing through where the scaled scene does not reach.
        let render_output_result = self.dtr.as_mut().unwrap().render_output(
            &mut self.renderer,
            &mut target,
            0,
            &elements,
            [0.0, 0.0, 0.0, 1.0],
        )?;

        // The NV12/VulkanImage paths import this GLES render target as external Vulkan
        // memory and sample it during `to_gs_buffer`. Waiting in the caller after
        // `create_frame` returns is too late: the converter has already submitted its
        // Vulkan read by then, so the GLES write and the Vulkan read are only ordered if
        // the driver happens to serialize the two APIs -- which nothing guarantees.
        // Complete the GLES render here, before any hardware conversion or hand-off
        // consumes the target.
        render_output_result
            .sync
            .wait()
            .expect("Error during render_result.sync");
        // HDR render-path spike ISOLATION probe: read the fp16 RGBA dmabuf straight off the GLES
        // framebuffer (after render_output, before the Vulkan converter in to_gs_buffer) and log
        // the bar-center float values -- proving whether GLES kept >1.0 or clamped. Gated behind
        // WOLF_HDR_SPIKE, only on an fp16 target, and logged once (cheap, a handful of pixels).
        if crate::utils::allocator::hdr_spike()
            && render_rgba_fourcc == Some(Fourcc::Abgr16161616f)
            && let Some(vi) = self.video_info.as_ref()
        {
            let (w, h) = (vi.width() as i32, vi.height() as i32);
            if !HDR_READBACK_DONE.swap(true, Ordering::Relaxed) {
                hdr_spike_gles_readback(&mut self.renderer, &target, w, h);
            }
        }

        match self.output_buffer.clone().unwrap().to_gs_buffer(
            &mut target,
            &mut self.renderer,
            color,
        ) {
            Ok(buffer) => Ok((buffer, render_output_result)),
            Err(e) => {
                tracing::warn!("Failed to convert buffer to gst buffer: {:?}", e);
                Err(OutputDamageTrackerError::Rendering(GlesError::MappingError))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::scene_transform;
    use smithay::utils::{Physical, Point, Size};

    #[track_caller]
    fn check(enc: (i32, i32), ren: (i32, i32), scale: f64, offset: (i32, i32)) {
        let enc: Size<i32, Physical> = enc.into();
        let ren: Size<i32, Physical> = ren.into();
        let (got_scale, got_offset) = scene_transform(enc, ren);
        assert!(
            (got_scale - scale).abs() < 1e-9,
            "{enc:?} <- {ren:?}: expected scale {scale}, got {got_scale}",
        );
        assert_eq!(
            got_offset,
            Point::<i32, Physical>::from(offset),
            "{enc:?} <- {ren:?}: wrong centring offset",
        );
    }

    #[test]
    fn scene_transform_table() {
        // Render == encode: the exact identity the default (no render size) path relies on.
        check((1920, 1080), (1920, 1080), 1.0, (0, 0));
        // A degenerate render size ("not known yet") is also the identity.
        check((1920, 1080), (0, 0), 1.0, (0, 0));

        // Matching aspect ratio: fills the framebuffer, no bars.
        check((1920, 1080), (960, 540), 2.0, (0, 0));
        check((1920, 1080), (1280, 720), 1.5, (0, 0));

        // Narrower than the encode aspect -> width-limited scale, PILLARbox (bars left/right).
        check((1920, 1080), (1280, 1080), 1.0, (320, 0));
        // Wider than the encode aspect -> height-limited scale, LETTERbox (bars top/bottom).
        check((1920, 1080), (1920, 800), 1.0, (0, 140));

        // Non-integer scale, with a rounded offset: 1920/1000 = 1.92 is width-limited, leaving
        // 1080 - 540*1.92 = 43.2px of vertical slack -> 21.6 rounds to a 22px top bar (and a
        // 21px bottom one). Asymmetry of at most 1px is inherent to centring on an integer grid.
        check((1920, 1080), (1000, 540), 1.92, (0, 22));

        // Odd leftover on the limiting axis: 1920 - 1281 = 639 -> a 320px left bar and a 319px
        // right one. Pinned deliberately; `round()` on `.5` goes away from zero.
        check((1920, 1080), (1281, 1080), 1.0, (320, 0));
    }
}
