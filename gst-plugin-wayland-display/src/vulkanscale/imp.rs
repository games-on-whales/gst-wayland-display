//! `vulkanscale`: change the size of a `memory:VulkanImage` NV12/P010 frame on the GPU.
//!
//! The element exists so a Vulkan encode path can carry a **mutable capsfilter** the way
//! the CUDA (`cudaconvertscale`), VA (`vapostproc`) and software (`videoscale`) paths
//! already do, which makes a live resolution change possible on that path.
//! At the launch size in == out and `GstBaseTransform` runs the element in passthrough, so
//! a session that never resizes pays nothing and the producer's buffers reach the encoder
//! untouched.
//!
//! Caps behave like `videoscale`: width/height are free on the other side, everything else
//! (format, framerate, colorimetry, interlace mode) passes through, and fixation prefers
//! the size that is already fixed on the originating pad. Aspect ratio is the caller's
//! problem: a mismatched aspect is a plain stretch, not a letterbox.

use gst::glib;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use std::sync::Mutex;
use waylanddisplaycore::utils::video_info::EncodeTarget;
use waylanddisplaycore::utils::vulkan_nv12::PixFmt;
use waylanddisplaycore::utils::vulkan_scale::{ScaleOutput, VulkanScaler};
use waylanddisplaycore::utils::vulkan_share::{self, VulkanShare};

/// Output ring depth. The floor is the encoder's reference depth plus one in-flight
/// frame; below it a slot comes round while the encoder still holds its buffer and every
/// frame pays the `get_mut` stall.
const RING: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, glib::Enum, Default)]
#[repr(u32)]
#[enum_type(name = "GstVulkanScaleMethod")]
pub enum ScaleMethod {
    /// Bilinear (`VK_FILTER_LINEAR`) — the default.
    #[default]
    #[enum_value(name = "Bilinear", nick = "bilinear")]
    Bilinear,
    /// Nearest neighbour (`VK_FILTER_NEAREST`).
    #[enum_value(name = "Nearest", nick = "nearest")]
    Nearest,
}

#[derive(Default)]
struct Settings {
    method: ScaleMethod,
    linear_encode_src: bool,
}

#[derive(Default)]
struct State {
    in_info: Option<gst_video::VideoInfo>,
    out_info: Option<gst_video::VideoInfo>,
    scaler: Option<VulkanScaler>,
}

pub struct VulkanScale {
    settings: Mutex<Settings>,
    state: Mutex<State>,
    /// Absorbs the `gst.vulkan.device` context so our output images land on the same
    /// `GstVulkanDevice` the encoder uses — the crux of the whole element.
    share: std::sync::Arc<VulkanShare>,
}

impl Default for VulkanScale {
    fn default() -> Self {
        VulkanScale {
            settings: Mutex::new(Settings::default()),
            state: Mutex::new(State::default()),
            share: VulkanShare::new(),
        }
    }
}

static CAT: std::sync::LazyLock<gst::DebugCategory> = std::sync::LazyLock::new(|| {
    gst::DebugCategory::new(
        "vulkanscale",
        gst::DebugColorFlags::empty(),
        Some("Vulkan image scaler"),
    )
});

/// Template caps for both pads: the memory feature and the two 4:2:0 formats the Vulkan
/// encoders accept, with the size free.
fn pad_caps() -> gst::Caps {
    gst::Caps::builder("video/x-raw")
        .features(["memory:VulkanImage"])
        .field("format", gst::List::new(["NV12", "P010_10LE"]))
        .field("width", gst::IntRange::new(1, i32::MAX))
        .field("height", gst::IntRange::new(1, i32::MAX))
        .field(
            "framerate",
            gst::FractionRange::new(gst::Fraction::new(0, 1), gst::Fraction::new(i32::MAX, 1)),
        )
        .build()
}

/// `videoscale` semantics: the other side may be any size, everything else is preserved.
/// Pure, so the negotiation maths is testable with no GPU.
pub(crate) fn free_size(caps: &gst::Caps) -> gst::Caps {
    let mut out = gst::Caps::new_empty();
    {
        let out = out.get_mut().unwrap();
        for (s, f) in caps.iter_with_features() {
            let mut s = s.to_owned();
            s.set("width", gst::IntRange::new(1, i32::MAX));
            s.set("height", gst::IntRange::new(1, i32::MAX));
            // A size change invalidates a fixed display aspect ratio; leave it open so a
            // peer that cares can pick one.
            s.remove_field("pixel-aspect-ratio");
            out.append_structure_full(s, Some(f.to_owned()));
        }
    }
    out
}

/// Fixate `other` towards the size already fixed in `caps` (falling back to whatever the
/// range allows). Pure — the second half of the GPU-free negotiation seam.
pub(crate) fn fixate_prefer(caps: &gst::Caps, other: gst::Caps) -> gst::Caps {
    let mut other = other;
    other.truncate();
    if let (Some(w), Some(h)) = (
        caps.structure(0).and_then(|s| s.get::<i32>("width").ok()),
        caps.structure(0).and_then(|s| s.get::<i32>("height").ok()),
    ) && let Some(o) = other.get_mut()
        && let Some(s) = o.structure_mut(0)
    {
        s.fixate_field_nearest_int("width", w);
        s.fixate_field_nearest_int("height", h);
    }
    other.fixate();
    other
}

#[glib::object_subclass]
impl ObjectSubclass for VulkanScale {
    const NAME: &'static str = "GstVulkanScale";
    type Type = super::VulkanScale;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for VulkanScale {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: std::sync::OnceLock<Vec<glib::ParamSpec>> = std::sync::OnceLock::new();
        PROPS.get_or_init(|| {
            vec![
                glib::ParamSpecEnum::builder_with_default("method", ScaleMethod::Bilinear)
                    .nick("Scaling method")
                    .blurb("Sampling filter used by the compute scaler")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("linear-encode-src")
                    .nick("Linear encode-src")
                    .blurb(
                        "Try a LINEAR output image first, falling back to OPTIMAL when the \
                         driver refuses it; as waylanddisplaysrc's property of the same name.",
                    )
                    .default_value(false)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecInt::builder("current-width")
                    .nick("Current width")
                    .blurb("Negotiated output width (0 before caps); diagnostics only")
                    .minimum(0)
                    .read_only()
                    .build(),
                glib::ParamSpecInt::builder("current-height")
                    .nick("Current height")
                    .blurb("Negotiated output height (0 before caps); diagnostics only")
                    .minimum(0)
                    .read_only()
                    .build(),
            ]
        })
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "method" => {
                self.settings.lock().unwrap().method = value.get().expect("type checked upstream");
            }
            "linear-encode-src" => {
                self.settings.lock().unwrap().linear_encode_src =
                    value.get().expect("type checked upstream");
            }
            other => unimplemented!("set_property {other}"),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "method" => self.settings.lock().unwrap().method.to_value(),
            "linear-encode-src" => self.settings.lock().unwrap().linear_encode_src.to_value(),
            "current-width" => {
                let st = self.state.lock().unwrap();
                (st.out_info.as_ref().map(|i| i.width()).unwrap_or(0) as i32).to_value()
            }
            "current-height" => {
                let st = self.state.lock().unwrap();
                (st.out_info.as_ref().map(|i| i.height()).unwrap_or(0) as i32).to_value()
            }
            other => unimplemented!("property {other}"),
        }
    }
}

impl GstObjectImpl for VulkanScale {}

impl ElementImpl for VulkanScale {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: std::sync::OnceLock<gst::subclass::ElementMetadata> =
            std::sync::OnceLock::new();
        Some(META.get_or_init(|| {
            gst::subclass::ElementMetadata::new(
                "Vulkan Scale",
                "Filter/Converter/Video/Scaler",
                "Scales memory:VulkanImage NV12/P010 frames on the GPU",
                "Victoria Brekenfeld <wayland@drakulix.de>, ABeltramo <https://github.com/ABeltramo>",
            )
        }))
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static TEMPLATES: std::sync::OnceLock<Vec<gst::PadTemplate>> = std::sync::OnceLock::new();
        TEMPLATES.get_or_init(|| {
            let caps = pad_caps();
            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
            ]
        })
    }

    fn set_context(&self, context: &gst::Context) {
        if context.context_type() == "gst.vulkan.device" {
            self.share.handle_set_context(context);
        }
        self.parent_set_context(context);
    }
}

impl BaseTransformImpl for VulkanScale {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = true;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_caps(
        &self,
        _direction: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        let out = free_size(caps);
        Some(match filter {
            Some(f) => f.intersect_with_mode(&out, gst::CapsIntersectMode::First),
            None => out,
        })
    }

    fn fixate_caps(
        &self,
        _direction: gst::PadDirection,
        caps: &gst::Caps,
        othercaps: gst::Caps,
    ) -> gst::Caps {
        fixate_prefer(caps, othercaps)
    }

    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let in_info = gst_video::VideoInfo::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "invalid sink caps {incaps}"))?;
        let out_info = gst_video::VideoInfo::from_caps(outcaps)
            .map_err(|_| gst::loggable_error!(CAT, "invalid src caps {outcaps}"))?;
        if in_info.format() != out_info.format() {
            return Err(gst::loggable_error!(
                CAT,
                "format conversion is not supported ({:?} -> {:?})",
                in_info.format(),
                out_info.format()
            ));
        }
        let passthrough =
            in_info.width() == out_info.width() && in_info.height() == out_info.height();
        {
            let mut st = self.state.lock().unwrap();
            // A live rung step lands here: dropping the old scaler releases its output
            // ring and descriptor sets, and the next buffer builds one at the new size.
            st.scaler = None;
            st.in_info = Some(in_info.clone());
            st.out_info = Some(out_info.clone());
        }
        self.obj().set_passthrough(passthrough);
        gst::info!(
            CAT,
            imp = self,
            "{}x{} -> {}x{} {:?}{}",
            in_info.width(),
            in_info.height(),
            out_info.width(),
            out_info.height(),
            out_info.format(),
            if passthrough { " (passthrough)" } else { "" }
        );
        self.obj().notify("current-width");
        self.obj().notify("current-height");
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let mut st = self.state.lock().unwrap();
        st.scaler = None;
        st.in_info = None;
        st.out_info = None;
        drop(st);
        self.share.clear();
        Ok(())
    }

    /// All the work happens here: we hand downstream a buffer from our own encode-src
    /// ring rather than letting `GstBaseTransform` allocate one, because the encoder can
    /// only consume a single multiplanar `VIDEO_ENCODE_SRC` image and no generic pool on
    /// NVIDIA will produce one.
    fn prepare_output_buffer(
        &self,
        inbuf: gst_base::subclass::base_transform::InputBuffer,
    ) -> Result<gst_base::subclass::base_transform::PrepareOutputBufferSuccess, gst::FlowError>
    {
        if self.obj().is_passthrough() {
            return Ok(gst_base::subclass::base_transform::PrepareOutputBufferSuccess::InputBuffer);
        }
        let inbuf = match inbuf {
            gst_base::subclass::base_transform::InputBuffer::Readable(b) => b.to_owned(),
            gst_base::subclass::base_transform::InputBuffer::Writable(b) => b.to_owned(),
        };
        let out = self.scale(&inbuf).map_err(|e| {
            gst::element_imp_error!(self, gst::ResourceError::Failed, ["{e}"]);
            gst::FlowError::Error
        })?;
        Ok(gst_base::subclass::base_transform::PrepareOutputBufferSuccess::Buffer(out))
    }

    fn transform(
        &self,
        _inbuf: &gst::Buffer,
        _outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        // prepare_output_buffer already produced the scaled frame.
        Ok(gst::FlowSuccess::Ok)
    }
}

impl VulkanScale {
    /// Scale one buffer, building the Vulkan engine on first use (the shared device is not
    /// available until contexts have propagated, which is after `set_caps`).
    fn scale(&self, inbuf: &gst::Buffer) -> Result<gst::Buffer, String> {
        let mut st = self.state.lock().unwrap();
        if st.scaler.is_none() {
            let out_info = st.out_info.clone().ok_or("no negotiated output caps")?;
            let device = self
                .ensure_device()
                .ok_or("no shared GstVulkanDevice: vulkanscale must sit on a Vulkan path")?;
            let raw = vulkan_share::raw_handles(&device)
                .ok_or("shared GstVulkanDevice exposes no graphics+compute queue")?;
            let (nearest, linear_encode_src) = {
                let settings = self.settings.lock().unwrap();
                (
                    settings.method == ScaleMethod::Nearest,
                    settings.linear_encode_src,
                )
            };
            // The raw caps carry no codec: suit every profile that can encode the format, as
            // waylanddisplaysrc does for an encoder it cannot see.
            let output = ScaleOutput {
                targets: EncodeTarget::all_for(out_info.format()),
                linear_encode_src,
            };
            let scaler = VulkanScaler::new(
                device,
                raw,
                out_info.width(),
                out_info.height(),
                PixFmt::from_gst(out_info.format()),
                nearest,
                &output,
                RING,
            )
            .map_err(|e| format!("scaler init failed: {e}"))?;
            st.scaler = Some(scaler);
        }
        st.scaler
            .as_mut()
            .unwrap()
            .scale(inbuf)
            .map_err(|e| format!("scale failed: {e}"))
    }

    /// The shared `GstVulkanDevice`: from a context already delivered to us, else asked
    /// for downstream (the encoder) and then upstream (the producer).
    fn ensure_device(&self) -> Option<gstreamer_vulkan::VulkanDevice> {
        if let Some(d) = self.share.shared_device() {
            return Some(d);
        }
        let obj = self.obj();
        for pad in [obj.src_pad(), obj.sink_pad()] {
            let mut q = gst::query::Context::new("gst.vulkan.device");
            if pad.peer_query(&mut q)
                && let Some(ctx) = q.context_owned()
                && self.share.handle_set_context(&ctx)
            {
                return self.share.shared_device();
            }
        }
        if let Some(ctx) = obj.context("gst.vulkan.device")
            && self.share.handle_set_context(&ctx)
        {
            return self.share.shared_device();
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vk_caps(w: i32, h: i32) -> gst::Caps {
        gst::Caps::builder("video/x-raw")
            .features(["memory:VulkanImage"])
            .field("format", "NV12")
            .field("width", w)
            .field("height", h)
            .field("framerate", gst::Fraction::new(60, 1))
            .build()
    }

    #[test]
    fn free_size_opens_width_and_height_and_keeps_the_rest() {
        gst::init().unwrap();
        let out = free_size(&vk_caps(1920, 1080));
        let s = out.structure(0).unwrap();
        assert!(s.get::<gst::IntRange<i32>>("width").is_ok(), "{out}");
        assert!(s.get::<gst::IntRange<i32>>("height").is_ok(), "{out}");
        assert_eq!(s.get::<String>("format").unwrap(), "NV12");
        assert_eq!(
            s.get::<gst::Fraction>("framerate").unwrap(),
            gst::Fraction::new(60, 1)
        );
        assert!(
            out.features(0).unwrap().contains("memory:VulkanImage"),
            "the memory feature must survive: {out}"
        );
    }

    #[test]
    fn fixate_prefers_the_fixed_side() {
        gst::init().unwrap();
        let sink = vk_caps(1920, 1080);
        let other = free_size(&sink);
        let fixed = fixate_prefer(&sink, other);
        let s = fixed.structure(0).unwrap();
        assert_eq!(s.get::<i32>("width").unwrap(), 1920);
        assert_eq!(s.get::<i32>("height").unwrap(), 1080);
    }

    #[test]
    fn fixate_honours_a_downstream_size_the_range_pins() {
        gst::init().unwrap();
        let sink = vk_caps(1920, 1080);
        // What a downstream capsfilter(1280x720) leaves us to fixate.
        let other = free_size(&sink).intersect(&vk_caps(1280, 720));
        let fixed = fixate_prefer(&sink, other);
        let s = fixed.structure(0).unwrap();
        assert_eq!(s.get::<i32>("width").unwrap(), 1280);
        assert_eq!(s.get::<i32>("height").unwrap(), 720);
    }
}
