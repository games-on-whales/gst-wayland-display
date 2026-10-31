#[cfg(feature = "cuda")]
use crate::utils::allocator::cuda;
use gst_video::{VideoInfo, VideoInfoDmaDrm};
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CUDAParams {
    pub video_info: VideoInfoDmaDrm,
    pub cuda_context: Arc<Mutex<cuda::CUDAContext>>,
}

/// The Vulkan Video codec consuming `memory:VulkanImage` output. The encode-src image must
/// be created with this codec's profile in its `VkVideoProfileListInfoKHR`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanCodec {
    H264,
    H265,
    Av1,
}

impl VulkanCodec {
    /// The codec of a gst Vulkan encoder element factory (`vulkanh264enc`, ...). On a
    /// multi-GPU host only the first Vulkan device gets the plain name; the others register
    /// as `vulkanh264device1enc` and so on.
    pub fn from_encoder_factory(name: &str) -> Option<Self> {
        let codec = name.strip_prefix("vulkan")?.strip_suffix("enc")?;
        let codec = match codec.split_once("device") {
            Some((codec, index))
                if !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) =>
            {
                codec
            }
            Some(_) => return None,
            None => codec,
        };
        Some(match codec {
            "h264" => VulkanCodec::H264,
            "h265" => VulkanCodec::H265,
            "av1" => VulkanCodec::Av1,
            _ => return None,
        })
    }

    /// Parse the `vulkan-codec` property value.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "h264" => VulkanCodec::H264,
            "h265" => VulkanCodec::H265,
            "av1" => VulkanCodec::Av1,
            _ => return None,
        })
    }

    /// Whether `profile` of this codec can encode `format` (NV12 or P010).
    pub fn accepts(self, format: gst_video::VideoFormat, profile: &str) -> bool {
        use gst_video::VideoFormat::{Nv12, P01010le};
        match (self, format) {
            (VulkanCodec::H264, Nv12) => matches!(
                profile,
                "high"
                    | "constrained-high"
                    | "progressive-high"
                    | "main"
                    | "baseline"
                    | "constrained-baseline"
            ),
            (VulkanCodec::H265, Nv12) => profile == "main",
            (VulkanCodec::H265, P01010le) => profile == "main-10",
            // vulkanav1enc (patches/vulkanav1enc.patch) only takes 8-bit NV12 input.
            (VulkanCodec::Av1, Nv12) => profile == "main",
            _ => false,
        }
    }
}

/// A video profile an encode-src image is created for. Vulkan only lets an encode session
/// read an image whose `VkVideoProfileListInfoKHR` lists the session's own profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeTarget {
    pub codec: VulkanCodec,
    pub profile: String,
}

impl EncodeTarget {
    pub fn new(codec: VulkanCodec, profile: impl Into<String>) -> Self {
        EncodeTarget {
            codec,
            profile: profile.into(),
        }
    }

    /// Every codec profile that can encode `format`: the targets for an encoder this element
    /// cannot see. Behind interpipe the consumers are unknown and may differ per session (one
    /// producer can feed an H.264 and an H.265 encoder), so the image must suit all of them.
    /// One entry per distinct Vulkan profile (the H.264 aliases share a profile_idc).
    pub fn all_for(format: gst_video::VideoFormat) -> Vec<EncodeTarget> {
        [
            (VulkanCodec::H264, "high"),
            (VulkanCodec::H264, "main"),
            (VulkanCodec::H264, "baseline"),
            (VulkanCodec::H265, "main"),
            (VulkanCodec::H265, "main-10"),
            (VulkanCodec::Av1, "main"),
        ]
        .into_iter()
        .filter(|(codec, profile)| codec.accepts(format, profile))
        .map(|(codec, profile)| EncodeTarget::new(codec, profile))
        .collect()
    }

    /// The targets for the codec and profile configured on, or found downstream of, the
    /// element: either may be unknown, and an unknown one covers every candidate for `format`.
    /// Empty when nothing can encode `format` under those constraints.
    pub fn resolve(
        format: gst_video::VideoFormat,
        codec: Option<VulkanCodec>,
        profile: Option<&str>,
    ) -> Vec<EncodeTarget> {
        match (codec, profile) {
            (Some(codec), Some(profile)) => {
                if codec.accepts(format, profile) {
                    vec![EncodeTarget::new(codec, profile)]
                } else {
                    Vec::new()
                }
            }
            (codec, Some(profile)) => [VulkanCodec::H264, VulkanCodec::H265, VulkanCodec::Av1]
                .into_iter()
                .filter(|c| codec.is_none_or(|codec| codec == *c) && c.accepts(format, profile))
                .map(|c| EncodeTarget::new(c, profile))
                .collect(),
            (codec, None) => EncodeTarget::all_for(format)
                .into_iter()
                .filter(|t| codec.is_none_or(|codec| codec == t.codec))
                .collect(),
        }
    }
}

/// NV12/P010 `memory:VulkanImage` output on the downstream encoder's shared `GstVulkanDevice`
/// (the Vulkan-encode/interpipe path). Carries the encode profiles the encode-src images are
/// created for (see [`EncodeTarget`]).
#[derive(Debug, Clone)]
pub struct VulkanParams {
    pub video_info: VideoInfo,
    pub targets: Vec<EncodeTarget>,
}

#[derive(Debug, Clone)]
pub enum GstVideoInfo {
    RAW(VideoInfo),
    DMA(VideoInfoDmaDrm),
    VULKAN(VulkanParams),
    #[cfg(feature = "cuda")]
    CUDA(CUDAParams),
}

impl From<VideoInfo> for GstVideoInfo {
    fn from(info: VideoInfo) -> Self {
        GstVideoInfo::RAW(info)
    }
}

impl From<VideoInfoDmaDrm> for GstVideoInfo {
    fn from(info: VideoInfoDmaDrm) -> Self {
        GstVideoInfo::DMA(info)
    }
}

impl From<GstVideoInfo> for VideoInfo {
    fn from(info: GstVideoInfo) -> Self {
        match info {
            GstVideoInfo::RAW(info) => info,
            GstVideoInfo::VULKAN(params) => params.video_info,
            GstVideoInfo::DMA(info) => match info.to_video_info() {
                Ok(info) => info,
                Err(_) => VideoInfo::builder(info.format(), info.width(), info.height())
                    .fps(info.fps())
                    .build()
                    .expect("Failed to build VideoInfo from VideoInfoDmaDrm"),
            },
            #[cfg(feature = "cuda")]
            GstVideoInfo::CUDA(params) => match params.video_info.to_video_info() {
                Ok(info) => info,
                Err(_) => VideoInfo::builder(
                    params.video_info.format(),
                    params.video_info.width(),
                    params.video_info.height(),
                )
                .fps(params.video_info.fps())
                .build()
                .expect("Failed to build VideoInfo from VideoInfoDmaDrm"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tests::test_init;

    #[test]
    fn encoder_factories_on_every_vulkan_device_map_to_their_codec() {
        for (name, codec) in [
            ("vulkanh264enc", Some(VulkanCodec::H264)),
            ("vulkanh265enc", Some(VulkanCodec::H265)),
            ("vulkanav1enc", Some(VulkanCodec::Av1)),
            ("vulkanh264device1enc", Some(VulkanCodec::H264)),
            ("vulkanh265device1enc", Some(VulkanCodec::H265)),
            ("vulkanav1device12enc", Some(VulkanCodec::Av1)),
            ("vulkanh265dec", None),
            ("vulkanh265device1dec", None),
            ("vulkanh265deviceenc", None),
            ("vulkanh265devicexenc", None),
            ("vah265enc", None),
        ] {
            assert_eq!(VulkanCodec::from_encoder_factory(name), codec, "{name}");
        }
    }

    #[test]
    fn raw_video_info_round_trips() {
        test_init();

        let info = VideoInfo::builder(gst_video::VideoFormat::Rgba, 1920, 1080)
            .fps(gst::Fraction::new(60, 1))
            .build()
            .unwrap();

        // VideoInfo -> GstVideoInfo::RAW -> VideoInfo is lossless.
        let wrapped: GstVideoInfo = info.clone().into();
        assert!(matches!(wrapped, GstVideoInfo::RAW(_)));

        let back: VideoInfo = wrapped.into();
        assert_eq!(back.format(), info.format());
        assert_eq!(back.width(), info.width());
        assert_eq!(back.height(), info.height());
        assert_eq!(back.fps(), info.fps());
    }

    #[test]
    fn dma_video_info_converts_to_raw_preserving_dimensions() {
        test_init();

        let caps = gst_video::VideoCapsBuilder::new()
            .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .format(gst_video::VideoFormat::DmaDrm)
            .field("drm-format", "RGBA")
            .width(640)
            .height(480)
            .pixel_aspect_ratio(1.into())
            .framerate(gst::Fraction::new(30, 1))
            .build();
        assert!(caps.is_fixed());
        let dma = VideoInfoDmaDrm::from_caps(&caps).expect("Failed to create video info");

        let wrapped: GstVideoInfo = dma.into();
        assert!(matches!(wrapped, GstVideoInfo::DMA(_)));

        // The DMA arm resolves to a plain VideoInfo carrying the same geometry.
        let back: VideoInfo = wrapped.into();
        assert_eq!(back.width(), 640);
        assert_eq!(back.height(), 480);
    }
}

#[cfg(test)]
mod encode_target_tests {
    use super::*;
    use gst_video::VideoFormat::{Nv12, P01010le};

    #[test]
    fn an_unseen_encoder_gets_every_profile_for_the_format() {
        let nv12 = EncodeTarget::resolve(Nv12, None, None);
        assert!(nv12.contains(&EncodeTarget::new(VulkanCodec::H264, "main")));
        assert!(nv12.contains(&EncodeTarget::new(VulkanCodec::H264, "high")));
        assert!(nv12.contains(&EncodeTarget::new(VulkanCodec::H265, "main")));
        assert!(!nv12.iter().any(|t| t.profile == "main-10"));
        assert_eq!(
            EncodeTarget::resolve(P01010le, None, None),
            vec![EncodeTarget::new(VulkanCodec::H265, "main-10")]
        );
    }

    #[test]
    fn a_known_codec_or_profile_narrows_the_targets() {
        assert_eq!(
            EncodeTarget::resolve(Nv12, Some(VulkanCodec::H264), Some("main")),
            vec![EncodeTarget::new(VulkanCodec::H264, "main")]
        );
        let h265 = EncodeTarget::resolve(Nv12, Some(VulkanCodec::H265), None);
        assert_eq!(h265, vec![EncodeTarget::new(VulkanCodec::H265, "main")]);
        let main = EncodeTarget::resolve(Nv12, None, Some("main"));
        assert!(main.iter().all(|t| t.profile == "main"));
        assert_eq!(main.len(), 3);
        assert!(EncodeTarget::resolve(Nv12, Some(VulkanCodec::H265), Some("main-10")).is_empty());
    }
}
