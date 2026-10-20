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
    /// The codec of a gst Vulkan encoder element factory (`vulkanh264enc`, ...).
    pub fn from_encoder_factory(name: &str) -> Option<Self> {
        Some(match name {
            "vulkanh264enc" => VulkanCodec::H264,
            "vulkanh265enc" => VulkanCodec::H265,
            "vulkanav1enc" => VulkanCodec::Av1,
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

    /// The codec assumed when neither the property nor a linked encoder names one:
    /// H.264 for 8-bit NV12, H.265 Main-10 for P010.
    pub fn default_for(format: gst_video::VideoFormat) -> Self {
        if format == gst_video::VideoFormat::P01010le {
            VulkanCodec::H265
        } else {
            VulkanCodec::H264
        }
    }

    /// The profile used when none is configured or found downstream.
    pub fn default_profile(self, format: gst_video::VideoFormat) -> &'static str {
        match (self, format == gst_video::VideoFormat::P01010le) {
            (VulkanCodec::H264, _) => "high",
            (VulkanCodec::H265, false) => "main",
            (VulkanCodec::H265, true) => "main-10",
            (VulkanCodec::Av1, _) => "main",
        }
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
            (VulkanCodec::Av1, Nv12 | P01010le) => profile == "main",
            _ => false,
        }
    }
}

/// NV12/P010 `memory:VulkanImage` output on the downstream encoder's shared `GstVulkanDevice`
/// (the Vulkan-encode/interpipe path). Carries the encoder's `codec` and `profile` so the
/// encode-src image is created with a matching `VkVideoProfileListInfoKHR`.
#[derive(Debug, Clone)]
pub struct VulkanParams {
    pub video_info: VideoInfo,
    pub codec: VulkanCodec,
    pub profile: String,
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
