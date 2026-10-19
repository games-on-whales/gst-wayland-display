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

/// NV12 `memory:VulkanImage` output on the downstream encoder's shared `GstVulkanDevice`
/// (the Vulkan-encode/interpipe path). Carries the H.264 `profile` so the encode-src image
/// is created with a byte-matching `VkVideoProfileListInfoKHR`.
#[derive(Debug, Clone)]
pub struct VulkanParams {
    pub video_info: VideoInfo,
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
