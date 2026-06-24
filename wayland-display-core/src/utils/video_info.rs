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
