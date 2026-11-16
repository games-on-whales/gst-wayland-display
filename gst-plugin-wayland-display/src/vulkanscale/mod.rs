//! `vulkanscale` — a GPU scaler for `memory:VulkanImage` NV12/P010 frames.
//!
//! See [`imp`] for the element itself and
//! [`waylanddisplaycore::utils::vulkan_scale`] for the Vulkan engine it drives.

use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct VulkanScale(ObjectSubclass<imp::VulkanScale>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "vulkanscale",
        gst::Rank::NONE,
        VulkanScale::static_type(),
    )
}
