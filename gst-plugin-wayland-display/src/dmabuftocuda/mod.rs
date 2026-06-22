use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct DmabufToCuda(ObjectSubclass<imp::DmabufToCuda>) @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "dmabuftocuda",
        gst::Rank::NONE,
        DmabufToCuda::static_type(),
    )
}
