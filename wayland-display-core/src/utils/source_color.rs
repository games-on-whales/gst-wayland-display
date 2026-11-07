//! Frame-local color interpretation for raw RGB compositor buffers.
//!
//! This is the deployed contract name. Keep it in sync with the CUDA patch in
//! `patches/series`. It does not describe already-converted NV12/P010 buffers.
use gst::meta::CustomMeta;
use std::sync::Once;

pub const SOURCE_COLOR_META: &str = "UnraidBeamSourceColorMeta";
static REGISTER: Once = Once::new();

/// `true`: samples already encode BT.2020/PQ. `false`: sRGB/BT.709 samples.
/// The consumer only uses this bit for RGB input and negotiated PQ output.
/// General format/colorimetry and mastering metadata remain caps contracts.
pub fn add_source_color_meta(
    buffer: &mut gst::Buffer,
    pq_passthrough: bool,
) -> Result<(), gst::glib::BoolError> {
    REGISTER.call_once(|| {
        if !CustomMeta::is_registered(SOURCE_COLOR_META) {
            CustomMeta::register_simple(SOURCE_COLOR_META);
        }
    });
    let mut meta = CustomMeta::add(buffer.make_mut(), SOURCE_COLOR_META)?;
    meta.mut_structure().set("pq-passthrough", pq_passthrough);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(buffer: &gst::BufferRef) -> bool {
        CustomMeta::from_buffer(buffer, SOURCE_COLOR_META)
            .unwrap()
            .structure()
            .get::<bool>("pq-passthrough")
            .unwrap()
    }

    #[test]
    fn copied_frames_retain_their_own_color_across_sdr_pq_sdr() {
        gst::init().unwrap();
        let mut frames = Vec::new();
        for pq in [false, true, false, true, false] {
            let mut frame = gst::Buffer::from_mut_slice(vec![17u8; 16]);
            add_source_color_meta(&mut frame, pq).unwrap();
            frames.push((frame.copy(), frame.copy_deep().unwrap(), pq));
        }
        for (shallow, deep, expected) in frames {
            assert_eq!(value(&shallow), expected);
            assert_eq!(value(&deep), expected);
            assert_eq!(deep.map_readable().unwrap().as_slice(), &[17u8; 16]);
        }
    }

    #[test]
    fn missing_and_malformed_metadata_are_not_a_pq_declaration() {
        gst::init().unwrap();
        let mut buffer = gst::Buffer::new();
        assert!(CustomMeta::from_buffer(&buffer, SOURCE_COLOR_META).is_err());
        add_source_color_meta(&mut buffer, true).unwrap();
        let mut meta = CustomMeta::from_mut_buffer(buffer.make_mut(), SOURCE_COLOR_META).unwrap();
        meta.mut_structure().set("pq-passthrough", "true");
        assert!(meta.structure().get::<bool>("pq-passthrough").is_err());
    }
}
