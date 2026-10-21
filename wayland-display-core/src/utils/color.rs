//! Colour description of a composited frame, shared by the compositor (which derives it
//! from the clients' colour-management state) and the converter (which picks its shader
//! from it).

/// How a surface's pixel values are encoded, as declared by the client. Selects the
/// converter shader for a frame whose active surface carries it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ContentTransfer {
    /// sRGB / gamma 2.2 code values (also the answer for undeclared surfaces).
    #[default]
    Sdr,
    /// SMPTE ST 2084 (PQ) code values in BT.2020.
    Pq,
}

/// What the converter must produce for one frame under dynamic HDR.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameColor {
    /// The committed output state the element tags this frame with: BT.2100 PQ when
    /// `true`, BT.709 when `false`.
    pub output_hdr: bool,
    /// The transfer function of the content being sampled.
    pub content: ContentTransfer,
}
