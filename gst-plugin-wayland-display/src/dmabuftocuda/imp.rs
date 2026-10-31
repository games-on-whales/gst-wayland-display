use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::base_transform::{BaseTransformMode, GenerateOutputSuccess};
use gst_base::subclass::prelude::*;
use gst_video::{VideoFormat, VideoInfoDmaDrm};
use once_cell::sync::Lazy;

use waylanddisplaycore::utils::allocator::cuda::{
    self, CAPS_FEATURE_MEMORY_CUDA_MEMORY, CUDABufferPool, CUDAContext, CudaUploader,
    GstCudaContext,
};

#[derive(Default)]
struct Settings {
    render_node: Option<String>,
}

/// Per-caps negotiated state.
struct Negotiated {
    in_info: VideoInfoDmaDrm,
    pool: CUDABufferPool,
}

// Field order matters for Drop: the CUDA buffer pool (in `negotiated`) and the
// uploader must drop before `cuda_context`, since they reference the context.
#[derive(Default)]
pub struct DmabufToCuda {
    settings: Mutex<Settings>,
    negotiated: Mutex<Option<Negotiated>>,
    uploader: Mutex<Option<CudaUploader>>,
    cuda_context: Mutex<Option<Arc<Mutex<CUDAContext>>>>,
    cuda_raw_ptr: AtomicPtr<GstCudaContext>,
    // Set while acquiring the context, so the re-entrant set_context that
    // gst_cuda_ensure_element_context triggers doesn't create a second wrapper.
    acquiring: AtomicBool,
}

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "dmabuftocuda",
        gst::DebugColorFlags::empty(),
        Some("Wayland display NV12 DMABuf to CUDA uploader"),
    )
});

impl DmabufToCuda {
    /// Acquire (or reuse) a CUDA context shared with the rest of the pipeline.
    fn ensure_cuda_context(&self) -> Option<Arc<Mutex<CUDAContext>>> {
        if let Some(ctx) = self.cuda_context.lock().unwrap().as_ref() {
            return Some(ctx.clone());
        }
        let elem = self.obj().upcast_ref::<gst::Element>().to_owned();
        let raw = self.cuda_raw_ptr.as_ptr();
        // gst_cuda_ensure_element_context re-enters set_context; block it from
        // creating a competing wrapper for the same context.
        self.acquiring.store(true, Ordering::SeqCst);
        let result = CUDAContext::new_from_gstreamer(&elem, -1, raw);
        self.acquiring.store(false, Ordering::SeqCst);
        match result {
            Ok(ctx) => {
                let mut guard = self.cuda_context.lock().unwrap();
                // A re-entrant set_context may already have stored one.
                if let Some(existing) = guard.as_ref() {
                    return Some(existing.clone());
                }
                let arc = Arc::new(Mutex::new(ctx));
                *guard = Some(arc.clone());
                Some(arc)
            }
            Err(e) => {
                gst::error!(CAT, imp = self, "Failed to acquire CUDA context: {e}");
                None
            }
        }
    }

    fn convert(&self, inbuf: &gst::Buffer) -> Result<gst::Buffer, gst::FlowError> {
        let neg_guard = self.negotiated.lock().unwrap();
        let neg = neg_guard.as_ref().ok_or(gst::FlowError::NotNegotiated)?;
        let uploader_guard = self.uploader.lock().unwrap();
        let uploader = uploader_guard
            .as_ref()
            .ok_or(gst::FlowError::NotNegotiated)?;
        let cuda_guard = self.cuda_context.lock().unwrap();
        let cuda_arc = cuda_guard.as_ref().ok_or(gst::FlowError::NotNegotiated)?;
        let ctx = cuda_arc.lock().unwrap();

        let mut output = uploader
            .upload(inbuf, &neg.in_info, &ctx, Some(&neg.pool))
            .map_err(|e| {
                gst::error!(CAT, imp = self, "dmabuf -> CUDA failed: {e}");
                gst::FlowError::Error
            })?;
        inbuf
            .copy_into(
                output.get_mut().ok_or(gst::FlowError::Error)?,
                gst::BufferCopyFlags::TIMESTAMPS | gst::BufferCopyFlags::FLAGS,
                ..,
            )
            .map_err(|_| gst::FlowError::Error)?;
        Ok(output)
    }
}

/// Preserve colour and HDR metadata while changing only the memory representation.
fn transformed_caps(caps: &gst::Caps, to_cuda: bool, target_minor: Option<u32>) -> gst::Caps {
    let mut result = gst::Caps::new_empty();
    for structure in caps.iter() {
        let requested = structure
            .get::<&str>(if to_cuda { "drm-format" } else { "format" })
            .ok();
        for (drm, format) in [("NV12", "NV12"), ("P010", "P010_10LE")] {
            if requested.is_some_and(|value| {
                if to_cuda {
                    !value.starts_with(drm)
                } else {
                    value != format
                }
            }) {
                continue;
            }
            if to_cuda {
                let mut s = structure.to_owned();
                s.remove_field("drm-format");
                s.set("format", format);
                result.get_mut().unwrap().append_structure_full(
                    s,
                    Some(gst::CapsFeatures::new([CAPS_FEATURE_MEMORY_CUDA_MEMORY])),
                );
            } else {
                let modifiers = if drm == "P010" {
                    waylanddisplaycore::utils::vulkan_nv12::supported_p010_modifiers(target_minor)
                } else {
                    waylanddisplaycore::utils::vulkan_nv12::supported_nv12_modifiers(target_minor)
                };
                for modifier in modifiers
                    .iter()
                    .copied()
                    .filter(|m| *m != 0x00ff_ffff_ffff_ffff)
                {
                    let mut s = structure.to_owned();
                    s.set("format", "DMA_DRM");
                    s.set(
                        "drm-format",
                        if modifier == 0 {
                            drm.to_owned()
                        } else {
                            format!("{drm}:0x{modifier:016x}")
                        },
                    );
                    result.get_mut().unwrap().append_structure_full(
                        s,
                        Some(gst::CapsFeatures::new([
                            gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF,
                        ])),
                    );
                }
            }
        }
    }
    result
}

#[glib::object_subclass]
impl ObjectSubclass for DmabufToCuda {
    const NAME: &'static str = "GstDmabufToCuda";
    type Type = super::DmabufToCuda;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for DmabufToCuda {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecString::builder("render-node")
                    .nick("Render node")
                    .blurb("DRM render node of the GPU that produced the dmabuf (e.g. /dev/dri/renderD128)")
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "render-node" => {
                self.settings.lock().unwrap().render_node = value.get().expect("type checked");
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "render-node" => self.settings.lock().unwrap().render_node.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for DmabufToCuda {}

impl ElementImpl for DmabufToCuda {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Wayland display DMABuf to CUDA uploader",
                "Filter/Video/Converter",
                "Imports an NV12/P010 DMABuf into CUDA memory via EGLImage (convert to CUDA late)",
                "Games on Whales",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let sink_caps = gst_video::VideoCapsBuilder::new()
                .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
                .format(VideoFormat::DmaDrm)
                .build();
            let src_caps = gst_video::VideoCapsBuilder::new()
                .features([CAPS_FEATURE_MEMORY_CUDA_MEMORY])
                .format_list([VideoFormat::Nv12, VideoFormat::P01010le])
                .build();
            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &sink_caps,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &src_caps,
                )
                .unwrap(),
            ]
        });
        PAD_TEMPLATES.as_ref()
    }

    fn set_context(&self, context: &gst::Context) {
        // Skip while ensure_cuda_context is mid-acquire: gst_cuda_ensure_element_context
        // re-enters here, and creating a second wrapper for the same context would
        // over-unref it at teardown. Also skip if we already have one.
        if !self.acquiring.load(Ordering::SeqCst) {
            let mut guard = self.cuda_context.lock().unwrap();
            if guard.is_none() {
                let elem = self.obj().upcast_ref::<gst::Element>().to_owned();
                let raw = self.cuda_raw_ptr.as_ptr();
                if let Ok(ctx) = CUDAContext::new_from_set_context(&elem, context, -1, raw) {
                    *guard = Some(Arc::new(Mutex::new(ctx)));
                }
            }
        }
        self.parent_set_context(context)
    }
}

impl BaseTransformImpl for DmabufToCuda {
    const MODE: BaseTransformMode = BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_caps(
        &self,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        // Sink is DMABuf NV12, Src is CUDAMemory NV12.
        let to_cuda = direction == gst::PadDirection::Sink;
        // Query NV12 export modifiers on the GPU that produced the dmabuf (this element's
        // render node), so a multi-GPU host advertises the right set.
        let target_minor = self
            .settings
            .lock()
            .unwrap()
            .render_node
            .as_deref()
            .and_then(waylanddisplaycore::utils::vulkan_nv12::render_node_minor);
        let mut result = transformed_caps(caps, to_cuda, target_minor);
        if let Some(filter) = filter {
            result = filter.intersect_with_mode(&result, gst::CapsIntersectMode::First);
        }
        Some(result)
    }

    fn query(&self, direction: gst::PadDirection, query: &mut gst::QueryRef) -> bool {
        if query.type_() == gst::QueryType::Context {
            if let Some(ctx) = self.cuda_context.lock().unwrap().as_ref() {
                let ctx = ctx.lock().unwrap();
                let elem = self.obj();
                return cuda::gst_cuda_handle_context_query_wrapped(
                    elem.upcast_ref::<gst::Element>(),
                    query,
                    &ctx,
                );
            }
        }
        BaseTransformImplExt::parent_query(self, direction, query)
    }

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let render_node = self.settings.lock().unwrap().render_node.clone();
        let uploader = CudaUploader::new(render_node.as_deref());
        *self.uploader.lock().unwrap() = Some(uploader);
        self.ensure_cuda_context();
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        // Release the pool and uploader, but keep the CUDA context: downstream CUDA
        // buffers may still reference it until they are freed.
        *self.negotiated.lock().unwrap() = None;
        *self.uploader.lock().unwrap() = None;
        Ok(())
    }

    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        *self.uploader.lock().unwrap() = Some(CudaUploader::new(
            self.settings.lock().unwrap().render_node.as_deref(),
        ));
        let in_info = VideoInfoDmaDrm::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "invalid input caps {incaps}"))?;

        let cuda_arc = self
            .ensure_cuda_context()
            .ok_or_else(|| gst::loggable_error!(CAT, "no CUDA context"))?;
        let cuda_ctx = cuda_arc.lock().unwrap();

        // Configure the output pool with the negotiated (fixed) src caps.
        let pool = CUDABufferPool::new(&cuda_ctx)
            .map_err(|e| gst::loggable_error!(CAT, "cuda pool: {e}"))?;
        pool.configure(
            outcaps,
            cuda_ctx
                .stream()
                .ok_or_else(|| gst::loggable_error!(CAT, "no cuda stream"))?,
            in_info.size() as u32,
            0,
            0,
        )
        .map_err(|e| gst::loggable_error!(CAT, "configure pool: {e}"))?;
        pool.activate()
            .map_err(|e| gst::loggable_error!(CAT, "activate pool: {e}"))?;
        drop(cuda_ctx);

        *self.negotiated.lock().unwrap() = Some(Negotiated { in_info, pool });
        Ok(())
    }

    fn generate_output(&self) -> Result<GenerateOutputSuccess, gst::FlowError> {
        match self.take_queued_buffer() {
            Some(buffer) => Ok(GenerateOutputSuccess::Buffer(self.convert(&buffer)?)),
            None => Ok(GenerateOutputSuccess::NoOutput),
        }
    }
}

impl Drop for DmabufToCuda {
    fn drop(&mut self) {
        unsafe {
            CUDAContext::release_slot(self.cuda_raw_ptr.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn p010_cuda_caps_preserve_hdr_metadata() {
        gst::init().unwrap();
        let input: gst::Caps = "video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=P010,width=3840,height=2160,framerate=60/1,colorimetry=bt2100-pq,mastering-display-info=test-metadata"
            .parse().unwrap();
        let output = super::transformed_caps(&input, true, None);
        assert_eq!(output.size(), 1);
        let s = output.structure(0).unwrap();
        assert_eq!(s.get::<&str>("format").unwrap(), "P010_10LE");
        assert_eq!(s.get::<&str>("colorimetry").unwrap(), "bt2100-pq");
        assert_eq!(
            s.get::<&str>("mastering-display-info").unwrap(),
            "test-metadata"
        );
        assert!(
            output
                .features(0)
                .unwrap()
                .contains(super::CAPS_FEATURE_MEMORY_CUDA_MEMORY)
        );
    }
}
