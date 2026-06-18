use std::sync::atomic::AtomicPtr;
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
    self, CAPS_FEATURE_MEMORY_CUDA_MEMORY, CUDABufferPool, CUDAContext, CudaUploader, GstCudaContext,
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
        match CUDAContext::new_from_gstreamer(&elem, -1, raw) {
            Ok(ctx) => {
                let arc = Arc::new(Mutex::new(ctx));
                *self.cuda_context.lock().unwrap() = Some(arc.clone());
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
        let uploader = uploader_guard.as_ref().ok_or(gst::FlowError::NotNegotiated)?;
        let cuda_guard = self.cuda_context.lock().unwrap();
        let cuda_arc = cuda_guard.as_ref().ok_or(gst::FlowError::NotNegotiated)?;
        let ctx = cuda_arc.lock().unwrap();

        uploader
            .upload(inbuf, &neg.in_info, &ctx, Some(&neg.pool))
            .map_err(|e| {
                gst::error!(CAT, imp = self, "dmabuf -> CUDA failed: {e}");
                gst::FlowError::Error
            })
    }
}

/// Build the opposite-direction caps, carrying over width/height/framerate.
fn transformed_caps(caps: &gst::Caps, to_cuda: bool) -> gst::Caps {
    let mut builder = if to_cuda {
        gst_video::VideoCapsBuilder::new()
            .features([CAPS_FEATURE_MEMORY_CUDA_MEMORY])
            .format(VideoFormat::Nv12)
    } else {
        gst_video::VideoCapsBuilder::new()
            .features([gstreamer_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .format(VideoFormat::DmaDrm)
    };
    if let Some(s) = caps.structure(0) {
        if let Ok(w) = s.get::<i32>("width") {
            builder = builder.width(w);
        }
        if let Ok(h) = s.get::<i32>("height") {
            builder = builder.height(h);
        }
        if let Ok(fr) = s.get::<gst::Fraction>("framerate") {
            builder = builder.framerate(fr);
        }
    }
    builder.build()
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
                "Imports an NV12 DMABuf into CUDA memory via EGLImage (convert to CUDA late)",
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
                .format(VideoFormat::Nv12)
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
        // Only build a context if we don't already have one; creating a transient
        // CUDAContext just to drop it churns refs on the shared handle.
        {
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
        let mut result = transformed_caps(caps, to_cuda);
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
