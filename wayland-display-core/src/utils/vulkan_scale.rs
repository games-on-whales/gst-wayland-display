//! GPU scaler for `memory:VulkanImage` NV12/P010 frames — the engine behind the
//! `vulkanscale` element.
//!
//! **Why this exists.** GStreamer 1.28 ships no element that can change the size of a
//! `memory:VulkanImage` frame, and `vulkanh264enc` / `vulkanh265enc` / our vendored
//! `vulkanav1enc` accept *only* `memory:VulkanImage` NV12/P010. So on a Vulkan encode
//! path there is nothing to put behind a mutable capsfilter, and a live resolution
//! change is impossible. A `vulkandownload ! videoscale ! vulkanupload`
//! detour would be a permanent per-frame double PCIe round trip and was rejected.
//!
//! **Shape.** One compute dispatch per frame into a LINEAR storage scratch, then a
//! `vkCmdCopyImage` into an encode-src output image allocated on the *encoder's*
//! `GstVulkanDevice` — the same two-step [`VulkanNv12`](super::vulkan_nv12) uses for its
//! shared-device output, and for the same reason: an image with `VIDEO_ENCODE_SRC` usage
//! cannot also be a storage image, so the shader cannot write it directly.
//!
//! **Ordering.** Everything runs on the shared graphics+compute queue under GStreamer's
//! external-submit lock ([`super::vulkan_share::QueueSubmitGuard`]), and the submit's fence is waited
//! before the output buffer is handed downstream. NVIDIA's encode queue does not observe
//! implicit dma-buf fences, so — exactly as in the producer — the CPU wait *is* the
//! ordering primitive between our dispatch and the encoder's submission. A slot is never
//! recycled while the encoder still holds a reference to its buffer (`get_mut` gate).

#![allow(unsafe_op_in_unsafe_fn)]

use super::vulkan_nv12::{
    PixFmt, create_storage, dsl_bind, image_info, img_barrier, plane_layers, plane_view, pool_size,
    write_img,
};
use super::vulkan_share::RawVk;
use ash::vk;

type Err = Box<dyn std::error::Error>;

const VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR: i32 = 1_000_299_001;

unsafe extern "C" {
    fn wayland_display_vk_image_layout(memory: *mut gst::ffi::GstMemory) -> i32;
    fn wayland_display_vk_image_format(memory: *mut gst::ffi::GstMemory) -> u32;
    fn wayland_display_vk_image_usage(memory: *mut gst::ffi::GstMemory) -> u32;
    fn wayland_display_vk_image_flags(memory: *mut gst::ffi::GstMemory) -> u32;
}

/// Reject an input image the shader may not legally sample, naming the reason.
///
/// This is where the producer's flag choice is *read back*. `alloc_encode_src_buffer` asks
/// for `SAMPLED|TRANSFER_SRC|MUTABLE_FORMAT|EXTENDED_USAGE` and silently falls back to the
/// encode-only flags when a driver refuses the superset, so the flag set actually used is
/// recorded on the image itself (`VkImageCreateInfo`) rather than passed alongside it —
/// every producer of a `GstVulkanImageMemory` carries it, not just ours. Without this check
/// a non-sampleable or non-mutable image would reach `vkCreateImageView` and be a VUID
/// violation (undefined behaviour) instead of a diagnosable element error.
unsafe fn check_sampleable(
    mem: *mut gst::ffi::GstMemory,
    plane_views: bool,
    which: &str,
) -> Result<(), Err> {
    let usage = vk::ImageUsageFlags::from_raw(wayland_display_vk_image_usage(mem));
    let flags = vk::ImageCreateFlags::from_raw(wayland_display_vk_image_flags(mem));
    if !usage.contains(vk::ImageUsageFlags::SAMPLED) {
        return Err(format!(
            "vulkanscale: input {which} was allocated without VK_IMAGE_USAGE_SAMPLED_BIT \
             (usage {usage:?}) — the compute scaler cannot read it. From waylanddisplaysrc \
             this means the driver rejected the readable encode-src flag set and \
             alloc_encode_src_buffer fell back to encode-only; see its log line."
        )
        .into());
    }
    if plane_views && !flags.contains(vk::ImageCreateFlags::MUTABLE_FORMAT) {
        return Err(format!(
            "vulkanscale: input {which} is multiplanar but was not created \
             VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT (flags {flags:?}) — the per-plane views the \
             shader samples cannot legally be created on it."
        )
        .into());
    }
    Ok(())
}

/// The source Y/UV views for one input `VkImage` set, cached because the producer
/// recycles a small ring of images and re-creating two views per frame is pure waste.
struct SrcViews {
    key: [vk::Image; 2],
    y: vk::ImageView,
    uv: vk::ImageView,
    /// The layout the input arrives in (and is restored to), read from the
    /// `GstVulkanImageMemory` rather than assumed.
    layout: vk::ImageLayout,
    /// Multiplanar (one memory, per-plane aspects) vs one image per plane.
    multiplanar: bool,
}

/// One output ring slot: the encode-src buffer handed downstream plus the private
/// scratch/command/descriptor state it is scaled through.
///
/// The ring is **not** for GPU/CPU overlap — [`VulkanScaler::scale`] waits its fence before
/// returning, so exactly one dispatch is ever in flight. It exists because the *encoder*
/// keeps a reference to the buffer it was handed until its own encode completes, and a slot
/// must not be rewritten while it does; depth is what stops that wait from being the common
/// case.
struct ScaleOut {
    image: vk::Image,
    buffer: gst::Buffer,
    scratch: vk::Image,
    scratch_mem: vk::DeviceMemory,
    y_view: vk::ImageView,
    uv_view: vk::ImageView,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_set: vk::DescriptorSet,
}

/// What the scaler's output images must suit: the encode profiles that may read them, and
/// whether to try a LINEAR image first (see `vulkan_share::alloc_encode_src_buffer`).
pub struct ScaleOutput {
    pub targets: Vec<crate::utils::video_info::EncodeTarget>,
    pub linear_encode_src: bool,
}

/// Owns the compute pipeline and the output ring for one negotiated in→out size pair.
/// Rebuilt from scratch on a live resize (`set_caps`), which is what makes the
/// destination pool and descriptor sets follow a rung step.
pub struct VulkanScaler {
    _entry: ash::Entry,
    device: ash::Device,
    queue: vk::Queue,
    shared_queue: gstreamer_vulkan::VulkanQueue,
    /// Keeps the encoder's device alive for as long as we hold images on it.
    _shared_device: gstreamer_vulkan::VulkanDevice,
    cmd_pool: vk::CommandPool,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    desc_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    outputs: Vec<ScaleOut>,
    src_views: Vec<SrcViews>,
    next: usize,
    dst_w: u32,
    dst_h: u32,
    fmt: PixFmt,
    /// Rolling scale cost (dispatch + copy + fence wait), reported every
    /// [`REPORT_EVERY`] frames so a rung's GPU cost is visible in the log without a probe.
    frames: u64,
    total_us: u64,
    max_us: u64,
}

/// How often the rolling scale cost is logged.
const REPORT_EVERY: u64 = 300;

impl std::fmt::Debug for VulkanScaler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VulkanScaler(-> {}x{} {:?})",
            self.dst_w, self.dst_h, self.fmt
        )
    }
}

const NV12_SCALE_SPV: &[u8] = include_bytes!("shaders/nv12_scale.spv");
const P010_SCALE_SPV: &[u8] = include_bytes!("shaders/p010_scale.spv");

impl VulkanScaler {
    /// Build a scaler that writes `dst_w`x`dst_h` `fmt` frames on the encoder's device.
    ///
    /// `ring` is the output depth. Keep it at or above the number of buffers the encoder
    /// holds at once (floor 4) or a slot comes round while the encoder still references it
    /// and every frame pays the writability poll in [`Self::scale`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device_gst: gstreamer_vulkan::VulkanDevice,
        raw: RawVk,
        dst_w: u32,
        dst_h: u32,
        fmt: PixFmt,
        nearest: bool,
        output: &ScaleOutput,
        ring: usize,
    ) -> Result<Self, Err> {
        unsafe { Self::new_inner(device_gst, raw, dst_w, dst_h, fmt, nearest, output, ring) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn new_inner(
        device_gst: gstreamer_vulkan::VulkanDevice,
        raw: RawVk,
        dst_w: u32,
        dst_h: u32,
        fmt: PixFmt,
        nearest: bool,
        output: &ScaleOutput,
        ring: usize,
    ) -> Result<Self, Err> {
        let entry = ash::Entry::load()?;
        let instance = ash::Instance::load(entry.static_fn(), raw.instance);
        let device = super::vulkan_share::load_device(&instance, raw.device);
        let memp = instance.get_physical_device_memory_properties(raw.physical);

        // Two sampled sources (Y, UV) + two storage destinations (Y, UV).
        let binds = [
            dsl_bind(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            dsl_bind(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            dsl_bind(2, vk::DescriptorType::STORAGE_IMAGE),
            dsl_bind(3, vk::DescriptorType::STORAGE_IMAGE),
        ];
        let desc_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
            None,
        )?;
        let dsls = [desc_layout];
        let pcr = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&dsls)
                .push_constant_ranges(&pcr),
            None,
        )?;
        let spv = match fmt {
            PixFmt::Nv12 => NV12_SCALE_SPV,
            PixFmt::P010 => P010_SCALE_SPV,
        };
        let pipeline = build_pipeline(&device, pipeline_layout, spv)?;

        let ring = ring.max(4);
        let psizes = [
            pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 2 * ring as u32),
            pool_size(vk::DescriptorType::STORAGE_IMAGE, 2 * ring as u32),
        ];
        let desc_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(ring as u32)
                .pool_sizes(&psizes),
            None,
        )?;
        // CLAMP_TO_EDGE, not the default REPEAT: a bilinear tap at the last row/column
        // would otherwise wrap and put the opposite edge into the frame.
        let filter = if nearest {
            vk::Filter::NEAREST
        } else {
            vk::Filter::LINEAR
        };
        let sampler = device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        )?;
        let cmd_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(raw.gfx_queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;

        // As the converter's encode-src ring: CONCURRENT across graphics and video encode.
        let mut queue_families = vec![raw.gfx_queue_family];
        if let Some(enc) = raw
            .encode_queue_family
            .filter(|f| *f != raw.gfx_queue_family)
        {
            queue_families.push(enc);
        }
        let mut outputs = Vec::with_capacity(ring);
        for _ in 0..ring {
            outputs.push(create_out(
                &device,
                &memp,
                &device_gst,
                output,
                &queue_families,
                desc_pool,
                desc_layout,
                cmd_pool,
                dst_w,
                dst_h,
                fmt,
            )?);
        }

        tracing::info!(
            "vulkanscale: {}x{} {:?} ring={} filter={:?} on shared GstVulkanDevice",
            dst_w,
            dst_h,
            fmt,
            ring,
            filter
        );

        Ok(VulkanScaler {
            _entry: entry,
            device,
            queue: raw.gfx_vk_queue(),
            shared_queue: raw.gfx_queue,
            _shared_device: device_gst,
            cmd_pool,
            pipeline,
            pipeline_layout,
            desc_layout,
            desc_pool,
            sampler,
            outputs,
            src_views: Vec::new(),
            next: 0,
            dst_w,
            dst_h,
            fmt,
            frames: 0,
            total_us: 0,
            max_us: 0,
        })
    }

    /// Scale one input frame into the next ring slot and return that slot's buffer.
    pub fn scale(&mut self, input: &gst::Buffer) -> Result<gst::Buffer, Err> {
        let t0 = std::time::Instant::now();
        let out = unsafe { self.scale_inner(input) };
        let us = t0.elapsed().as_micros() as u64;
        self.frames += 1;
        self.total_us += us;
        self.max_us = self.max_us.max(us);
        if self.frames.is_multiple_of(REPORT_EVERY) {
            tracing::info!(
                "vulkanscale: {}x{} mean {:.2} ms/frame, max {:.2} ms over {} frames",
                self.dst_w,
                self.dst_h,
                self.total_us as f64 / REPORT_EVERY as f64 / 1000.0,
                self.max_us as f64 / 1000.0,
                REPORT_EVERY
            );
            self.total_us = 0;
            self.max_us = 0;
        }
        out
    }

    unsafe fn scale_inner(&mut self, input: &gst::Buffer) -> Result<gst::Buffer, Err> {
        let src = self.src_for(input)?;
        let (src_y, src_uv, src_layout, src_multiplanar, src_images) = src;

        let idx = self.next;
        self.next = (self.next + 1) % self.outputs.len();

        // Never overwrite a slot the encoder still references (the producer's G1 gate, same
        // failure mode: a GPU data hazard that shows up as green bars or device loss). The
        // wait is a poll because the thing being waited on is a GStreamer mini-object
        // refcount, and GStreamer exposes no way to block until a buffer becomes writable —
        // there is no condition variable, and a pad probe on the encoder would tell us about
        // the frame it took, not the frame it released. In steady state the slot is free on
        // the first read (the ring is deeper than the encoder's hold), so the loop body
        // normally never runs.
        //
        // On timeout we return an error, which `VulkanScale::prepare_output_buffer` turns
        // into an element error and a `FlowError` — the session fails closed. We deliberately
        // do NOT re-emit a stale frame the way the producer's G1 gate does: the producer is a
        // live source that must keep the belt moving, whereas a filter dropping a resize into
        // silence would leave the stream at the wrong size with no diagnosis.
        let mut waited = 0u32;
        while self.outputs[idx].buffer.get_mut().is_none() {
            if waited >= 10_000 {
                return Err(format!(
                    "vulkanscale: output slot {idx} still held by the encoder after 1s \
                     (ring depth {})",
                    self.outputs.len()
                )
                .into());
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
            waited += 1;
        }

        let slot = &self.outputs[idx];
        let (cmd, fence, desc_set) = (slot.cmd, slot.fence, slot.desc_set);
        let (out_img, scratch) = (slot.image, slot.scratch);

        let src_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(src_y)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let src_uv_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(src_uv)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        self.device.update_descriptor_sets(
            &[
                write_img(
                    desc_set,
                    0,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    &src_info,
                ),
                write_img(
                    desc_set,
                    1,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    &src_uv_info,
                ),
            ],
            &[],
        );

        self.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        self.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

        // --- source -> SHADER_READ_ONLY_OPTIMAL -------------------------------------
        let read = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let mut pre: Vec<vk::ImageMemoryBarrier> = Vec::with_capacity(2);
        for img in src_images
            .iter()
            .copied()
            .filter(|i| *i != vk::Image::null())
        {
            pre.push(img_barrier(
                img,
                vk::ImageAspectFlags::COLOR,
                src_layout,
                read,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_READ,
            ));
        }
        pre.push(img_barrier(
            scratch,
            vk::ImageAspectFlags::PLANE_0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
        ));
        pre.push(img_barrier(
            scratch,
            vk::ImageAspectFlags::PLANE_1,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
        ));
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &pre,
        );

        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &[desc_set],
            &[],
        );
        let pc: [i32; 2] = [self.dst_w as i32, self.dst_h as i32];
        self.device.cmd_push_constants(
            cmd,
            self.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            std::slice::from_raw_parts(pc.as_ptr() as *const u8, 8),
        );
        let groups_x = (self.dst_w.div_ceil(2)).div_ceil(8);
        let groups_y = (self.dst_h.div_ceil(2)).div_ceil(8);
        self.device.cmd_dispatch(cmd, groups_x, groups_y, 1);

        // --- scratch -> encode-src output -------------------------------------------
        let post = [
            img_barrier(
                scratch,
                vk::ImageAspectFlags::PLANE_0,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            ),
            img_barrier(
                scratch,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            ),
            img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_0,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            ),
            img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            ),
        ];
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &post,
        );
        let regions = [
            vk::ImageCopy::default()
                .src_subresource(plane_layers(vk::ImageAspectFlags::PLANE_0))
                .dst_subresource(plane_layers(vk::ImageAspectFlags::PLANE_0))
                .extent(vk::Extent3D {
                    width: self.dst_w,
                    height: self.dst_h,
                    depth: 1,
                }),
            vk::ImageCopy::default()
                .src_subresource(plane_layers(vk::ImageAspectFlags::PLANE_1))
                .dst_subresource(plane_layers(vk::ImageAspectFlags::PLANE_1))
                .extent(vk::Extent3D {
                    width: self.dst_w / 2,
                    height: self.dst_h / 2,
                    depth: 1,
                }),
        ];
        self.device.cmd_copy_image(
            cmd,
            scratch,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            out_img,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );

        // Hand the encoder its input already in VIDEO_ENCODE_SRC_KHR, and put the source
        // back in the layout its owner tracks (the producer re-uses that image with
        // oldLayout=UNDEFINED, but a mismatched tracked layout is a trap for anyone else).
        let enc_layout = vk::ImageLayout::from_raw(VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR);
        let mut tail = vec![
            img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_0,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                enc_layout,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
            ),
            img_barrier(
                out_img,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                enc_layout,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
            ),
        ];
        if src_layout != read {
            for img in src_images
                .iter()
                .copied()
                .filter(|i| *i != vk::Image::null())
            {
                tail.push(img_barrier(
                    img,
                    vk::ImageAspectFlags::COLOR,
                    read,
                    src_layout,
                    vk::AccessFlags::SHADER_READ,
                    vk::AccessFlags::empty(),
                ));
            }
        }
        let _ = src_multiplanar;
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &tail,
        );

        self.device.end_command_buffer(cmd)?;
        let cbs = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs);
        {
            let _guard = super::vulkan_share::QueueSubmitGuard::new(&self.shared_queue);
            self.device.queue_submit(self.queue, &[submit], fence)?;
        }
        // NVIDIA's encode queue ignores implicit fences, and the encoder submits from
        // another thread the instant we return the buffer: the wait is the ordering.
        self.device.wait_for_fences(&[fence], true, u64::MAX)?;
        self.device.reset_fences(&[fence])?;

        // Hand downstream a shallow HEADER copy that shares the slot's memory and carries
        // a ParentBufferMeta back to the slot buffer. The child header is writable (so
        // GstBaseTransform can retime it and the encoder can attach its own metas) while
        // the parent's refcount stays the completion sentinel the reuse gate above reads.
        let parent = &self.outputs[idx].buffer;
        let mut child = parent.copy();
        {
            let cr = child
                .get_mut()
                .ok_or("vulkanscale: fresh output child unexpectedly shared")?;
            cr.set_pts(gst::ClockTime::NONE);
            cr.set_dts(gst::ClockTime::NONE);
            cr.set_duration(gst::ClockTime::NONE);
            gst::ParentBufferMeta::add(cr, parent);
        }
        Ok(child)
    }

    /// Y/UV sampled views over `input`, created once per distinct source image set.
    #[allow(clippy::type_complexity)]
    unsafe fn src_for(
        &mut self,
        input: &gst::Buffer,
    ) -> Result<
        (
            vk::ImageView,
            vk::ImageView,
            vk::ImageLayout,
            bool,
            [vk::Image; 2],
        ),
        Err,
    > {
        let n = input.n_memory();
        if n == 0 {
            return Err("vulkanscale: input buffer has no memory".into());
        }
        let mem0 = input.peek_memory(0).as_ptr() as *mut gst::ffi::GstMemory;
        let img0 = super::vulkan_share::recover_vk_image_at(input, 0)
            .ok_or("vulkanscale: input memory 0 is not a GstVulkanImageMemory")?;
        let layout = vk::ImageLayout::from_raw(wayland_display_vk_image_layout(mem0));
        let (img1, multiplanar) = if n == 1 {
            (vk::Image::null(), true)
        } else {
            (
                super::vulkan_share::recover_vk_image_at(input, 1)
                    .ok_or("vulkanscale: input memory 1 is not a GstVulkanImageMemory")?,
                false,
            )
        };
        let key = [img0, img1];
        if let Some(v) = self.src_views.iter().find(|v| v.key == key) {
            return Ok((v.y, v.uv, v.layout, v.multiplanar, key));
        }

        let (y, uv) = if multiplanar {
            // One multiplanar image: per-plane views, which the image must have been
            // created MUTABLE_FORMAT + SAMPLED for (see
            // vulkan_share::alloc_encode_src_buffer). Verified first: creating those views
            // on an image that lacks the flags is UB, not an error return.
            check_sampleable(mem0, true, "image")?;
            (
                plane_view(
                    &self.device,
                    img0,
                    self.fmt.y_view_format(),
                    vk::ImageAspectFlags::PLANE_0,
                )?,
                plane_view(
                    &self.device,
                    img0,
                    self.fmt.uv_view_format(),
                    vk::ImageAspectFlags::PLANE_1,
                )?,
            )
        } else {
            // A generic GstVulkanImageBufferPool hands out one single-plane image per
            // plane; view each with the format it was created with. Single-plane views need
            // no format mutability, only SAMPLED.
            let mem1 = input.peek_memory(1).as_ptr() as *mut gst::ffi::GstMemory;
            check_sampleable(mem0, false, "plane 0")?;
            check_sampleable(mem1, false, "plane 1")?;
            (
                plane_view(
                    &self.device,
                    img0,
                    vk::Format::from_raw(wayland_display_vk_image_format(mem0) as i32),
                    vk::ImageAspectFlags::COLOR,
                )?,
                plane_view(
                    &self.device,
                    img1,
                    vk::Format::from_raw(wayland_display_vk_image_format(mem1) as i32),
                    vk::ImageAspectFlags::COLOR,
                )?,
            )
        };
        // Bounded: the producer recycles a handful of images, and a stale entry would
        // otherwise pin a view to a destroyed image across a source rebuild.
        if self.src_views.len() >= 8 {
            let old = self.src_views.remove(0);
            self.device.destroy_image_view(old.y, None);
            self.device.destroy_image_view(old.uv, None);
        }
        self.src_views.push(SrcViews {
            key,
            y,
            uv,
            layout,
            multiplanar,
        });
        Ok((y, uv, layout, multiplanar, key))
    }
}

impl Drop for VulkanScaler {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for v in self.src_views.drain(..) {
                self.device.destroy_image_view(v.y, None);
                self.device.destroy_image_view(v.uv, None);
            }
            for o in self.outputs.drain(..) {
                self.device.destroy_image_view(o.y_view, None);
                self.device.destroy_image_view(o.uv_view, None);
                self.device.destroy_image(o.scratch, None);
                self.device.free_memory(o.scratch_mem, None);
                self.device.destroy_fence(o.fence, None);
                drop(o.buffer);
            }
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}

unsafe fn build_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    spv: &[u8],
) -> Result<vk::Pipeline, Err> {
    let module = device.create_shader_module(
        &vk::ShaderModuleCreateInfo {
            code_size: spv.len(),
            p_code: spv.as_ptr() as *const u32,
            ..Default::default()
        },
        None,
    )?;
    let name = c"main";
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(name);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipeline = device
        .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        .map_err(|(_, e)| e)?[0];
    device.destroy_shader_module(module, None);
    Ok(pipeline)
}

#[allow(clippy::too_many_arguments)]
unsafe fn create_out(
    device: &ash::Device,
    memp: &vk::PhysicalDeviceMemoryProperties,
    gst_device: &gstreamer_vulkan::VulkanDevice,
    output: &ScaleOutput,
    queue_families: &[u32],
    desc_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    cmd_pool: vk::CommandPool,
    width: u32,
    height: u32,
    fmt: PixFmt,
) -> Result<ScaleOut, Err> {
    let buffer = super::vulkan_share::alloc_encode_src_buffer(
        gst_device,
        width,
        height,
        &output.targets,
        fmt,
        output.linear_encode_src,
        queue_families,
    )
    .ok_or("vulkanscale: encode-src output allocation failed")?;
    let image = super::vulkan_share::recover_vk_image(&buffer)
        .ok_or("vulkanscale: encode-src output is not a single GstVulkanImageMemory")?;

    let (scratch, scratch_mem, y_view, uv_view) = create_storage(
        device,
        memp,
        width,
        height,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        false,
        fmt,
    )?;

    let cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .command_buffer_count(1),
    )?[0];
    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
    let dsls = [desc_layout];
    let desc_set = device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&dsls),
    )?[0];
    let y_info = [image_info(y_view)];
    let uv_info = [image_info(uv_view)];
    device.update_descriptor_sets(
        &[
            write_img(desc_set, 2, vk::DescriptorType::STORAGE_IMAGE, &y_info),
            write_img(desc_set, 3, vk::DescriptorType::STORAGE_IMAGE, &uv_info),
        ],
        &[],
    );

    Ok(ScaleOut {
        image,
        buffer,
        scratch,
        scratch_mem,
        y_view,
        uv_view,
        cmd,
        fence,
        desc_set,
    })
}
