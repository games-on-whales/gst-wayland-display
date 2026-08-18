#include <gst/vulkan/vulkan.h>

VkInstance
wayland_display_vk_instance (GstVulkanDevice *device)
{
  return device->instance->instance;
}

VkInstance
wayland_display_vk_instance_handle (GstVulkanInstance *instance)
{
  return instance->instance;
}

VkDevice
wayland_display_vk_device (GstVulkanDevice *device)
{
  return device->device;
}

guint32
wayland_display_vk_queue_family (GstVulkanQueue *queue)
{
  return queue->family;
}

VkImage
wayland_display_vk_image (GstMemory *memory)
{
  g_return_val_if_fail (gst_is_vulkan_image_memory (memory), VK_NULL_HANDLE);
  return ((GstVulkanImageMemory *) memory)->image;
}

void
wayland_display_vk_prepare_encode_image (GstMemory *memory)
{
  GstVulkanImageMemory *image;

  g_return_if_fail (gst_is_vulkan_image_memory (memory));
  image = (GstVulkanImageMemory *) memory;
  /* Preserve PR #37's fan-out contract, but make it header-checked rather than
   * writing guessed Rust byte offsets. The producer CPU-waits its write fence and
   * vulkanh26x synchronously waits encode completion before dropping the buffer. */
  if (image->barrier.parent.semaphore != VK_NULL_HANDLE) {
    /* Resolve through gst instead of calling vkDestroySemaphore directly. A direct call
     * puts libvulkan in DT_NEEDED, and ash is declared features = ["loaded"] so a missing
     * loader must stay a runtime error -- hard-linking it would break the CUDA and VA
     * builds, which never wanted Vulkan. */
    PFN_vkDestroySemaphore destroy_semaphore =
        (PFN_vkDestroySemaphore) gst_vulkan_device_get_proc_address (image->device,
        "vkDestroySemaphore");
    if (destroy_semaphore)
      destroy_semaphore (image->device->device, image->barrier.parent.semaphore, NULL);
    else
      GST_WARNING ("vkDestroySemaphore unavailable; encode-src semaphore not freed");
  }
  /* Deliberate defensive no-op: barrier.parent.queue is NULL at alloc (g_new0, never set
   * in _init), so this cannot fire from the sole call site. Kept because it states the
   * function's postcondition and matches gst's own _vk_image_mem_free. NOT a leak fix. */
  gst_clear_object (&image->barrier.parent.queue);
  image->barrier.parent.semaphore = VK_NULL_HANDLE;
  image->barrier.parent.semaphore_value = 0;
  image->barrier.image_layout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
}
