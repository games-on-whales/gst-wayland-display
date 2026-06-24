# gstreamer patches

Patches to gstreamer (not this plugin) needed by the Vulkan-encode path.
Apply against a gstreamer monorepo checkout before building:

```
git apply patches/vkh264enc-dpb-pool-in-new-sequence.patch
```

## vkh264enc-dpb-pool-in-new-sequence.patch

`vulkanh264enc` creates its DPB pool in `propose_allocation`. Behind
`interpipesrc` the allocation query reaches the encoder before its
`set_format`, so the encoder isn't started yet and `create_dpb_pool` fails.
Moves the call into `new_sequence`, after `gst_vulkan_encoder_start`.

Tested on gst 1.28.4 and 1.29.1. Upstream fix pending.
