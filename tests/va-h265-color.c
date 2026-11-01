/* Exercise the production VA color policy and serialized HEVC syntax without a GPU. */
#include "gstvah265color.h"
#ifndef TEST_COLORIMETRY_FIELDS_ONLY
#include <gst/codecparsers/gsth265bitwriter.h>
#endif

static void
check_color (const gchar * name, gboolean full, guint primaries, guint transfer,
    guint matrix)
{
  GstVideoColorimetry color;
  GstH265VUIParams vui = { 0, };
  g_assert_true (gst_video_colorimetry_from_string (&color, name));
  if (full)
    color.range = GST_VIDEO_COLOR_RANGE_0_255;
  gst_va_h265_fill_colorimetry (&vui, &color);
  g_assert_true (vui.video_signal_type_present_flag);
  g_assert_cmpuint (vui.video_format, ==, 5);
  g_assert_cmpuint (vui.video_full_range_flag, ==, full);
  g_assert_true (vui.colour_description_present_flag);
  g_assert_cmpuint (vui.colour_primaries, ==, primaries);
  g_assert_cmpuint (vui.transfer_characteristics, ==, transfer);
  g_assert_cmpuint (vui.matrix_coefficients, ==, matrix);

#ifndef TEST_COLORIMETRY_FIELDS_ONLY
  GstH265VPS vps = { .id = 0, };
  GstH265SPS sps = {
    .vps = &vps,
    .temporal_id_nesting_flag = 1,
    .profile_tier_level = { .profile_idc = 2, .level_idc = 120,
      .profile_compatibility_flag = { [2] = 1, },
      .progressive_source_flag = 1, .frame_only_constraint_flag = 1, },
    .chroma_format_idc = 1,
    .pic_width_in_luma_samples = 128,
    .pic_height_in_luma_samples = 128,
    .bit_depth_luma_minus8 = 2, .bit_depth_chroma_minus8 = 2,
    .max_dec_pic_buffering_minus1 = { 1, },
    .log2_diff_max_min_luma_coding_block_size = 3,
    .log2_diff_max_min_transform_block_size = 3,
    .vui_parameters_present_flag = 1,
    .vui_params = vui,
  };
  guint8 data[4096] = { 0, }, nal[4096] = { 0, };
  guint size = sizeof (data), nal_size = sizeof (nal);
  g_assert_cmpint (gst_h265_bit_writer_sps (&sps, TRUE, data, &size),
      ==, GST_H265_BIT_WRITER_OK);
  g_assert_cmpint (gst_h265_bit_writer_convert_to_nal (4, FALSE, TRUE, FALSE,
          data, size * 8, nal, &nal_size), ==, GST_H265_BIT_WRITER_OK);
  GstH265Parser *parser = gst_h265_parser_new ();
  parser->vps[0] = vps;
  parser->vps[0].valid = TRUE;
  GstH265NalUnit unit;
  GstH265SPS parsed;
  g_assert_cmpint (gst_h265_parser_identify_nalu (parser, nal, 0, nal_size, &unit),
      ==, GST_H265_PARSER_NO_NAL_END);
  g_assert_cmpint (gst_h265_parser_parse_sps (parser, &unit, &parsed, TRUE),
      ==, GST_H265_PARSER_OK);
  g_assert_true (parsed.vui_parameters_present_flag);
  g_assert_true (parsed.vui_params.video_signal_type_present_flag);
  g_assert_true (parsed.vui_params.colour_description_present_flag);
  g_assert_cmpuint (parsed.vui_params.video_full_range_flag, ==, full);
  g_assert_cmpuint (parsed.vui_params.colour_primaries, ==, primaries);
  g_assert_cmpuint (parsed.vui_params.transfer_characteristics, ==, transfer);
  g_assert_cmpuint (parsed.vui_params.matrix_coefficients, ==, matrix);
  gst_h265_parser_free (parser);
#endif
}

#ifndef TEST_COLORIMETRY_FIELDS_ONLY
/* The HDR10 SEI parses back to the caps' metadata, including a minimum
 * luminance of 1 whose 00 00 00 01 bytes need an emulation prevention byte. */
static void
check_hdr_sei (void)
{
  GstCaps *caps = gst_caps_from_string ("video/x-raw, "
      "mastering-display-info=(string)35400:14600:8500:39850:6550:2300:15635:16450:10000000:1, "
      "content-light-level=(string)1000:400");
  guint8 nal[GST_VA_H265_HDR_SEI_MAX];
  guint nal_len = gst_va_h265_hdr_sei_nal (caps, nal);
  GstH265Parser *parser = gst_h265_parser_new ();
  GstH265NalUnit unit;
  GArray *messages = NULL;
  gboolean saw_mdcv = FALSE, saw_cll = FALSE;
  guint i;

  g_assert_cmpuint (nal_len, >, 0);
  g_assert_cmpint (gst_h265_parser_identify_nalu_unchecked (parser, nal, 0,
          nal_len, &unit), ==, GST_H265_PARSER_OK);
  g_assert_cmpint (unit.type, ==, GST_H265_NAL_PREFIX_SEI);
  g_assert_cmpint (gst_h265_parser_parse_sei (parser, &unit, &messages), ==,
      GST_H265_PARSER_OK);
  for (i = 0; i < messages->len; i++) {
    GstH265SEIMessage *m = &g_array_index (messages, GstH265SEIMessage, i);
    if (m->payloadType == GST_H265_SEI_MASTERING_DISPLAY_COLOUR_VOLUME) {
      GstH265MasteringDisplayColourVolume *v =
          &m->payload.mastering_display_colour_volume;
      /* HEVC order is green, blue, red. */
      g_assert_cmpuint (v->display_primaries_x[0], ==, 8500);
      g_assert_cmpuint (v->display_primaries_y[0], ==, 39850);
      g_assert_cmpuint (v->display_primaries_x[1], ==, 6550);
      g_assert_cmpuint (v->display_primaries_y[1], ==, 2300);
      g_assert_cmpuint (v->display_primaries_x[2], ==, 35400);
      g_assert_cmpuint (v->display_primaries_y[2], ==, 14600);
      g_assert_cmpuint (v->white_point_x, ==, 15635);
      g_assert_cmpuint (v->white_point_y, ==, 16450);
      g_assert_cmpuint (v->max_display_mastering_luminance, ==, 10000000);
      g_assert_cmpuint (v->min_display_mastering_luminance, ==, 1);
      saw_mdcv = TRUE;
    } else if (m->payloadType == GST_H265_SEI_CONTENT_LIGHT_LEVEL) {
      g_assert_cmpuint (m->payload.content_light_level.max_content_light_level,
          ==, 1000);
      g_assert_cmpuint (m->payload.content_light_level.
          max_pic_average_light_level, ==, 400);
      saw_cll = TRUE;
    }
  }
  g_assert_true (saw_mdcv);
  g_assert_true (saw_cll);
  g_array_unref (messages);
  gst_h265_parser_free (parser);
  gst_caps_unref (caps);

  /* SDR caps carry no HDR metadata, so no SEI. */
  caps = gst_caps_from_string ("video/x-raw, colorimetry=(string)bt709");
  g_assert_cmpuint (gst_va_h265_hdr_sei_nal (caps, nal), ==, 0);
  gst_caps_unref (caps);
}
#endif

int
main (int argc, char **argv)
{
  gst_init (&argc, &argv);
  check_color ("bt709", FALSE, 1, 1, 1);
  check_color ("bt2100-pq", FALSE, 9, 16, 9);
  check_color ("bt2100-pq", TRUE, 9, 16, 9);
  check_color ("bt2020", FALSE, 9, 15, 9);
  GstVideoColorimetry unknown = { 0, };
  GstH265VUIParams vui = { 0, };
  gst_va_h265_fill_colorimetry (&vui, &unknown);
  g_assert_false (vui.video_signal_type_present_flag);
  g_assert_false (vui.colour_description_present_flag);
#ifndef TEST_COLORIMETRY_FIELDS_ONLY
  check_hdr_sei ();
#endif
  g_print ("VA HEVC SDR/PQ/range/unspecified color policy and HDR SEI passed\n");
  return 0;
}
