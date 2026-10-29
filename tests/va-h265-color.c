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
  g_print ("VA HEVC SDR/PQ/range/unspecified color policy passed\n");
  return 0;
}
