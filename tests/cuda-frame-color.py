"""Exercise the patched CUDA decision and math with real GstBuffer metadata.

No CUDA device is required. This does not replace GPU pixel/ownership tests.
The production decision, PQ LUT and kernel functions are extracted verbatim.
"""
import ast
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

source_root = pathlib.Path(sys.argv[1])
converter = (source_root / "subprojects/gst-plugins-bad/sys/nvcodec/gstcudaconverter.cpp").read_text()
kernel = (source_root / "subprojects/gst-plugins-bad/sys/nvcodec/kernel/gstcudaconverter.cu").read_text()
producer = (pathlib.Path(__file__).resolve().parents[1] /
            "wayland-display-core/src/utils/source_color.rs").read_text()
meta_name = re.search(r'pub const SOURCE_COLOR_META: &str = "([^"]+)";', producer).group(1)
assert '"' + meta_name + '"' in converter, "producer and consumer metadata ABI differ"
start = converter.index("  /*\n   * Unraid Beam keeps CUDA allocation caps stable")
decision = converter[start:converter.index("  if (!priv->fill_border", start)]
lut = converter[converter.index("#define PQ_TRANSFER_LUT_SIZE"):converter.index("/* from GstD3D11 */")]
math_start = kernel.index("__device__ inline float\npq_oetf")
math_end = kernel.index("__device__ inline unsigned char\nscale_to_2bits", math_start)
math = kernel[math_start:math_end]
# GStreamer ships both CUDA source and an NVRTC source string. Check both paths.
embedded = "".join(ast.literal_eval(line.strip()) for line in
                   kernel[kernel.index("static const char GstCudaConverterMain_str[]"):].splitlines()
                   if line.strip().startswith('"') and line.strip().endswith('"'))
embedded_start = embedded.index("__device__ inline float\npq_oetf")
embedded_end = embedded.index("__device__ inline unsigned char\nscale_to_2bits", embedded_start)


def tokens(text):
    return re.sub(r"\s+", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


assert tokens(math) == tokens(embedded[embedded_start:embedded_end])
assert "linear_bt709_to_bt2100_pq (rgb, pq_transfer_texture)" in embedded

program = r'''
#include <gst/gst.h>
#include <gst/video/video.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
constexpr unsigned CU_TRSF_SRGB = 16;
struct Constants { int transfer_mode; };
struct Private { GstVideoInfo out_info, texture_info; Constants *const_buf; };
struct Frame { GstBuffer *buffer; };
static void require(bool ok, const char *message) {
  if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
''' + lut + r'''
int decide(bool hdr, bool rgb, int initial, GstBuffer *buffer) {
  Constants constants{initial};
  Private state{};
  gst_video_info_set_format(&state.out_info, GST_VIDEO_FORMAT_P010_10LE, 16, 16);
  gst_video_info_set_format(&state.texture_info,
      rgb ? GST_VIDEO_FORMAT_RGBA : GST_VIDEO_FORMAT_NV12, 16, 16);
  state.out_info.colorimetry.transfer = hdr ? GST_VIDEO_TRANSFER_SMPTE2084 : GST_VIDEO_TRANSFER_BT709;
  state.const_buf = &constants;
  Private *priv = &state;
  Frame frame{buffer};
  Frame *src_frame = &frame;
  guint texture_flags = 0;
''' + decision + r'''
  return constants.transfer_mode | (texture_flags ? 2 : 0);
}
// CPU sampling of the same uploaded, hardware-interpolated 1024-entry LUT.
#define __device__
using cudaTextureObject_t = unsigned long long;
struct float3 { float x, y, z; };
float3 make_float3(float x,float y,float z) { return {x,y,z}; }
float clamp(float v,float a,float b) { return std::max(a,std::min(v,b)); }
std::array<float,PQ_TRANSFER_LUT_SIZE> table;
template<typename T> T tex2D(cudaTextureObject_t, float coordinate, float) {
  float x = clamp(coordinate * (float)table.size() - .5f, 0.f, (float)table.size()-1.f);
  unsigned a = (unsigned)x, b = std::min(a+1, (unsigned)table.size()-1);
  return table[a] + (table[b]-table[a]) * (x-(float)a);
}
''' + math + r'''
int main(int argc, char **argv) {
  gst_init(&argc,&argv);
  const gchar *tags[] = {nullptr};
  gst_meta_register_custom("UnraidBeamSourceColorMeta", tags, nullptr, nullptr, nullptr);
  GstBuffer *missing = gst_buffer_new(), *malformed = gst_buffer_new();
  GstCustomMeta *meta = gst_buffer_add_custom_meta(malformed, "UnraidBeamSourceColorMeta");
  gst_structure_set(gst_custom_meta_get_structure(meta), "pq-passthrough", G_TYPE_STRING, "true", nullptr);
  require(decide(true,true,1,missing)==3, "missing metadata must preserve configured conversion");
  require(decide(true,true,0,malformed)==0, "malformed metadata must not invent PQ");
  int previous = 1;
  for (bool pq : {false,true,false,true,false}) {
    GstBuffer *frame = gst_buffer_new_allocate(nullptr,16,nullptr);
    meta = gst_buffer_add_custom_meta(frame,"UnraidBeamSourceColorMeta");
    gst_structure_set(gst_custom_meta_get_structure(meta),"pq-passthrough",G_TYPE_BOOLEAN,pq,nullptr);
    GstBuffer *copy = gst_buffer_copy_deep(frame);
    require(decide(true,true,previous,copy)==(pq?0:3), "SDR/PQ/SDR decision on copied frame");
    require(decide(false,true,0,copy)==0, "SDR output must not apply PQ transfer");
    require(decide(true,false,0,copy)==0, "YUV input must not apply RGB transfer");
    previous = pq?0:1;
    gst_buffer_unref(copy); gst_buffer_unref(frame);
  }
  for (unsigned i=0;i<table.size();++i) {
    float x=(float)i/(table.size()-1);
    table[i]=pq_oetf_reference(BT2408_GRAPHICS_WHITE*x*x);
  }
  // Independent GStreamer PQ transfer reference. Preserve the downstream
  // 12-code acceptance limit; do not fit an inverse gamma to screenshots.
  double maximum_error=0;
  for (unsigned i=0;i<=10000;++i) {
    float linear=(float)i/10000;
    float3 actual=linear_bt709_to_bt2100_pq(make_float3(linear,linear,linear),0);
    double expected=gst_video_transfer_function_encode(GST_VIDEO_TRANSFER_SMPTE2084,
        203.0/10000.0*linear);
    for (float channel : {actual.x,actual.y,actual.z}) {
      double error=std::abs(channel-expected)*876.0;
      maximum_error=std::max(maximum_error,error);
      require(error<=12, "production LUT differs from independent PQ reference");
    }
  }
  // Non-neutral samples exercise the actual kernel's BT.709 -> BT.2020 matrix.
  for (float3 c : {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1),make_float3(1,.21404114f,0)}) {
    float3 actual=linear_bt709_to_bt2100_pq(c,0);
    double expected[3]={.627404*c.x+.329282*c.y+.0433136*c.z,
        .069097*c.x+.919540*c.y+.0113612*c.z,
        .0163916*c.x+.0880132*c.y+.895595*c.z};
    float channels[3]={actual.x,actual.y,actual.z};
    for (int i=0;i<3;++i) require(std::abs(channels[i]-gst_video_transfer_function_encode(
        GST_VIDEO_TRANSFER_SMPTE2084,expected[i]*203.0/10000.0))*876<=12,
        "production gamut conversion changed");
  }
  gst_buffer_unref(missing); gst_buffer_unref(malformed);
  std::printf("Real GstMeta/copy/SDR-PQ-SDR policy passed; PQ ramp max error %.4f codes\n",maximum_error);
}
'''
with tempfile.TemporaryDirectory(prefix="gwd-frame-color-") as directory:
    root = pathlib.Path(directory)
    (root / "test.cpp").write_text(program)
    flags = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "gstreamer-video-1.0"], text=True))
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                    "-Werror", str(root / "test.cpp"), "-o", str(root / "test"), *flags], check=True)
    subprocess.run([str(root / "test")], check=True)
