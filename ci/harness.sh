#!/usr/bin/env bash
#
# Coherent build / test / benchmark harness for gst-plugin-wayland-display.
#
# One entry point that detects the GPU(s) present, builds the workspace (default
# and, when an Nvidia GPU is present, the `cuda` feature), runs the unit tests,
# the GPU-gated tests, per-vendor encode integration smoke tests, and the
# converter benchmarks -- then prints a single pass/fail summary.
#
# It encodes the environment knowledge needed to build and run on a bare box:
# LIBCLANG_PATH for bindgen, XDG_RUNTIME_DIR for the compositor, render-node
# detection, per-vendor encoder elements, and the libgstcuda link shim some
# distros need for the cuda feature.
#
# Usage:
#   ci/harness.sh [options] [phase ...]
#
#   phases (default: every phase relevant to the detected hardware):
#     detect       print the detected build/runtime environment
#     lint         cargo fmt --check
#     build        cargo build (default + cuda when applicable)
#     unit         cargo test (non-ignored) across the workspace
#     gpu          cargo test -- --ignored (the GPU-gated Rust tests)
#     integration  per-vendor `waylanddisplaysrc ! <enc> ! fakesink` to EOS,
#                  asserting clean teardown (no GLib criticals)
#     bench        converter per-frame timing (benchmark/run-conv-timing.sh)
#
#   options:
#     --features <auto|default|cuda>  feature set (default: auto)
#     --node <path>                   force a render node (default: autodetect per vendor)
#     --release                       build/test in release
#     --keep-going                    run all phases even if one fails
#     -v, --verbose                   stream sub-command output
#     -h, --help                      this help
#
# Exit status is non-zero if any selected phase failed.
set -uo pipefail

# --- locate repo root (this script lives in <root>/ci) -----------------------
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
cd "$ROOT"

# --- options -----------------------------------------------------------------
FEATURES="auto"
FORCE_NODE=""
PROFILE="dev"
TARGET_SUBDIR="debug"
KEEP_GOING=0
VERBOSE=0
PHASES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --features) FEATURES="$2"; shift 2;;
    --node) FORCE_NODE="$2"; shift 2;;
    --release) PROFILE="release"; TARGET_SUBDIR="release"; shift;;
    --keep-going) KEEP_GOING=1; shift;;
    -v|--verbose) VERBOSE=1; shift;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0;;
    detect|lint|build|unit|gpu|integration|bench) PHASES+=("$1"); shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

# --- pretty output -----------------------------------------------------------
c_reset=$'\e[0m'; c_red=$'\e[31m'; c_grn=$'\e[32m'; c_ylw=$'\e[33m'; c_cyn=$'\e[36m'
log()  { printf '%s==>%s %s\n' "$c_cyn" "$c_reset" "$*"; }
warn() { printf '%sWARN%s %s\n' "$c_ylw" "$c_reset" "$*"; }
err()  { printf '%sFAIL%s %s\n' "$c_red" "$c_reset" "$*" >&2; }

declare -A RESULT  # phase -> PASS/FAIL/SKIP
run_phase() {       # run_phase <name> <fn>
  local name="$1" fn="$2"
  log "phase: $name"
  if "$fn"; then RESULT[$name]=PASS; else
    RESULT[$name]=FAIL; err "phase $name failed"
    [[ $KEEP_GOING -eq 1 ]] || return 1
  fi
  return 0
}

# Run a command, capturing output unless --verbose. Returns the command status.
sh_run() {
  if [[ $VERBOSE -eq 1 ]]; then "$@"; else
    local out; out="$("$@" 2>&1)"; local rc=$?
    [[ $rc -eq 0 ]] || { echo "$out" | tail -40; }
    return $rc
  fi
}

# --- environment detection ---------------------------------------------------
CARGO="${CARGO:-cargo}"
LIBCLANG="${LIBCLANG_PATH:-}"
HAVE_NVIDIA=0
declare -A NODE_FOR_VENDOR  # vendor -> /dev/dri/renderDNNN

detect_env() {
  command -v "$CARGO" >/dev/null || { err "cargo not found (install rustup)"; return 1; }
  # bindgen needs libclang for input-event-codes-sys
  if [[ -z "$LIBCLANG" ]]; then
    LIBCLANG="$(dirname "$(find /usr/lib -name 'libclang.so*' 2>/dev/null | head -1)" 2>/dev/null)"
  fi
  [[ -n "$LIBCLANG" ]] && export LIBCLANG_PATH="$LIBCLANG"

  # compositor needs an XDG_RUNTIME_DIR
  if [[ -z "${XDG_RUNTIME_DIR:-}" ]] || [[ ! -d "${XDG_RUNTIME_DIR:-/nonexistent}" ]]; then
    export XDG_RUNTIME_DIR; XDG_RUNTIME_DIR="$(mktemp -d)"; chmod 700 "$XDG_RUNTIME_DIR"
  fi

  # map render nodes -> vendor via the kernel driver
  local n drv card
  for n in /dev/dri/renderD*; do
    [[ -e "$n" ]] || continue
    card="$(basename "$n")"
    drv="$(sed -n 's/^DRIVER=//p' "/sys/class/drm/$card/device/uevent" 2>/dev/null)"
    case "$drv" in
      amdgpu|radeon) NODE_FOR_VENDOR[amd]="$n";;
      i915|xe)       NODE_FOR_VENDOR[intel]="$n";;
      nvidia)        NODE_FOR_VENDOR[nvidia]="$n"; HAVE_NVIDIA=1;;
    esac
  done
}

# choose the feature flags for cargo
feature_args() {
  case "$FEATURES" in
    cuda) echo "--features cuda";;
    default) echo "";;
    auto) [[ $HAVE_NVIDIA -eq 1 ]] && echo "--features cuda" || echo "";;
    *) echo "";;
  esac
}

# gst-inspect against the freshly built plugin
PLUGIN_DIR="$ROOT/target/$TARGET_SUBDIR"
ginspect() { GST_PLUGIN_PATH="$PLUGIN_DIR" gst-inspect-1.0 "$1" >/dev/null 2>&1; }

phase_detect() {
  echo "  root            : $ROOT"
  echo "  cargo           : $($CARGO --version 2>/dev/null)"
  echo "  libclang        : ${LIBCLANG:-<not found>}"
  echo "  XDG_RUNTIME_DIR : $XDG_RUNTIME_DIR"
  echo "  profile         : $PROFILE"
  echo "  features        : $(feature_args) [requested: $FEATURES]"
  echo "  render nodes    :"
  local v
  for v in amd intel nvidia; do
    [[ -n "${NODE_FOR_VENDOR[$v]:-}" ]] && echo "      $v -> ${NODE_FOR_VENDOR[$v]}"
  done
  [[ ${#NODE_FOR_VENDOR[@]} -eq 0 ]] && echo "      (none -- GPU phases will be skipped)"
  if command -v gst-inspect-1.0 >/dev/null; then
    echo "  encoders        :"
    local e
    for e in vah265enc vah265lpenc nvh265enc vulkanh265enc vulkanh264enc dmabuftocuda; do
      gst-inspect-1.0 "$e" >/dev/null 2>&1 && echo "      $e"
    done
  fi
  return 0
}

# --- build / lint / test phases ----------------------------------------------
phase_lint() {
  if ! "$CARGO" fmt --version >/dev/null 2>&1; then
    warn "rustfmt not installed (rustup component add rustfmt); skipping lint"
    return 0
  fi
  sh_run "$CARGO" fmt --all -- --check
}

phase_build() {
  sh_run "$CARGO" build --profile "$([[ $PROFILE == dev ]] && echo dev || echo release)" \
    -p gst-plugin-wayland-display $(feature_args) || return 1
  # always also prove the default (non-cuda) build compiles, so a cfg-gated
  # caller can't rot unnoticed (this exact class of break shipped once).
  if [[ -n "$(feature_args)" ]]; then
    sh_run "$CARGO" build -p gst-plugin-wayland-display || return 1
  fi
}

phase_unit() {
  # non-ignored tests across the workspace (cargo excludes #[ignore] by default)
  sh_run "$CARGO" test --workspace $(feature_args) \
    $([[ $PROFILE == release ]] && echo --release)
}

phase_gpu() {
  if [[ ${#NODE_FOR_VENDOR[@]} -eq 0 ]]; then warn "no GPU; skipping"; return 0; fi
  local node="${FORCE_NODE:-${NODE_FOR_VENDOR[amd]:-${NODE_FOR_VENDOR[intel]:-${NODE_FOR_VENDOR[nvidia]:-}}}}"
  log "  GPU-gated Rust tests on $node"
  NV12_TEST_NODE="$node" VULKAN_ENC_NODE="$node" \
    sh_run "$CARGO" test --workspace $(feature_args) \
      $([[ $PROFILE == release ]] && echo --release) -- --ignored
}

# --- per-vendor encode integration smoke tests -------------------------------
# Run a pipeline to EOS, failing on a bus ERROR or any GLib CRITICAL at teardown
# (G_DEBUG=fatal-criticals turns the teardown double-unref class into a crash).
run_to_eos() { # run_to_eos <label> <pipeline...>
  local label="$1"; shift
  local log_out rc
  log_out="$(GST_PLUGIN_PATH="$PLUGIN_DIR" G_DEBUG=fatal-criticals RUST_BACKTRACE=1 \
    timeout 90 gst-launch-1.0 -e "$@" 2>&1)"; rc=$?
  if echo "$log_out" | grep -q "Got EOS" && [[ $rc -eq 0 ]]; then
    local mod; mod="$(echo "$log_out" | grep -oE 'drm-format=\(string\)NV12(:0x[0-9a-f]+)?' | sort -u | head -1)"
    printf '  %sok%s   %-22s %s\n' "$c_grn" "$c_reset" "$label" "${mod:-}"
    return 0
  fi
  printf '  %sFAIL%s %-22s (rc=%s)\n' "$c_red" "$c_reset" "$label" "$rc"
  echo "$log_out" | grep -iE 'error|critical|segmentation|not-negotiat|reconstruct|assertion' \
    | grep -viE 'EGL|GL_' | head -6 | sed 's/^/        /'
  return 1
}

phase_integration() {
  if [[ ${#NODE_FOR_VENDOR[@]} -eq 0 ]]; then warn "no GPU; skipping"; return 0; fi
  command -v gst-inspect-1.0 >/dev/null || { warn "no gst-launch; skipping"; return 0; }
  ginspect waylanddisplaysrc || { err "plugin not built (run 'build' first)"; return 1; }
  local ok=1 node res="width=1280,height=720,framerate=60/1"

  # AMD / Intel: the VA encoder pulls NV12 straight from the source dmabuf.
  if [[ -n "${NODE_FOR_VENDOR[amd]:-}" ]] && gst-inspect-1.0 vah265enc >/dev/null 2>&1; then
    node="${NODE_FOR_VENDOR[amd]}"
    run_to_eos "amd:vah265enc" waylanddisplaysrc render-node="$node" num-buffers=30 \
      ! vah265enc ! fakesink || ok=0
  fi
  if [[ -n "${NODE_FOR_VENDOR[intel]:-}" ]] && gst-inspect-1.0 vah265lpenc >/dev/null 2>&1; then
    node="${NODE_FOR_VENDOR[intel]}"
    LIBVA_DRIVER_NAME=iHD run_to_eos "intel:vah265lpenc" \
      waylanddisplaysrc render-node="$node" num-buffers=30 ! vah265lpenc ! fakesink || ok=0
  fi
  # Nvidia: NV12 dmabuf -> CUDA -> NVENC (needs the cuda feature build + nvcodec).
  if [[ -n "${NODE_FOR_VENDOR[nvidia]:-}" ]] && gst-inspect-1.0 nvh265enc >/dev/null 2>&1 \
     && ginspect dmabuftocuda; then
    node="${NODE_FOR_VENDOR[nvidia]}"
    run_to_eos "nvidia:nvh265enc" waylanddisplaysrc render-node="$node" num-buffers=30 \
      ! "video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12,$res" \
      ! dmabuftocuda render-node="$node" ! nvh265enc ! fakesink || ok=0
  fi
  # Vulkan encode (any vendor with vulkanh264enc): the source harvests the
  # encoder's GstVulkanDevice and hands it a shared-device NV12 memory:VulkanImage.
  if gst-inspect-1.0 vulkanh264enc >/dev/null 2>&1; then
    node="${FORCE_NODE:-${NODE_FOR_VENDOR[nvidia]:-${NODE_FOR_VENDOR[amd]:-${NODE_FOR_VENDOR[intel]:-}}}}"
    if [[ -n "$node" ]]; then
      run_to_eos "vulkan:vulkanh264enc" waylanddisplaysrc render-node="$node" vulkan=true \
        num-buffers=30 ! vulkanh264enc ! fakesink || ok=0
    fi
  fi
  [[ $ok -eq 1 ]]
}

# --- benchmark ---------------------------------------------------------------
phase_bench() {
  if [[ ${#NODE_FOR_VENDOR[@]} -eq 0 ]]; then warn "no GPU; skipping"; return 0; fi
  [[ -f benchmark/conv-timing.patch ]] || { warn "no benchmark; skipping"; return 0; }
  local applied=0
  if git apply --check benchmark/conv-timing.patch 2>/dev/null; then
    git apply benchmark/conv-timing.patch && applied=1
  else
    warn "conv-timing.patch does not apply cleanly; skipping bench"; return 0
  fi
  sh_run "$CARGO" build -p gst-plugin-wayland-display $(feature_args) || { [[ $applied -eq 1 ]] && git apply -R benchmark/conv-timing.patch; return 1; }

  local v node tail rc=0
  for v in amd intel nvidia; do
    node="${NODE_FOR_VENDOR[$v]:-}"; [[ -n "$node" ]] || continue
    case "$v" in
      nvidia) tail="dmabuftocuda render-node=$node ! nvh265enc ! fakesink sync=false"; export DRM_FORMAT=NV12;;
      intel)  tail="vah265lpenc ! fakesink sync=false"; unset DRM_FORMAT;;
      amd)    tail="vah265enc ! fakesink sync=false"; unset DRM_FORMAT;;
    esac
    log "  bench $v ($node)"
    GST_PLUGIN_PATH="$PLUGIN_DIR" ./benchmark/run-conv-timing.sh "$node" "$tail" 1280 720 120 240 2>&1 \
      | grep -i CONVTIMING | tail -1 | sed "s/^/      $v 720p: /" || rc=1
  done
  unset DRM_FORMAT
  [[ $applied -eq 1 ]] && git apply -R benchmark/conv-timing.patch
  return $rc
}

# --- driver ------------------------------------------------------------------
detect_env || exit 1

if [[ ${#PHASES[@]} -eq 0 ]]; then
  PHASES=(detect lint build unit gpu integration bench)
fi

declare -A PHASE_FN=(
  [detect]=phase_detect [lint]=phase_lint [build]=phase_build [unit]=phase_unit
  [gpu]=phase_gpu [integration]=phase_integration [bench]=phase_bench
)

overall=0
for p in "${PHASES[@]}"; do
  run_phase "$p" "${PHASE_FN[$p]}" || { overall=1; break; }
done

echo
log "summary"
for p in "${PHASES[@]}"; do
  r="${RESULT[$p]:-SKIP}"
  col="$c_ylw"; [[ "$r" == PASS ]] && col="$c_grn"; [[ "$r" == FAIL ]] && col="$c_red"
  printf '  %-12s %s%s%s\n' "$p" "$col" "$r" "$c_reset"
  [[ "$r" == FAIL ]] && overall=1
done
exit $overall
