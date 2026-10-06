#!/usr/bin/env bash
# PHM onboard session on a Unitree GO2's onboard computer. No actuation:
# phm_recovery is never launched, and the shadow policy publishes only
# /policy/embedding.
#
# Phases (evidence lands in $OUT):
#   1 calibrate   CAL_SEC of live latent -> rolling-spread threshold (calib.npz)
#   2 nominal     NOM_SEC of the full graph with no fault -> false-alarm rate, CPU cost
#   3 freeze_obs  TRIALS runs: policy fed a stale snapshot FAULT_AT s into FAULT_SEC
#   4 stop        TRIALS runs: policy process goes silent FAULT_AT s into FAULT_SEC
#   (optional) sensors  freeze_sensors trials: stale sensor reading, live last action;
#                 enabled through FAULTS (default "freeze stop")
#
# Every process starts in its own process group (setsid) and is stopped by group.
# Executables are run directly from the sourced colcon overlay (no `ros2 run`
# wrapper process per node).
#
# Usage: onboard_session.sh <out_dir>     (env: see defaults below)
set -euo pipefail

OUT="${1:?usage: onboard_session.sh <out_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="${PHM_GO2_DEPS:-$HOME/phm_go2_deps}"
ONNX="${PHM_GO2_ONNX:-$DEPS/models/stand_v3_latent.onnx}"
CAL_SEC="${CAL_SEC:-60}"
NOM_SEC="${NOM_SEC:-120}"
TRIALS="${TRIALS:-1}"
FAULT_SEC="${FAULT_SEC:-45}"
FAULT_AT="${FAULT_AT:-20}"
FAULTS="${FAULTS:-freeze stop}"  # fault kinds per trial round, in order: freeze stop sensors
WINDOW="${WINDOW:-30}"
PERCENTILE="${PERCENTILE:-1.0}"
MIN_CONSEC="${MIN_CONSEC:-3}"
# Above the detectors node's 1 s verdict period: at exactly 1.0 the arbiter's
# 20 Hz check can land a few ms before the next 1 Hz verdict arrives and report
# a brief stale DEGRADED. The C++ nodes start within milliseconds of each other,
# which lines their timers up that way. The 0.1.x sessions used 1.0.
ARB_STALENESS="${ARB_STALENESS:-1.5}"
export PHM_BACKEND="${PHM_BACKEND:-plain}"  # plain measured about 2x faster than eigen on the GO2

# Embedder fault mode of a fault kind (freeze | stop | sensors).
fault_mode() {
  case "$1" in
    freeze) echo freeze_obs ;;
    stop) echo stop ;;
    sensors) echo freeze_sensors ;;
    *) return 1 ;;
  esac
}

# Validate FAULTS before anything starts.
read -r -a FAULT_KINDS <<<"$FAULTS"
if [ "${#FAULT_KINDS[@]}" -eq 0 ]; then
  echo "FAULTS is empty; use any of: freeze stop sensors" >&2
  exit 2
fi
for kind in "${FAULT_KINDS[@]}"; do
  if ! fault_mode "$kind" >/dev/null; then
    echo "unknown fault kind '$kind' in FAULTS='$FAULTS'; valid kinds: freeze stop sensors" >&2
    exit 2
  fi
  if [ "$(printf '%s\n' "${FAULT_KINDS[@]}" | grep -cx -- "$kind")" -ne 1 ]; then
    echo "fault kind '$kind' repeated in FAULTS='$FAULTS'" >&2
    exit 2
  fi
done

# Path of an executable installed by a colcon package in the sourced overlay.
find_exe() {  # $1 = package, $2 = executable
  local prefix
  IFS=: read -r -a prefixes <<<"${AMENT_PREFIX_PATH:-}"
  for prefix in "${prefixes[@]}"; do
    if [ -x "$prefix/lib/$1/$2" ]; then
      echo "$prefix/lib/$1/$2"
      return 0
    fi
  done
  echo "missing $1/$2 in AMENT_PREFIX_PATH; source the PHM install overlay" >&2
  return 1
}
EMBEDDER="$(find_exe phm_go2 phoenix_shadow_embedder)"
PROBE="$(find_exe phm_go2 phm_go2_probe)"
DETECTORS="$(find_exe phm_detectors phm_detectors_node)"
ARBITER="$(find_exe phm_arbiter phm_arbiter)"
OOD="$(find_exe phm_ood_cpp ood_node)"

mkdir -p "$OUT"
PIDS=()
EMB_PID=""

cleanup() {
  for p in "${PIDS[@]}" $EMB_PID; do kill -INT -- "-$p" 2>/dev/null || true; done
  sleep 2
  for p in "${PIDS[@]}" $EMB_PID; do kill -KILL -- "-$p" 2>/dev/null || true; done
}

cpu_seconds() {  # cumulative CPU seconds (user + system) of every process in group $1
  # From /proc in clock ticks: `ps -o times` counts whole seconds, which rounds
  # a node using about 1% of a core to zero over a two-minute phase.
  local pid ticks=0 tck
  tck="$(getconf CLK_TCK)"
  for pid in $(ps -o pid= -g "$1" 2>/dev/null); do
    # Fields after "pid (comm) ": $12 = utime, $13 = stime.
    ticks=$((ticks + $(sed 's/.*) //' "/proc/$pid/stat" 2>/dev/null | awk '{t += $12 + $13} END {print t + 0}')))
  done
  awk -v t="$ticks" -v k="$tck" 'BEGIN {printf "%.2f", t / k}'
}
trap cleanup EXIT

start_embedder() {  # $1 = fault (none | freeze_obs | freeze_sensors | stop), $2 = log name
  setsid "$EMBEDDER" --ros-args \
    -p onnx_path:="$ONNX" -p stats_every_sec:=10.0 \
    -p fault:="$1" -p fault_after_sec:="$(printf '%.1f' "$FAULT_AT")" >"$OUT/$2" 2>&1 &
  EMB_PID=$!
  sleep 3
  if ! kill -0 "$EMB_PID" 2>/dev/null; then
    echo "embedder exited at startup; see $OUT/$2" >&2
    exit 1
  fi
  echo "$(date -u +%FT%T.%NZ) embedder start fault=$1 pid=$EMB_PID" >>"$OUT/events.log"
}

stop_embedder() {
  [ -n "$EMB_PID" ] || return 0
  kill -INT -- "-$EMB_PID" 2>/dev/null || true
  wait "$EMB_PID" 2>/dev/null || true
  EMB_PID=""
}

{
  echo "start $(date -u +%FT%TZ) host $(hostname)"
  echo "model $(tr -d '\0' </proc/device-tree/model)"
  echo "onnx $(sha256sum "$ONNX" | cut -c1-16)"
  echo "CAL_SEC=$CAL_SEC NOM_SEC=$NOM_SEC FAULT_SEC=$FAULT_SEC FAULT_AT=$FAULT_AT"
  echo "WINDOW=$WINDOW PERCENTILE=$PERCENTILE MIN_CONSEC=$MIN_CONSEC ARB_STALENESS=$ARB_STALENESS"
  echo "TRIALS=$TRIALS PHM_BACKEND=$PHM_BACKEND FAULTS=$FAULTS"
  echo "phm $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >"$OUT/session.txt"

# -- 1 calibrate -----------------------------------------------------------
start_embedder none embedder_calibrate.log
"$PROBE" calibrate --seconds "$CAL_SEC" --window "$WINDOW" \
  --percentile "$PERCENTILE" --out "$OUT/calib.npz" | tee "$OUT/calibrate.json"
THR="$(sed -n 's/.*"threshold": \([^,}]*\).*/\1/p' "$OUT/calibrate.json" | tail -n 1)"
[ -n "$THR" ] || { echo "calibration failed; see $OUT/calibrate.json" >&2; exit 1; }
echo "threshold $THR" >>"$OUT/session.txt"

# -- PHM graph (stays up through phases 2-4) -------------------------------
setsid "$DETECTORS" --ros-args \
  --params-file "$HERE/phm_go2_detectors.yaml" >"$OUT/detectors.log" 2>&1 &
PIDS+=($!)
setsid "$ARBITER" --ros-args \
  -p staleness_sec:="$ARB_STALENESS" >"$OUT/arbiter.log" 2>&1 &
PIDS+=($!)
setsid "$OOD" --ros-args -p threshold:="$THR" -p window:="$WINDOW" \
  -p min_consecutive:="$MIN_CONSEC" >"$OUT/ood_cpp.log" 2>&1 &
PIDS+=($!)
sleep 3
ros2 lifecycle set /phm_ood_cpp configure >>"$OUT/events.log" 2>&1
ros2 lifecycle set /phm_ood_cpp activate >>"$OUT/events.log" 2>&1
# Frequency detectors learn their baseline over the first ticks; let them settle.
sleep 12

# -- 2 nominal --------------------------------------------------------------
echo "$(date -u +%FT%T.%NZ) phase nominal" >>"$OUT/events.log"
NAMES=(detectors arbiter ood_cpp embedder)
GROUPS_=("${PIDS[@]}" "$EMB_PID")
declare -a CPU0
for i in "${!GROUPS_[@]}"; do CPU0[$i]="$(cpu_seconds "${GROUPS_[$i]}")"; done
T0="$(date +%s.%N)"
"$PROBE" record --seconds "$NOM_SEC" --window "$WINDOW" \
  --out "$OUT/nominal.jsonl"
T1="$(date +%s.%N)"
{
  echo "# process, cpu_seconds, wall_seconds, percent_of_one_core (nominal phase)"
  for i in "${!GROUPS_[@]}"; do
    c1="$(cpu_seconds "${GROUPS_[$i]}")"
    awk -v n="${NAMES[$i]}" -v a="${CPU0[$i]}" -v b="$c1" -v t0="$T0" -v t1="$T1" \
      'BEGIN {printf "%s, %.2f, %.1f, %.2f\n", n, b - a, t1 - t0, 100 * (b - a) / (t1 - t0)}'
  done
  echo "# nproc $(nproc)"
} >"$OUT/cpu_nominal.csv"
top -b -n 1 -w 200 | head -25 >"$OUT/top_nominal.txt"
stop_embedder

# -- 3, 4 induced faults, TRIALS of each, interleaved (kinds in FAULTS order) ---
for trial in $(seq 1 "$TRIALS"); do
  for fault in "${FAULT_KINDS[@]}"; do
    mode="$(fault_mode "$fault")"
    start_embedder "$mode" "embedder_${fault}_${trial}.log"
    echo "$(date -u +%FT%T.%NZ) phase ${mode} trial ${trial} (fault at +${FAULT_AT}s)" \
      >>"$OUT/events.log"
    "$PROBE" record --seconds "$FAULT_SEC" --window "$WINDOW" \
      --out "$OUT/fault_${fault}_${trial}.jsonl"
    stop_embedder
  done
done

echo "end $(date -u +%FT%TZ)" >>"$OUT/session.txt"
echo DONE >>"$OUT/session.txt"
