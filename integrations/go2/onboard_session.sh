#!/usr/bin/env bash
# PHM onboard session on a Unitree GO2 Jetson. No actuation: phm_recovery is never
# launched, and the shadow policy publishes only /policy/embedding.
#
# Phases (evidence lands in $OUT):
#   1 calibrate   CAL_SEC of live latent -> rolling-spread threshold (calib.npz)
#   2 nominal     NOM_SEC of the full graph with no fault -> false-alarm rate, CPU cost
#   3 freeze_obs  TRIALS runs: policy fed a stale snapshot FAULT_AT s into FAULT_SEC
#   4 stop        TRIALS runs: policy process goes silent FAULT_AT s into FAULT_SEC
#
# Every process starts in its own process group (setsid) and is stopped by group:
# `ros2 run` forks the node, and signalling only the wrapper PID left nodes running.
#
# Usage: onboard_session.sh <out_dir>     (env: see defaults below)
set -euo pipefail

OUT="${1:?usage: onboard_session.sh <out_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="${PHM_GO2_DEPS:-$HOME/yusuf/phm_go2_deps}"
ONNX="${PHM_GO2_ONNX:-$DEPS/models/stand_v3_latent.onnx}"
CAL_SEC="${CAL_SEC:-60}"
NOM_SEC="${NOM_SEC:-120}"
TRIALS="${TRIALS:-1}"
FAULT_SEC="${FAULT_SEC:-45}"
FAULT_AT="${FAULT_AT:-20}"
WINDOW="${WINDOW:-30}"
PERCENTILE="${PERCENTILE:-1.0}"
MIN_CONSEC="${MIN_CONSEC:-3}"
ARB_STALENESS="${ARB_STALENESS:-1.0}"
export PHM_BACKEND="${PHM_BACKEND:-plain}"  # plain measured 2x faster than Eigen on the GO2

mkdir -p "$OUT"
PIDS=()
EMB_PID=""

cleanup() {
  for p in "${PIDS[@]}" $EMB_PID; do kill -INT -- "-$p" 2>/dev/null || true; done
  sleep 2
  for p in "${PIDS[@]}" $EMB_PID; do kill -KILL -- "-$p" 2>/dev/null || true; done
}

cpu_seconds() {  # cumulative CPU seconds of every process in group $1
  ps -o times= -g "$1" 2>/dev/null | awk '{s += $1} END {print s + 0}'
}
trap cleanup EXIT

start_embedder() {  # $1 = fault (none | freeze_obs | stop), $2 = log name
  setsid python3 "$HERE/phoenix_shadow_embedder.py" --ros-args \
    -p onnx_path:="$ONNX" -p phoenix_src:="$DEPS" -p stats_every_sec:=10.0 \
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
  echo "TRIALS=$TRIALS PHM_BACKEND=$PHM_BACKEND"
  echo "phm $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >"$OUT/session.txt"

# -- 1 calibrate -----------------------------------------------------------
start_embedder none embedder_calibrate.log
python3 "$HERE/phm_go2_probe.py" calibrate --seconds "$CAL_SEC" --window "$WINDOW" \
  --percentile "$PERCENTILE" --out "$OUT/calib.npz" | tee "$OUT/calibrate.json"
THR="$(python3 -c "import numpy as n; print(float(n.load('$OUT/calib.npz')['threshold']))")"
echo "threshold $THR" >>"$OUT/session.txt"

# -- PHM graph (stays up through phases 2-4) -------------------------------
setsid ros2 run phm_detectors phm_detectors_node --ros-args \
  --params-file "$HERE/phm_go2_detectors.yaml" >"$OUT/detectors.log" 2>&1 &
PIDS+=($!)
setsid ros2 run phm_arbiter phm_arbiter --ros-args \
  -p staleness_sec:="$ARB_STALENESS" >"$OUT/arbiter.log" 2>&1 &
PIDS+=($!)
setsid ros2 run phm_ood_cpp ood_node --ros-args -p threshold:="$THR" -p window:="$WINDOW" \
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
python3 "$HERE/phm_go2_probe.py" record --seconds "$NOM_SEC" --window "$WINDOW" \
  --out "$OUT/nominal.jsonl"
T1="$(date +%s.%N)"
{
  echo "# process, cpu_seconds, wall_seconds, percent_of_one_core (nominal phase)"
  for i in "${!GROUPS_[@]}"; do
    c1="$(cpu_seconds "${GROUPS_[$i]}")"
    awk -v n="${NAMES[$i]}" -v a="${CPU0[$i]}" -v b="$c1" -v t0="$T0" -v t1="$T1" \
      'BEGIN {printf "%s, %d, %.1f, %.1f\n", n, b - a, t1 - t0, 100 * (b - a) / (t1 - t0)}'
  done
  echo "# nproc $(nproc)"
} >"$OUT/cpu_nominal.csv"
top -b -n 1 -w 200 | head -25 >"$OUT/top_nominal.txt"
stop_embedder

# -- 3, 4 induced faults, TRIALS of each, interleaved --------------------------
for trial in $(seq 1 "$TRIALS"); do
  for fault in freeze stop; do
    [ "$fault" = freeze ] && mode=freeze_obs || mode=stop
    start_embedder "$mode" "embedder_${fault}_${trial}.log"
    echo "$(date -u +%FT%T.%NZ) phase ${mode} trial ${trial} (fault at +${FAULT_AT}s)" \
      >>"$OUT/events.log"
    python3 "$HERE/phm_go2_probe.py" record --seconds "$FAULT_SEC" --window "$WINDOW" \
      --out "$OUT/fault_${fault}_${trial}.jsonl"
    stop_embedder
  done
done

echo "end $(date -u +%FT%TZ)" >>"$OUT/session.txt"
echo DONE >>"$OUT/session.txt"
