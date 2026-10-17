#!/usr/bin/env bash
# PHM watching the Phoenix policy that is in control of a Unitree GO2 (no shadow policy).
#
# Phoenix's policy node publishes the latent of its own forward pass on /policy/embedding
# (go2-phoenix: --phm-embedding-topic, PHOENIX_PHM_EMBEDDING_TOPIC in its harness). This
# script runs the passive PHM graph around it until SIGINT/SIGTERM or DURATION_SEC:
#   phm_detectors          robot sensor streams (phm_go2_phoenix_detectors.yaml)
#   phm_ood_cpp            rolling-spread collapse detector on /policy/embedding, with a
#                          FIXED threshold (THRESHOLD, required): it must come from earlier
#                          data, never from the run being judged
#   phm_go2_stale_monitor  stale-input check on /lowstate
#   phm_arbiter            /phm/health
#   ros2 bag record        passive recorder (BAG=1, default)
# No actuation: phm_recovery is never launched, every node subscribes and publishes only
# /phm/health and /phm/verdicts.
#
# Usage: phoenix_live_session.sh <out_dir>
#   env: THRESHOLD (required) WINDOW MIN_CONSEC ARB_STALENESS DURATION_SEC CPU_EVERY_SEC BAG
set -euo pipefail

OUT="${1:?usage: phoenix_live_session.sh <out_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THR="${THRESHOLD:?THRESHOLD is required (rolling-spread threshold from earlier data)}"
WINDOW="${WINDOW:-30}"
MIN_CONSEC="${MIN_CONSEC:-3}"
ARB_STALENESS="${ARB_STALENESS:-1.5}"  # see onboard_session.sh
DURATION_SEC="${DURATION_SEC:-0}"      # 0: until stopped
CPU_EVERY_SEC="${CPU_EVERY_SEC:-5}"
BAG="${BAG:-1}"
BAG_TOPICS=(/policy/embedding /phm/health /phm/verdicts /lowstate /lowcmd /cmd_vel
  /joint_group_position_controller/command /phoenix/estop /wirelesscontroller
  /utlidar/robot_odom /sportmodestate)
export PHM_BACKEND="${PHM_BACKEND:-plain}"

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
DETECTORS="$(find_exe phm_detectors phm_detectors_node)"
ARBITER="$(find_exe phm_arbiter phm_arbiter)"
OOD="$(find_exe phm_ood_cpp ood_node)"
STALE="$(find_exe phm_go2 phm_go2_stale_monitor)"

mkdir -p "$OUT"
NAMES=()
PIDS=()
start() {  # $1 = name, rest = command; own process group, log in $OUT/<name>.log
  local name="$1"
  shift
  setsid "$@" >"$OUT/$name.log" 2>&1 &
  NAMES+=("$name")
  PIDS+=($!)
}
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
  sleep 2
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
  echo "$(date -u +%FT%T.%NZ) stop" >>"$OUT/events.log"
}
trap cleanup EXIT
trap 'exit 0' INT TERM

cpu_seconds() {  # cumulative CPU seconds (user + system) of process group $1, as onboard_session.sh
  local pid ticks=0 tck
  tck="$(getconf CLK_TCK)"
  for pid in $(ps -o pid= -g "$1" 2>/dev/null); do
    ticks=$((ticks + $(sed 's/.*) //' "/proc/$pid/stat" 2>/dev/null | awk '{t += $12 + $13} END {print t + 0}')))
  done
  awk -v t="$ticks" -v k="$tck" 'BEGIN {printf "%.2f", t / k}'
}

{
  echo "start $(date -u +%FT%TZ) host $(hostname)"
  echo "model $(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown)"
  echo "THRESHOLD=$THR WINDOW=$WINDOW MIN_CONSEC=$MIN_CONSEC ARB_STALENESS=$ARB_STALENESS"
  echo "DURATION_SEC=$DURATION_SEC BAG=$BAG PHM_BACKEND=$PHM_BACKEND"
  echo "phm $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >"$OUT/session.txt"

if [ "$BAG" = 1 ]; then
  start bag ros2 bag record -o "$OUT/bag" "${BAG_TOPICS[@]}"
fi
start detectors "$DETECTORS" --ros-args --params-file "$HERE/phm_go2_phoenix_detectors.yaml"
start arbiter "$ARBITER" --ros-args -p staleness_sec:="$ARB_STALENESS"
start ood_cpp "$OOD" --ros-args -p threshold:="$THR" -p window:="$WINDOW" \
  -p min_consecutive:="$MIN_CONSEC"
start stale_monitor "$STALE" --ros-args -p stats_every_sec:=10.0
sleep 3
ros2 lifecycle set /phm_ood_cpp configure >>"$OUT/events.log" 2>&1
ros2 lifecycle set /phm_ood_cpp activate >>"$OUT/events.log" 2>&1
for i in "${!PIDS[@]}"; do
  kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "${NAMES[$i]} exited at startup; see $OUT/${NAMES[$i]}.log" >&2; exit 1; }
done
echo "$(date -u +%FT%T.%NZ) phm ready" >>"$OUT/events.log"
echo "PHM ready ($OUT)"

echo "utc,${NAMES[*]}" | tr ' ' ',' >"$OUT/cpu_seconds.csv"
T0="$(date +%s)"
while :; do
  row="$(date -u +%FT%T.%NZ)"
  for p in "${PIDS[@]}"; do row="$row,$(cpu_seconds "$p")"; done
  echo "$row" >>"$OUT/cpu_seconds.csv"
  if [ "$DURATION_SEC" != 0 ] && [ $(($(date +%s) - T0)) -ge "$DURATION_SEC" ]; then
    break
  fi
  sleep "$CPU_EVERY_SEC" &
  wait $! || true
done
