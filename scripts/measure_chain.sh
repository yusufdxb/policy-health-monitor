#!/usr/bin/env bash
# Measure the deployed PHM chain end to end on this machine, isolated from any
# robot: ROS_LOCALHOST_ONLY=1 on a private domain id, synthetic input only.
#
#   phm_chain_bench --/policy/embedding--> ood_node --/phm/verdicts--> phm_arbiter
#   phm_detectors (watching /policy/embedding and /phm_bench/aux) --/phm/verdicts-^
#   phm_arbiter --/phm/health--> recovery_node --/phm/cmd_vel--> phm_chain_bench
#
# The recovery node runs with recovery.enabled: true only inside this isolated
# graph, so fault -> zero-velocity latency is measured; nothing listens to
# /phm/cmd_vel except the benchmark. Source the overlay to measure first.
#
#   scripts/measure_chain.sh <out_dir>
# Environment (defaults): RATE_HZ=50 DIM=384 WINDOW=30 MIN_CONSEC=3 TRIALS=5
#   HEALTHY_SEC=20 FAULT_SEC=5 FAULT=collapse AUX_HZ=500 DOMAIN=87
#
# Writes chain.json (phm_chain_bench report), resources.csv (CPU percent of one
# core and peak RSS per process over the benchmark), and the node logs.
set -euo pipefail

OUT="${1:?usage: measure_chain.sh <out_dir>}"
RATE_HZ="${RATE_HZ:-50}"
DIM="${DIM:-384}"
WINDOW="${WINDOW:-30}"
MIN_CONSEC="${MIN_CONSEC:-3}"
TRIALS="${TRIALS:-5}"
HEALTHY_SEC="${HEALTHY_SEC:-20}"
FAULT_SEC="${FAULT_SEC:-5}"
FAULT="${FAULT:-collapse}"
AUX_HZ="${AUX_HZ:-500}"
export ROS_DOMAIN_ID="${DOMAIN:-87}"
export ROS_LOCALHOST_ONLY=1
mkdir -p "$OUT"

# ROS 2 rejects an integer override for a double parameter: force a float.
as_float() {
  case "$1" in
    *.* | *e* | *E*) echo "$1" ;;
    *) echo "$1.0" ;;
  esac
}

find_exe() {  # $1 = package, $2 = executable
  local prefix
  IFS=: read -r -a prefixes <<<"${AMENT_PREFIX_PATH:-}"
  for prefix in "${prefixes[@]}"; do
    if [ -x "$prefix/lib/$1/$2" ]; then
      echo "$prefix/lib/$1/$2"
      return 0
    fi
  done
  echo "missing $1/$2; source the PHM install overlay" >&2
  return 1
}

# Healthy frames are N(0, 1) per element, so the 30-frame spread is about DIM;
# half of it is a threshold no healthy window approaches and a frozen
# (constant) embedding, spread 0, always crosses.
THRESHOLD="$(awk -v d="$DIM" 'BEGIN {printf "%.1f", d / 2}')"
cat >"$OUT/detectors.yaml" <<EOF
/**:
  ros__parameters:
    freq_topics: [/policy/embedding, /phm_bench/aux]
    dead_topics: [/policy/embedding, /phm_bench/aux]
    dead_timeout_sec: 2.0
EOF
cat >"$OUT/recovery.yaml" <<EOF
/**:
  ros__parameters:
    recovery:
      enabled: true
      cooldown_seconds: 5.0
      publish_hz: 20.0
EOF

NAMES=(detectors ood arbiter recovery)
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
  sleep 1
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
}
trap cleanup EXIT

setsid "$(find_exe phm_detectors phm_detectors_node)" --ros-args \
  --params-file "$OUT/detectors.yaml" >"$OUT/detectors.log" 2>&1 &
PIDS+=($!)
setsid "$(find_exe phm_ood_cpp ood_node)" --ros-args -p threshold:="$THRESHOLD" \
  -p window:="$WINDOW" -p min_consecutive:="$MIN_CONSEC" >"$OUT/ood.log" 2>&1 &
PIDS+=($!)
setsid "$(find_exe phm_arbiter phm_arbiter)" >"$OUT/arbiter.log" 2>&1 &
PIDS+=($!)
setsid "$(find_exe phm_recovery recovery_node)" --ros-args \
  --params-file "$OUT/recovery.yaml" >"$OUT/recovery.log" 2>&1 &
PIDS+=($!)
sleep 2
ros2 lifecycle set /phm_ood_cpp configure >"$OUT/lifecycle.log" 2>&1
ros2 lifecycle set /phm_ood_cpp activate >>"$OUT/lifecycle.log" 2>&1

# CPU ticks (utime + stime) and RSS of every process in a process group.
group_ticks() {
  local total=0 pid stat
  for pid in $(pgrep -g "$1" || true); do
    stat="$(cut -d' ' -f14,15 "/proc/$pid/stat" 2>/dev/null || true)"
    [ -n "$stat" ] && total=$((total + ${stat% *} + ${stat#* }))
  done
  echo "$total"
}
group_rss_kb() {
  local total=0 pid kb
  for pid in $(pgrep -g "$1" || true); do
    kb="$(awk '/^VmRSS/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
    [ -n "$kb" ] && total=$((total + kb))
  done
  echo "$total"
}

declare -a T0 PEAK
for i in "${!PIDS[@]}"; do
  T0[$i]="$(group_ticks "${PIDS[$i]}")"
  PEAK[$i]=0
done
W0="$(date +%s.%N)"
"$(find_exe phm_sim phm_chain_bench)" --ros-args -p rate_hz:="$(as_float "$RATE_HZ")" \
  -p dim:="$DIM" -p trials:="$TRIALS" -p healthy_sec:="$(as_float "$HEALTHY_SEC")" \
  -p fault_sec:="$(as_float "$FAULT_SEC")" -p fault:="$FAULT" \
  -p aux_rate_hz:="$(as_float "$AUX_HZ")" >"$OUT/chain.json" 2>"$OUT/chain.log" &
BENCH=$!
while kill -0 "$BENCH" 2>/dev/null; do
  for i in "${!PIDS[@]}"; do
    r="$(group_rss_kb "${PIDS[$i]}")"
    [ "$r" -gt "${PEAK[$i]}" ] && PEAK[$i]="$r"
  done
  sleep 1
done
wait "$BENCH"
W1="$(date +%s.%N)"
HZ="$(getconf CLK_TCK)"
{
  echo "process,cpu_percent_of_one_core,peak_rss_kb,wall_seconds"
  for i in "${!PIDS[@]}"; do
    t1="$(group_ticks "${PIDS[$i]}")"
    awk -v n="${NAMES[$i]}" -v a="${T0[$i]}" -v b="$t1" -v hz="$HZ" -v w0="$W0" -v w1="$W1" \
      -v rss="${PEAK[$i]}" 'BEGIN {printf "%s,%.2f,%d,%.1f\n", n, 100 * (b - a) / hz / (w1 - w0), rss, w1 - w0}'
  done
} >"$OUT/resources.csv"
cat "$OUT/resources.csv"
echo "wrote $OUT/chain.json"
