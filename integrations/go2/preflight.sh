#!/usr/bin/env bash
# Checks to run on the robot's onboard computer right before
# onboard_session.sh, in the same shell (ROS 2, unitree_go, the PHM overlay
# and the DDS settings already sourced and exported). Read-only: it starts
# nothing that publishes and commands nothing.
#
#   integrations/go2/preflight.sh          (env: PHM_GO2_DEPS, PHM_GO2_ONNX,
#                                           MIN_LOWSTATE_HZ, as for the session)
#
# Prints one line per check and exits non-zero if any check fails.
set -u

DEPS="${PHM_GO2_DEPS:-$HOME/phm_go2_deps}"
ONNX="${PHM_GO2_ONNX:-$DEPS/models/stand_v3_latent.onnx}"
MIN_LOWSTATE_HZ="${MIN_LOWSTATE_HZ:-400}"
fails=0
ok() { echo "ok    $*"; }
warn() { echo "warn  $*"; }
bad() { echo "FAIL  $*"; fails=$((fails + 1)); }

# 1. The session's executables are in the sourced overlay and their shared
#    libraries resolve (the shadow node finds ONNX Runtime through the path it
#    was built against, so that directory must not have moved).
IFS=: read -r -a prefixes <<<"${AMENT_PREFIX_PATH:-}"
for exe in phm_go2/phoenix_shadow_embedder phm_go2/phm_go2_probe \
  phm_detectors/phm_detectors_node phm_arbiter/phm_arbiter phm_ood_cpp/ood_node; do
  path=""
  for prefix in "${prefixes[@]}"; do
    if [ -x "$prefix/lib/$exe" ]; then
      path="$prefix/lib/$exe"
      break
    fi
  done
  if [ -z "$path" ]; then
    bad "$exe not found: source the PHM install/setup.bash (the shadow node is only built" \
      "when unitree_go is sourced and -DONNXRUNTIME_ROOT is given)"
  elif missing="$(ldd "$path" 2>/dev/null | grep 'not found')" && [ -n "$missing" ]; then
    bad "$exe has unresolved libraries: $(echo "$missing" | awk '{printf "%s ", $1}')"
  else
    ok "$exe"
  fi
done

# 2. The latent model and its external weights.
if [ -f "$ONNX" ] && [ -f "$ONNX.data" ]; then
  ok "model $ONNX (sha256 $(sha256sum "$ONNX" | cut -c1-16))"
else
  bad "model missing: need $ONNX and $ONNX.data (set PHM_GO2_DEPS or PHM_GO2_ONNX)"
fi

# 3. Nothing from an earlier run is still up, and nothing can actuate. The
#    match is on the executable (bare, by path, or as the script argument of
#    an interpreter for the 0.1.x nodes), not anywhere in a command line, so
#    an editor or a `tail` of a log does not count.
running_exe() {  # $1 = alternation of executable names
  pgrep -a -f "^([^ ]*python[0-9.]* )?([^ ]*/)?($1)(\.py)?( |\$)" || true
}
running="$(running_exe 'phoenix_shadow_embedder|phm_go2_probe|phm_detectors_node|phm_arbiter|arbiter_node|ood_node|phm_ood_node')"
if [ -n "$running" ]; then
  bad "PHM processes already running (stop them first):"
  echo "$running" | cut -c1-160 | sed 's/^/        /'
else
  ok "no PHM processes running"
fi
recovery="$(running_exe 'recovery_node|phm_recovery')"
if [ -n "$recovery" ]; then
  bad "phm_recovery is running; this session must not actuate:"
  echo "$recovery" | cut -c1-160 | sed 's/^/        /'
else
  ok "phm_recovery not running"
fi

# 4. DDS environment (a wrong interface lists topics but delivers no data).
if [ "${RMW_IMPLEMENTATION:-}" = "rmw_cyclonedds_cpp" ]; then
  ok "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
else
  warn "RMW_IMPLEMENTATION is '${RMW_IMPLEMENTATION:-unset}', expected rmw_cyclonedds_cpp"
fi
[ -n "${CYCLONEDDS_URI:-}" ] && ok "CYCLONEDDS_URI set" || warn "CYCLONEDDS_URI unset"

# 5. Robot data is flowing and deserializes as unitree_go/msg/LowState.
if ros2 interface show unitree_go/msg/LowState >/dev/null 2>&1; then
  ok "unitree_go/msg/LowState available"
else
  bad "unitree_go messages not found: source the unitree_ros2 workspace"
fi
hz="$(PYTHONUNBUFFERED=1 timeout 8 ros2 topic hz /lowstate 2>/dev/null |
  sed -n 's/.*average rate: \([0-9.]*\).*/\1/p' | tail -n 1)"
if [ -z "$hz" ]; then
  bad "no /lowstate data in 8 s (check the robot connection and CYCLONEDDS_URI)"
elif awk -v h="$hz" -v m="$MIN_LOWSTATE_HZ" 'BEGIN {exit !(h >= m)}'; then
  ok "/lowstate at $hz Hz"
else
  bad "/lowstate at $hz Hz, below $MIN_LOWSTATE_HZ Hz"
fi

# 6. Nobody publishes the recovery command topic.
pubs="$(ros2 topic info /phm/cmd_vel 2>/dev/null | sed -n 's/^Publisher count: //p')"
if [ -z "$pubs" ] || [ "$pubs" = "0" ]; then
  ok "no publisher on /phm/cmd_vel"
else
  bad "/phm/cmd_vel has $pubs publisher(s)"
fi

# 7. Room for the session output.
free_mb="$(df -Pm . | awk 'NR == 2 {print $4}')"
if [ "${free_mb:-0}" -ge 500 ]; then
  ok "${free_mb} MB free in $(pwd)"
else
  bad "only ${free_mb:-0} MB free in $(pwd)"
fi

if [ "$fails" -eq 0 ]; then
  echo "preflight passed"
else
  echo "preflight: $fails check(s) failed"
fi
exit $((fails > 0))
