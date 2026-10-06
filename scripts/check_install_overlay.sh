#!/usr/bin/env bash
# Smoke-check a sourced colcon install overlay (run after `colcon build` and
# `source install/setup.bash`, from any directory):
#   - every PHM package resolves through the ament index;
#   - every documented executable is installed, executable, and has all its
#     shared libraries resolvable;
#   - the launch files and configs are installed;
#   - no project Python was installed (only phm_msgs' generated bindings);
#   - phm_core's exported CMake package builds and links a consumer.
# Optional packages (phm_go2's shadow embedder and replay need ONNX Runtime and
# unitree_go) are reported but not required.
set -euo pipefail

if [ -z "${AMENT_PREFIX_PATH:-}" ]; then
  echo "ERROR: no colcon overlay detected; source /opt/ros/<distro>/setup.bash and install/setup.bash" >&2
  exit 2
fi
IFS=: read -r -a prefixes <<<"$AMENT_PREFIX_PATH"

failures=0
fail() {
  echo "  FAIL: $1" >&2
  failures=$((failures + 1))
}
prefix_of() {  # package -> install prefix, via the ament resource index
  local p
  for p in "${prefixes[@]}"; do
    if [ -e "$p/share/ament_index/resource_index/packages/$1" ]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

packages=(phm_msgs phm_core phm_tools phm_detectors phm_ood phm_ood_cpp phm_arbiter
  phm_recovery phm_sim phm_go2)
declare -A executables=(
  [phm_detectors]="phm_detectors_node"
  [phm_ood]="phm_ood_node"
  [phm_ood_cpp]="ood_node bench_latency"
  [phm_arbiter]="phm_arbiter"
  [phm_recovery]="recovery_node"
  [phm_sim]="embedding_publisher phm_chain_bench"
  [phm_go2]="phm_go2_probe phm_go2_summarize phm_go2_plot"
)
optional_executables=(phm_go2/phoenix_shadow_embedder phm_go2/phoenix_shadow_replay)
share_files=(
  phm_detectors/launch/detectors.launch.xml phm_detectors/config/detectors.yaml
  phm_ood/launch/phm_ood.launch.xml phm_ood/config/phm_ood.yaml
  phm_arbiter/launch/arbiter.launch.xml phm_arbiter/config/phm_arbiter.yaml
  phm_recovery/launch/recovery.launch.xml phm_recovery/config/recovery.yaml
  phm_sim/launch/sim.launch.xml phm_sim/config/phm_sim.yaml
  phm_go2/phm_go2_detectors.yaml phm_go2/onboard_session.sh phm_go2/preflight.sh
)

check_exe() {
  local exe="$1"
  if [ ! -x "$exe" ]; then
    fail "$exe is missing or not executable"
    return
  fi
  local missing
  missing="$(ldd "$exe" 2>/dev/null | grep "not found" || true)"
  if [ -n "$missing" ]; then
    fail "$exe has unresolved libraries: $missing"
  else
    echo "exe OK: ${exe#*/lib/}"
  fi
}

for pkg in "${packages[@]}"; do
  if prefix="$(prefix_of "$pkg")"; then
    echo "ament OK: $pkg -> $prefix"
  else
    fail "$pkg not found in the ament index"
    continue
  fi
  for exe in ${executables[$pkg]:-}; do
    check_exe "$prefix/lib/$pkg/$exe"
  done
done
for item in "${optional_executables[@]}"; do
  pkg="${item%%/*}"
  prefix="$(prefix_of "$pkg" || true)"
  if [ -n "$prefix" ] && [ -x "$prefix/lib/$item" ]; then
    check_exe "$prefix/lib/$item"
  else
    echo "optional, not built: $item"
  fi
done
for item in "${share_files[@]}"; do
  prefix="$(prefix_of "${item%%/*}" || true)"
  if [ -z "$prefix" ] || [ ! -f "$prefix/share/$item" ]; then
    fail "share/$item not installed"
  fi
done

# No project Python in the overlay: only rosidl's generated phm_msgs bindings.
for pkg in "${packages[@]}"; do
  prefix="$(prefix_of "$pkg" || true)"
  [ -n "$prefix" ] || continue
  while IFS= read -r f; do
    case "$f" in
      */phm_msgs/*) ;;
      *) fail "Python file installed: $f" ;;
    esac
  done < <(find "$prefix/lib" "$prefix/local" -path "*python3*" -name "*.py" \
    -path "*phm_*" 2>/dev/null || true)
done

# phm_core's exported CMake package builds a consumer.
if command -v cmake >/dev/null; then
  if ! "$(dirname "${BASH_SOURCE[0]}")/check_core_consumer.sh"; then
    fail "find_package(phm_core) consumer did not build or run"
  fi
fi

if [ "$failures" -gt 0 ]; then
  echo "Install overlay check FAILED ($failures)" >&2
  exit 1
fi
echo "Install overlay check PASSED"
