#!/usr/bin/env bash
# Build and run a minimal CMake project that uses the installed phm_core
# through find_package(phm_core) and links phm_core::phm_core. Works against a
# plain CMake install (pass its prefix) or a sourced colcon overlay (no args).
#   scripts/check_core_consumer.sh [install_prefix]
set -euo pipefail

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cat >"$work/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.16)
project(phm_core_consumer CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(phm_core REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer phm_core::phm_core)
CMAKE
cat >"$work/main.cpp" <<'CPP'
#include <cstdio>
#include "phm_core/ood_core.hpp"
int main()
{
  phm_core::OodConfig c;
  c.window = 2;
  c.threshold = 0.5;
  c.min_consecutive = 1;
  phm_core::OodCore core(c, phm_core::make_plain_backend());
  const float f[2] = {1.0f, 2.0f};
  core.update(f, 2, "");
  std::printf("%s\n", core.update(f, 2, "").violating ? "violating" : "healthy");
  return 0;
}
CPP
args=()
if [ $# -ge 1 ]; then
  args+=("-DCMAKE_PREFIX_PATH=$1")
  export LD_LIBRARY_PATH="$1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if cmake -S "$work" -B "$work/build" "${args[@]}" >"$work/log" 2>&1 &&
  cmake --build "$work/build" >>"$work/log" 2>&1 &&
  [ "$("$work/build/consumer")" = "violating" ]; then
  echo "cmake OK: find_package(phm_core) consumer builds and runs"
else
  cat "$work/log" >&2
  echo "find_package(phm_core) consumer FAILED" >&2
  exit 1
fi
