#!/usr/bin/env bash
# Fail if the repository carries project Python: Python sources or notebooks,
# Python packaging or test configuration, ament_python packages, Python
# shebangs, or scripts / CI / CMake that invoke a Python interpreter. ROS 2's
# own tooling (ros2 launch, colcon, rosidl-generated bindings) is upstream and
# lives outside the tracked tree. ALLOWED lists justified exceptions (none).
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"
ALLOWED=()

is_allowed() {
  local f
  for f in "${ALLOWED[@]}"; do
    [ "$f" = "$1" ] && return 0
  done
  return 1
}

problems=0
report() {
  echo "  $1" >&2
  problems=$((problems + 1))
}

while IFS= read -r -d '' file; do
  [ -f "$file" ] || continue
  is_allowed "$file" && continue
  case "$file" in
    *.py | *.pyi | *.pyx | *.ipynb) report "$file: Python source" ;;
    */setup.py | setup.py | */setup.cfg | setup.cfg | */pyproject.toml | pyproject.toml | \
      */pytest.ini | pytest.ini | */conftest.py | conftest.py | */tox.ini | tox.ini | \
      */requirements*.txt | requirements*.txt | */.flake8 | .flake8)
      report "$file: Python packaging or test configuration" ;;
  esac
  if head -c 64 "$file" | grep -aqE '^#!.*python'; then
    report "$file: Python shebang"
  fi
  case "$file" in
    */package.xml | package.xml)
      if grep -q "<build_type>ament_python</build_type>" "$file"; then
        report "$file: ament_python package"
      fi ;;
    *.sh | *.bash | *.yml | *.yaml | */CMakeLists.txt | CMakeLists.txt | *.cmake | *.cmake.in)
      # The patterns are split so this script does not match itself.
      while IFS= read -r hit; do
        report "$file:${hit%%:*}: Python interpreter invocation"
      done < <(grep -nE '(^|[[:space:]"/=(])py''thon3?([[:space:]]|$)|pip''3? install' "$file" \
        || true) ;;
  esac
done < <(git ls-files -z --cached --others --exclude-standard)

if [ "$problems" -gt 0 ]; then
  echo "Project Python found ($problems findings)" >&2
  exit 1
fi
echo "No project Python: check passed"
