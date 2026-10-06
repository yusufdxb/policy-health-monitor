#!/usr/bin/env bash
# Re-run the reliability benchmark and compare every metric and CI column of
# results_<mode>.csv with the committed files (latency is host-dependent and
# not compared). The gradient-trained RND row depends on the SIMD kernels of
# the matrix products, so a difference there is reported as a warning only.
#   scripts/check_benchmark_reproduces.sh <path/to/phm_benchmark>
set -euo pipefail

exe="${1:?usage: check_benchmark_reproduces.sh <phm_benchmark>}"
root="$(git rev-parse --show-toplevel)"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$exe" --out-dir "$out" --latency-repeats 1 --mlp-latency-repeats 1 >/dev/null

status=0
for mode in collapse shift; do
  # Strip the latency column; compare row by row.
  while IFS= read -r line; do
    name="${line%%,*}"
    case "$line" in
      *"gradient-trained"*) severity=warning ;;
      *) severity=error ;;
    esac
    if ! grep -qxF -- "$line" <(cut -d, -f1-10 "$root/benchmark/results_$mode.csv" |
      sed 's/"//g' | tr -d '\r'); then
      echo "$severity: $mode row $name differs from the committed results" >&2
      [ "$severity" = error ] && status=1
    fi
  done < <(cut -d, -f1-10 "$out/results_$mode.csv" | sed 's/"//g' | tr -d '\r')
done
[ "$status" -eq 0 ] && echo "Benchmark metrics reproduce the committed results"
exit "$status"
