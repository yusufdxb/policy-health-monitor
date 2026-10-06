#!/usr/bin/env bash
# Reject tracked text that exposes machine-local or private build details:
# user-home paths, home-directory shorthand, private hardware identifiers and
# the U+2014 em dash, in tracked and untracked (not ignored) files. Binary
# files are skipped. Run from anywhere in the repo.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

# Patterns are assembled from pieces so this file does not match itself.
home_unix="/""home/[^/[:space:]]+/"
home_mac="/""Users/[^/[:space:]]+/"
home_win="[A-Za-z]:\\\\""Users\\\\[^\\\\[:space:]]+\\\\"
home_short="~""/"
hardware="Jet""son[[:space:]]+Orin|(GeForce[[:space:]]+)?R""TX[[:space:]]*[0-9]+"
hardware+="|A""MD[[:space:]]+Ryzen[[:space:]]+[0-9]+"
hardware+="|In""tel[[:space:]]+Core([[:space:]]+i[3579])?(-|[[:space:]]+)[0-9]+"
hardware+="|App""le[[:space:]]+M[0-9]+|[0-9]+[[:space:]]*(GB|GiB)[[:space:]]+(VRAM|GPU[[:space:]]+memory)"
emdash="$(printf '\xe2\x80\x94')"

labels=("Unix user-home path" "macOS user-home path" "Windows user-home path"
  "home-directory shorthand" "private hardware identifier" "U+2014 em dash")
patterns=("$home_unix" "$home_mac" "$home_win" "$home_short" "$hardware" "$emdash")
flags=("-E" "-E" "-E" "-F" "-Ei" "-F")

findings=0
while IFS= read -r -d '' file; do
  [ -f "$file" ] || continue
  for i in "${!patterns[@]}"; do
    # -I skips binary files; -n prints line numbers.
    while IFS= read -r hit; do
      echo "  $file:${hit%%:*}: ${labels[$i]}" >&2
      findings=$((findings + 1))
    done < <(grep -I -n ${flags[$i]} -- "${patterns[$i]}" "$file" || true)
  done
done < <(git ls-files -z --cached --others --exclude-standard)

if [ "$findings" -gt 0 ]; then
  echo "Public repository hygiene check failed ($findings findings)" >&2
  exit 1
fi
echo "Public repository hygiene check passed"
