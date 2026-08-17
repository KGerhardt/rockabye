#!/usr/bin/env bash
# Run compare_to_rockout.py across every ROCkOut project that cached its read
# tables, smallest first, and summarise the headline numbers.
#
# Usage: tests/compare_all.sh <root_dir> [max_rows]
set -uo pipefail

ROOT="${1:?usage: compare_all.sh <root_dir> [max_rows]}"
MAX_ROWS="${2:-999999999}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${TMPDIR:-/tmp}/rockabye_compare"
mkdir -p "$OUT"

mapfile -t PROJECTS < <(
  find "$ROOT" -name "bitscore_vs_MA_pos.txt" 2>/dev/null \
    | sed 's|/final_outputs/model/bitscore_vs_MA_pos.txt||' \
    | while read -r p; do
        n=$(cat "$p"/final_outputs/reads/complete_reads_read_length_*.txt 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] && [ "$n" -le "$MAX_ROWS" ] && echo "$n|$p"
      done | sort -t'|' -k1 -n | cut -d'|' -f2
)

printf '%-50s %11s %8s %8s %8s %8s %8s\n' \
  PROJECT READS BIN% MAPOS% AGREE% F1_ABYE F1_ROCK
printf '%s\n' "-------------------------------------------------------------------------------------------------"

for p in "${PROJECTS[@]}"; do
  name="${p#"$ROOT"/}"
  log="$OUT/$(echo "$name" | tr '/' '_').log"
  if ! timeout 7200 python3 "$HERE/compare_to_rockout.py" "$p" > "$log" 2>&1; then
    printf '%-50s %11s %8s\n' "$name" "-" "ERROR"
    tail -3 "$log" | sed 's/^/      /'
    continue
  fi
  reads=$(grep -oP 'overall verdict agreement:\s+[\d.]+% of \K[\d,]+' "$log")
  agree=$(grep -oP 'overall verdict agreement:\s+\K[\d.]+' "$log")
  f1mine=$(grep -oP 'rockabye curves F1 \K[\d.]+' "$log")
  f1theirs=$(grep -oP 'ROCkOut  curves F1 \K[\d.]+' "$log")
  # Worst binning / MA-mapping agreement across this project's read lengths.
  binmin=$(grep -oP '(bitscore_bin|id_bin|aln_index) \K[\d.]+(?=%)' "$log" | sort -n | head -1)
  mapmin=$(grep -oP 'median MA position matches\s+\K[\d.]+' "$log" | sort -n | head -1)
  printf '%-50s %11s %7s%% %7s%% %7s%% %8s %8s\n' \
    "$name" "$reads" "$binmin" "$mapmin" "$agree" "$f1mine" "$f1theirs"
done

echo
echo "full logs in $OUT"
