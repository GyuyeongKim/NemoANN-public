#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"

ds=bigann100M
lp=$(label_prefix $ds 10)
table=$RESULTS/vamana.tsv
logs=$RESULTS/logs/vamana
require "$(index_prefix $ds)_disk.index" "${lp}_base.bin" "${lp}_query.bin" "${lp}_gt.bin"
new_table "$table"
mkdir -p "$logs"

for t in ${THREADS:-1 32}; do
  out=$logs/${ds}_sel10_vamana_T${t}.log
  log "vamana $ds T=$t"
  "$BIN/search_mem_index" "$(dtype $ds)" "$(index_prefix $ds)" "$t" "$(query_file $ds)" \
    "${lp}_base.bin" "${lp}_query.bin" "${lp}_gt.bin" 10 $LS_100M > "$out" 2>&1 || die "search failed, see $out"
  awk -v t="$t" '$1 ~ /^[0-9]+$/ && NF == 4 { printf "vamana\tbigann100M\t10\t%s\t-\t-\t-\t%s\t%s\t%s\t%s\t0\n", t, $1, $2, $3, $4 }' \
    "$out" >> "$table"
done
log "results in $table"
