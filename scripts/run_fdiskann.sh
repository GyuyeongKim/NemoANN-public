#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"

commit=${DISKANN_COMMIT:-f198a8a56fa793c8a7fba11a0db1e41227d7096f}
src=$ROOT/baselines/DiskANN
if [ ! -x "$src/build/apps/search_disk_index" ]; then
  [ -d "$src" ] || git clone https://github.com/microsoft/DiskANN.git "$src"
  git -C "$src" checkout -q "$commit"
  git -C "$src" apply "$ROOT/patches/diskann-label-parse.patch"
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$src/build" -j"$NPROC" --target build_disk_index search_disk_index
fi

table=$RESULTS/fdiskann.tsv
logs=$RESULTS/logs/fdiskann
new_table "$table"
mkdir -p "$logs"

for ds in ${DATASETS:-bigann100M deep100M}; do
  lp=$(label_prefix "$ds" 10)
  require "$(base_file "$ds")" "${lp}_base.bin" "${lp}_query.bin" "${lp}_gt.bin"
  for side in base query; do
    [ -f "${lp}_${side}.txt" ] || python3 "$ROOT/scripts/convert_labels.py" "${lp}_${side}.bin" "${lp}_${side}.txt"
  done

  prefix=$DATA/$ds/index/${ds}_filtered
  case $ds in
    deep*) lbuild=128 ;;
    *) lbuild=100 ;;
  esac
  if [ ! -f "${prefix}_disk.index" ]; then
    log "building the FilteredVamana index for $ds"
    "$src/build/apps/build_disk_index" --data_type "$(dtype "$ds")" --dist_fn l2 --data_path "$(base_file "$ds")" \
      --index_path_prefix "$prefix" -R 96 --FilteredLbuild "$lbuild" -B 4 -M 80 -T "$NPROC" --PQ_disk_bytes 0 \
      --label_file "${lp}_base.txt"
  fi

  for t in ${THREADS:-1 32}; do
    out=$logs/${ds}_sel10_fdiskann_T${t}.log
    log "fdiskann $ds T=$t"
    "$src/build/apps/search_disk_index" --data_type "$(dtype "$ds")" --dist_fn l2 --index_path_prefix "$prefix" \
      --query_file "$(query_file "$ds")" --gt_file "${lp}_gt.bin" --query_filters_file "${lp}_query.txt" \
      -K 10 -L $LS_100M -W 8 -T "$t" --result_path "$logs/${ds}_T${t}_res" > "$out" 2>&1 || die "search failed, see $out"
    awk -v ds="$ds" -v t="$t" '$1 ~ /^[0-9]+$/ && NF >= 9 && $2 == 8 {
      printf "fdiskann\t%s\t10\t%s\t8\t-\t-\t%s\t%s\t%.1f\t%s\t%s\n", ds, t, $1, $3, t * 1e6 / $3, $NF / 100, $6 }' \
      "$out" >> "$table"
  done
done
log "results in $table"
