#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"

commit=${PIPEANN_COMMIT:-6c7d9dae21bc0590aa570c8d2998f34abf37f61c}
src=$ROOT/baselines/PipeANN
if [ ! -x "$src/build/tests/search_disk_index_filtered" ]; then
  [ -d "$src" ] || git clone https://github.com/thustorage/PipeANN.git "$src"
  git -C "$src" checkout -q "$commit"
  (cd "$src/third_party/liburing" && ./configure && make -C src -j"$NPROC")
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$src/build" -j"$NPROC" --target build_disk_index_filtered search_disk_index_filtered
fi

table=$RESULTS/pipeann_filter.tsv
logs=$RESULTS/logs/pipeann_filter
new_table "$table"
mkdir -p "$logs"

for ds in ${DATASETS:-bigann100M deep100M}; do
  sels=10
  [ "$ds" != bigann100M ] || sels=${SELS:-5 10 20}
  for sel in $sels; do
    lp=$(label_prefix "$ds" "$sel")
    require "$(base_file "$ds")" "${lp}_base.bin" "${lp}_query.bin" "${lp}_gt.bin"
    for side in base query; do
      [ -f "${lp}_${side}.spmat" ] || python3 "$ROOT/scripts/convert_labels.py" "${lp}_${side}.bin" "${lp}_${side}.spmat"
    done

    prefix=$DATA/$ds/index/${ds}_pf_sel${sel}
    if [ ! -f "${prefix}_disk.index" ]; then
      log "building the PipeANN-Filter index for $ds at $sel% selectivity"
      "$src/build/tests/build_disk_index_filtered" "$(dtype "$ds")" "$(base_file "$ds")" "$prefix" 96 1100 128 32 \
        "${NEMO_BUILD_MEM:-200}" "$NPROC" l2 pq label_spmat "${lp}_base.spmat"
    fi
    printf '{"attr_indexes": [{"name": "tags", "key": 0, "type": "label", "file": "%s.label.0"}], "filter": "array_contains_all(tags, $$query_tags)", "bindings": {"query_tags": "%s"}}\n' \
      "$prefix" "${lp}_query.spmat" > "$prefix.json"

    for t in ${THREADS:-1 32}; do
      for L in $LS_100M; do
        out=$logs/${ds}_sel${sel}_pf_T${t}_L${L}.log
        log "pipeann-filter $ds sel=$sel T=$t L=$L"
        "$src/build/tests/search_disk_index_filtered" "$(dtype "$ds")" "$prefix" "$t" 32 "$(query_file "$ds")" \
          "${lp}_gt.bin" 10 l2 pq "$prefix.json" 0 "$L" > "$out" 2>&1 || die "search failed, see $out"
        awk -v ds="$ds" -v sel="$sel" -v t="$t" '$1 ~ /^[0-9]+$/ && NF >= 20 {
          printf "pipeann-filter\t%s\t%s\t%s\t32\t0\t-\t%s\t%s\t%.1f\t%s\t%s\n", ds, sel, t, $1, $3, t * 1e6 / $3, $NF / 100, $(NF - 1) }' \
          "$out" >> "$table"
      done
    done
  done
done
log "results in $table"
