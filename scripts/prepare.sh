#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"

ds=$1
case $ds in
  bigann100M)
    url=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann
    n=100000000 dim=128 elem=1 R=96 L=128 sels="5 10 20" rng=pcg
    ;;
  deep100M)
    url=https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP
    n=100000000 dim=96 elem=4 R=96 L=128 sels="10" rng=pcg
    ;;
  bigann1B)
    url=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann
    n=1000000000 dim=128 elem=1 R=128 L=200 sels="10" rng=legacy
    ;;
  *)
    die "usage: $0 bigann100M|deep100M|bigann1B"
    ;;
esac

base=$(base_file "$ds")
query=$(query_file "$ds")
prefix=$(index_prefix "$ds")
ext=$(vec_ext "$ds")
type=$(dtype "$ds")
mkdir -p "$DATA/$ds/index" "$DATA/$ds/labels"

if [ ! -f "$query" ]; then
  log "downloading $ds queries"
  curl -fL --retry 5 -o "$query.part" "$url/query.public.10K.$ext"
  mv "$query.part" "$query"
fi

if [ ! -f "$base" ]; then
  bytes=$((n * dim * elem))
  log "downloading the first $n vectors of $url/base.1B.$ext"
  python3 -c "import struct, sys; sys.stdout.buffer.write(struct.pack('II', $n, $dim))" > "$base.part"
  curl -fL --retry 5 -r "8-$((bytes + 7))" "$url/base.1B.$ext" >> "$base.part"
  [ "$(stat -c %s "$base.part")" = "$((bytes + 8))" ] || die "incomplete download: $base.part"
  mv "$base.part" "$base"
fi

if [ ! -f "${prefix}_disk.index" ]; then
  log "building the disk index (R=$R, L=$L, 32-byte PQ)"
  "$BIN/build_disk_index" "$type" "$base" "$prefix" "$R" "$L" 32 "${NEMO_BUILD_MEM:-200}" "$NPROC" l2 pq
fi

if [ ! -f "${prefix}_mem.index" ]; then
  log "building the in-memory navigation graph"
  "$BIN/utils/gen_random_slice" "$type" "$base" "${prefix}_sample" 0.01
  "$BIN/build_memory_index" "$type" "${prefix}_sample_data.bin" "${prefix}_sample_ids.bin" "${prefix}_mem.index" \
    32 64 1.2 "$NPROC" l2
  rm -f "${prefix}_sample_data.bin" "${prefix}_sample_ids.bin"
fi

for sel in $sels; do
  lp=$(label_prefix "$ds" "$sel")
  if [ ! -f "${lp}_gt.bin" ]; then
    log "generating labels and ground truth for $sel% selectivity"
    python3 "$ROOT/scripts/make_labels.py" "$base" "$query" "$lp" --classes $((100 / sel)) --rng "$rng"
  fi
done
log "$ds ready"
