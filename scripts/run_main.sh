#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"

table=$RESULTS/main.tsv
for ds in ${DATASETS:-bigann100M deep100M}; do
  for t in ${THREADS:-1 32}; do
    run_disk "$table" "$ds" 10 diskann "$t" 8 0 0 $LS_100M
    run_disk "$table" "$ds" 10 pipeann "$t" 32 10 0 $LS_100M
    run_disk "$table" "$ds" 10 nemoann "$t" 32 10 32 $LS_100M
  done
done
log "results in $table"
