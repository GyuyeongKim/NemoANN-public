#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"

table=$RESULTS/billion.tsv
for t in ${THREADS:-1 32}; do
  run_disk "$table" bigann1B 10 diskann "$t" 8 0 0 $LS_1B
  run_disk "$table" bigann1B 10 pipeann "$t" 32 10 0 $LS_1B
  run_disk "$table" bigann1B 10 nemoann "$t" 32 10 32 $LS_1B
done
log "results in $table"
