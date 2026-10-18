#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"

table=$RESULTS/selectivity.tsv
for sel in ${SELS:-5 10 20}; do
  for t in ${THREADS:-1 32}; do
    run_disk "$table" bigann100M "$sel" pipeann "$t" 32 10 0 $LS_100M
    run_disk "$table" bigann100M "$sel" nemoann "$t" 32 10 32 $LS_100M
  done
done
log "results in $table"
