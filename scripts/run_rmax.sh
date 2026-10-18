#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"

table=$RESULTS/rmax.tsv
for r in ${RMAX:-4 8 16 32 48 64}; do
  for t in ${THREADS:-1 32}; do
    run_disk "$table" bigann100M 10 nemoann "$t" 32 10 "$r" $LS_100M
  done
done
log "results in $table"
