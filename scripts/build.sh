#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"
cd "$ROOT"

if [ ! -x third_party/liburing/configure ]; then
  if [ -d .git ]; then
    git submodule update --init third_party/liburing
  else
    git clone --depth 1 -b liburing-2.10 https://github.com/axboe/liburing.git third_party/liburing
  fi
fi
if [ ! -f third_party/liburing/src/liburing.a ]; then
  (cd third_party/liburing && ./configure && make -C src -j"$NPROC")
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$NPROC"
