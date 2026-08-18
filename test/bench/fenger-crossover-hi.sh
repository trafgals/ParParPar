#!/usr/bin/env bash
# Crossover calibration: find smallest R where Fenger beats Barycentric.
# N=16384, 64KiB blocks, R = 512,1024,2048,4096. Each run capped at 300s.
set -u
cd "$(dirname "$0")/../.."
declare -A RES
for R in 512 1024 2048 4096; do
  for MODE in 1 0; do
    NAME=$([ "$MODE" = "1" ] && echo fenger || echo bary)
    OUT=$(PAR3_GF64_USE_FENGER=$MODE PAR3_PROFILE=1 timeout 300 node test/bench/par3-create-bench.js \
      --size=1G --slices=16384 --block-size=65536 --recovery=$R 2>&1)
    MB=$(echo "$OUT" | grep -oE "Throughput: [0-9.]+" | grep -oE "[0-9.]+")
    if [ -z "$MB" ]; then MB="TMO"; fi
    echo "R=$R $NAME: $MB"
    RES[$R,$NAME]=$MB
  done
done
echo "=== SUMMARY ==="
for R in 512 1024 2048 4096; do
  echo "R=$R fenger=${RES[$R,fenger]} bary=${RES[$R,bary]}"
done
