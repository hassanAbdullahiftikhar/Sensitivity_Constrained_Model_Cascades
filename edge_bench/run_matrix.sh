#!/usr/bin/env sh
# Full benchmark matrix on a Linux/Android(Termux) target. Usage:
#   sh run_matrix.sh <deploy_dir> <device_tag> "<threads list>"
#   sh run_matrix.sh deploy/pathmnist_128_seed0 pixel7 "1 2 4"
# Runs every cascade_*.json config in <deploy_dir>, for FP32 and INT8, at each thread count.
set -e
DEPLOY=${1:?deploy dir}
TAG=${2:?device tag}
THREADS=${3:-"1 2 4"}
EXTRA=${4:-}  # extra bench.py flags, e.g. "--no-spin"
HERE=$(dirname "$0")
OUT="$HERE/../results/bench/$TAG"
mkdir -p "$OUT"
for cfg in "$DEPLOY"/cascade_*.json; do
  name=$(basename "$cfg")
  case "$name" in
    cascade_int8naive_*) precisions="int8naive" ;;
    cascade_int8_*) precisions="int8" ;;
    *) precisions="fp32" ;;
  esac
  for p in $precisions; do
    for t in $THREADS; do
      python3 "$HERE/bench.py" --deploy "$DEPLOY" --precision "$p" --config "$name" --threads "$t" --tag "$TAG" --out "$OUT" $EXTRA
      sleep 20  # let the CPU cool between runs (thermal throttling)
    done
  done
done
