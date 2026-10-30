#!/usr/bin/env bash
# Run every kernel at every supported size and print CSV: kernel,size,gflops
set -euo pipefail

KERNELS=${KERNELS:-"1 2 3 4 5 6 7 99"}
SIZES=${SIZES:-"1024 2048 4096 8192"}

echo "kernel,size,gflops"
for s in $SIZES; do
    for k in $KERNELS; do
        ./runner "$k" "$s" | awk -v k="$k" -v s="$s" '/GFLOPS/ {print k "," s "," $2}'
    done
done
