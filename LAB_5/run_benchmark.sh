#!/bin/bash
set -euo pipefail

SIZES=${SIZES:-200,400,800,1200,1600,2000}
REPEATS=${REPEATS:-3}
PROCESSES=${PROCESSES:-"1 2 4 8 16"}
MPIRUN=${MPIRUN:-mpirun}
BINARY=${BINARY:-./matrix_mpi_cluster}
OUTPUT=${OUTPUT:-measured_results.csv}

rm -f "$OUTPUT"

for processes in $PROCESSES; do
    echo "Running N=$SIZES, processes=$processes, repeats=$REPEATS"
    $MPIRUN -np "$processes" "$BINARY" \
        --sizes "$SIZES" \
        --repeats "$REPEATS" \
        --output "$OUTPUT" \
        --append
done

if command -v python3 >/dev/null 2>&1; then
    python3 plot_results.py --input "$OUTPUT" --prefix measured || echo "Plotting skipped: Python packages are not available"
fi
