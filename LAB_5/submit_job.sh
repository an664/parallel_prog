#!/bin/bash
#SBATCH --job-name=parallel_lab5
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=lab5_%j.out

set -euo pipefail

if [ -f /soft/intel/parallel_studio_xe_2016.3.067/bin/psxevars.sh ]; then
    source /soft/intel/parallel_studio_xe_2016.3.067/bin/psxevars.sh intel64
fi

make clean
make MPICXX="${MPICXX:-mpiicpc}"

export SIZES=${SIZES:-200,400,800,1200,1600,2000}
export REPEATS=${REPEATS:-3}
export PROCESSES=${PROCESSES:-"1 2 4 8 16"}
export OUTPUT=${OUTPUT:-measured_results.csv}
export MPIRUN=${MPIRUN:-mpirun}

./run_benchmark.sh
