#!/bin/bash -l
#PBS -N distmlip-bench
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:50:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A AISim4NS
#PBS -o bench.out
#PBS -j oe

cd "${PBS_O_WORKDIR}"
source aurora/env_aurora.sh

python aurora/tests/06_benchmark.py
echo "EXIT=$?"
