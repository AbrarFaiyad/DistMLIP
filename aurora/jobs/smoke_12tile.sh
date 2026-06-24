#!/bin/bash -l
#PBS -N distmlip-12tile
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A AISim4NS
#PBS -o test_12tile.out
#PBS -j oe

cd "${PBS_O_WORKDIR}"
source aurora/env_aurora.sh

python aurora/tests/05_xpu_12tile.py
echo "EXIT=$?"
