#!/bin/bash -l
#PBS -N distmlip-smoke
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:25:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A AISim4NS
#PBS -o smoke.out
#PBS -j oe

cd "${PBS_O_WORKDIR}"
source aurora/env_aurora.sh

echo "==== TEST 1: imports + tile visibility"
python aurora/tests/01_import.py
RC=$?
echo "EXIT_1=${RC}"
[ ${RC} -ne 0 ] && exit ${RC}

echo
echo "==== TEST 2: cpu baseline"
python aurora/tests/02_cpu_baseline.py
RC=$?
echo "EXIT_2=${RC}"
[ ${RC} -ne 0 ] && exit ${RC}

echo
echo "==== TEST 3: xpu two tile (min partitions = 2)"
python aurora/tests/03_xpu_two_tile.py
RC=$?
echo "EXIT_3=${RC}"
[ ${RC} -ne 0 ] && exit ${RC}

echo
echo "==== TEST 4: xpu four tile"
python aurora/tests/04_xpu_four_tile.py
RC=$?
echo "EXIT_4=${RC}"
exit ${RC}
