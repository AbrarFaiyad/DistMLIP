#!/bin/bash -l
#PBS -N distmlip-build
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A AISim4NS
#PBS -o build.out
#PBS -j oe

cd "${PBS_O_WORKDIR}"
source aurora/env_aurora.sh

REPO=/home/afaiyad/QuantumDS/afaiyad/DistMLIP
cd "${REPO}"

echo "==> compute node CPU"
grep "model name" /proc/cpuinfo | head -1

echo "==> building C extension (-march=native -fopenmp)"
pip install --no-build-isolation --no-deps -e . 2>&1

echo "==> import check"
python -c "
import torch
print('torch.xpu.is_available:', torch.xpu.is_available())
print('torch.xpu.device_count:', torch.xpu.device_count())
from DistMLIP.distributed.subgraph_creation_fast import get_subgraphs_fast
print('subgraph_creation_fast OK')
from DistMLIP.implementations.mace import MACECalculator_Dist
from DistMLIP.implementations.mace.models import ScaleShiftMACE_Dist
print('MACECalculator_Dist + ScaleShiftMACE_Dist import OK')
"
echo "EXIT=$?"
