#!/bin/bash
# Source BEFORE any DistMLIP command on Aurora.
# Path A deployment: single Python process holds all 12 PVC tiles.
# DO NOT set ZE_AFFINITY_MASK (would hide tiles from torch.xpu).

source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load frameworks/2025.3.1 2>/dev/null

VENV=/home/afaiyad/QuantumDS/afaiyad/venv_distmlip_aurora
if [ -d "${VENV}" ]; then
    source "${VENV}/bin/activate"
fi

# XPU runtime
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export MPICH_GPU_SUPPORT_ENABLED=1

# C ext threading (find_points_in_spheres + subgraph_creation_fast OpenMP)
export DISTMLIP_NUM_THREADS="${DISTMLIP_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_STACKSIZE="${OMP_STACKSIZE:-1G}"

ulimit -c 0
