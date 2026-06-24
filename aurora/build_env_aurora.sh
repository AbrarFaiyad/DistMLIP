#!/bin/bash
# Build venv for DistMLIP on Aurora. Run ONCE on a login node.
#   bash aurora/build_env_aurora.sh
#
# Creates /home/afaiyad/QuantumDS/afaiyad/venv_distmlip_aurora on top of
# frameworks/2025.3.1 (torch 2.10 + native torch.xpu). C extension build
# is deferred to aurora/jobs/build_c_ext.sh (must run on a compute node
# so -march=native picks up Sapphire Rapids).

set -e

VENV=/home/afaiyad/QuantumDS/afaiyad/venv_distmlip_aurora
REPO=/home/afaiyad/QuantumDS/afaiyad/DistMLIP

source /usr/share/lmod/lmod/init/bash
module load frameworks/2025.3.1

echo "==> base torch + xpu check (login node may show xpu_available=False; that is fine)"
python -c "import torch, sys; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('has_xpu_attr', hasattr(torch, 'xpu'))"

if [ ! -d "${VENV}" ]; then
    echo "==> creating venv at ${VENV} (--system-site-packages)"
    python -m venv --system-site-packages "${VENV}"
else
    echo "==> venv exists at ${VENV}, reusing"
fi
source "${VENV}/bin/activate"

echo "==> pip-installing leaf deps"
pip install --no-cache-dir \
    ase pymatgen torchmetrics torch-ema matscipy opt-einsum-fx \
    prettytable python-hostlist configargparse h5py tqdm lmdb orjson

# mace-torch pin: 0.3.16 (matches Auto-Finetuner verified Aurora pin)
# NOTE: DistMLIP pyproject pins commit e4d0a4e35; smoke-test 0.3.16 first.
echo "==> installing mace-torch 0.3.16 + e3nn 0.4.4 (no deps)"
pip install --no-deps "mace-torch==0.3.16"
pip install --no-deps "e3nn==0.4.4"

echo "==> applying DistMLIP Aurora patches"
python "${REPO}/aurora/patch_distmlip_for_aurora.py" "${REPO}"

echo
echo "==> sanity (pure Python, C ext build separate)"
python -c "
import torch, DistMLIP
print('DistMLIP imported from', DistMLIP.__file__)
print('torch', torch.__version__, 'has_xpu', hasattr(torch, 'xpu'))
"

echo
echo "Done. Next: qsub aurora/jobs/build_c_ext.sh"
