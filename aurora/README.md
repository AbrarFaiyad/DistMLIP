# DistMLIP on ALCF Aurora — MACE Path

Port of DistMLIP's MACE wrapper (`MACECalculator_Dist`, `ScaleShiftMACE_Dist`)
to Intel PVC GPUs via `torch.xpu`. Single Python process holds all 12 tiles
on one node (Path A: intra-process multi-device).

## What's in scope

- `DistMLIP/implementations/mace/` — full path patched
- `DistMLIP/distributed/` — C extension + `Distributed` class (device-string-driven, no changes needed)

## What's NOT in scope

- `DistMLIP/implementations/uma/` — fairchem's `RotMatWignerCudaGraph` is CUDA-only
- `DistMLIP/implementations/matgl/` (CHGNet, TensorNet) — depends on DGL, which has no XPU backend

## Install (one-time, login node)

```bash
ssh aurora.alcf.anl.gov
cd /home/afaiyad/QuantumDS/afaiyad
git clone -b aurora-xpu git@github.com:AbrarFaiyad/DistMLIP.git
cd DistMLIP
bash aurora/build_env_aurora.sh
```

This:
1. loads `frameworks/2025.3.1`
2. creates `/home/afaiyad/QuantumDS/afaiyad/venv_distmlip_aurora` (`--system-site-packages`)
3. pip installs leaf deps + `mace-torch==0.3.16` + `e3nn==0.4.4` (no-deps)
4. applies Aurora patches via `aurora/patch_distmlip_for_aurora.py`

C extension build runs separately on a compute node so `-march=native` resolves to Sapphire Rapids:

```bash
qsub aurora/jobs/build_c_ext.sh
# wait, then:
cat build.out
```

## Smoke tests

```bash
qsub aurora/jobs/smoke_xpu.sh      # tests 1-4 (~25 min)
qsub aurora/jobs/smoke_12tile.sh   # test 5: full node (~45 min)
```

Test 1 verifies imports + 12 tiles visible. Test 2 saves CPU baseline. Test 3
runs the **minimum supported partition count = 2** (DistMLIP's C kernel rejects
n_partitions=1, see `subgraph_creation_utils.c:49`). Tests 4 + 5 scale to 4
and 12 tiles.

## Usage

```python
from mace.calculators import mace_mp
from DistMLIP.implementations.mace import MACECalculator_Dist

# Load MACE on CPU (avoid initial xpu allocation)
calc = mace_mp(model="small", device="cpu")

# Wrap, distribute across all 12 PVC tiles
dist_calc = MACECalculator_Dist.from_existing(calc)
dist_calc.enable_distributed_mode(list(range(12)))

atoms.calc = dist_calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

The patched `enable_distributed_mode` auto-detects `torch.xpu` and builds
`xpu:N` device strings instead of `cuda:N`.

## Runtime env essentials (`aurora/env_aurora.sh`)

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT      # 12 flat IDs 0..11
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export MPICH_GPU_SUPPORT_ENABLED=1
# DO NOT set ZE_AFFINITY_MASK — Path A needs all tiles visible
```

## Patches applied

| File | Change |
|---|---|
| `DistMLIP/implementations/mace/models.py` | `enable_distributed_mode` builds `xpu:N` when `torch.xpu.is_available()`, else `cuda:N` |
| `DistMLIP/implementations/mace/mace.py` | bypass `torch_tools.init_device` for `xpu:N` (only accepts bare `"xpu"`); call `torch.xpu.set_device` explicitly |
| `DistMLIP/__init__.py` | fp16 guard accepts xpu accelerator |

All patches idempotent via `# AURORA-PATCHED` marker. Re-running
`aurora/patch_distmlip_for_aurora.py` is safe.

## Known caveats

- `mace-torch` is pinned to `0.3.16` (Auto-Finetuner's tested Aurora pin), not
  DistMLIP's upstream commit `e4d0a4e35`. If forward outputs diverge from the
  CUDA reference, retest against the upstream pin with Aurora patches.
- `cuEquivariance` paths in upstream MACE are disabled on XPU (no fallback
  needed — MACE detects and skips).
- Cross-tile transfer in `Distributed.aggregate` goes through Level Zero
  copies (may bounce via host if P2P unavailable). Bench test 4 to confirm.
