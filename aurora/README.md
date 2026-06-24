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

## Verified results

Aurora Sapphire Rapids node, frameworks/2025.3.1, mace-torch
@e4d0a4e35, single Python process, 1 node:

| test | atoms | tiles | result |
|---|---|---|---|
| 1 import | -- | -- | `torch.xpu.device_count() == 12` |
| 2 cpu baseline | 1024 | cpu | E=-5054.78 eV, F_max=0.065 eV/Å |
| 3 two-tile | 1024 | 2 | vs cpu: \|dE\|=4.7e-2 eV, \|dF\|=1.4e-6 |
| 4 four-tile | 5488 | 4 | vs 2-tile: \|dE\|=1.4e-2 eV, \|dF\|=5.1e-7 |
| 5 12-tile | **148,176** | 12 | E=-732,271 eV, forward 27.2 s |

Energy drift across partition counts is fp32 reduction noise
(~1e-5 eV/atom across partition boundaries). Forces agree to ≤2e-6 eV/Å.

## Cell size requirement (DistMLIP C kernel)

The partition algorithm enforces `wall_width > 2 × atom_cutoff` per
partition. For MACE small (cutoff=6 Å, 2×cutoff=12 Å):

| n_partitions | min cell side |
|---|---|
| 2 | 24 Å (~7³ Li-Mn) |
| 4 | 48 Å (~14³) |
| 12 | 144 Å (~42³ = 148k atoms) |

Smaller cells trigger `RuntimeError: Partition walls are too close`.

## Benchmark (1/2/6/12 tiles, MACE small, Li-Mn perovskite)

### Strong scaling — 148k atoms fixed

| tiles | min wall (s) | speedup vs 6t | efficiency |
|---|---|---|---|
| 6  | 7.55 | 1.00× | 100% |
| 12 | 7.28 | 1.04× | **52%** |

≤ 4 tiles OOM at this cell (per-tile memory > 64 GB).

### Weak scaling — ~12k atoms/tile

| tiles | atoms | time (s) | atoms/s | atoms/s/tile |
|---|---|---|---|---|
| 1  | 13,718 | 1.04 | 13,228 | **13,228** |
| 2  | 27,648 | 1.64 | 16,885 | 8,442 |
| 6  | 85,750 | 4.45 | 19,264 | 3,211 |
| 12 | 148,176 | 7.36 | 20,132 | 1,678 |

Per-tile throughput collapses 8× from 1→12 tiles. DistMLIP is
**communication-bound on XPU**; `Distributed.aggregate` cross-tile
copies (Level Zero, likely host-bounced) dominate over MACE compute.

**Use DistMLIP when**: single-tile OOMs (problem doesn't fit 64 GB).
**Use plain MACECalculator(device="xpu") when**: it fits.

`ZE_ENABLE_PEER_ACCESS=1` was tested and gave no measurable improvement
(<2% noise). The bottleneck is in DistMLIP's aggregate path (PyTorch
allocator memcpy), not raw L0 P2P. Real fix would need oneCCL
collectives + compute/comm overlap — out of scope for this port.

## Known caveats

- `mace-torch` is pinned to `0.3.16` (Auto-Finetuner's tested Aurora pin), not
  DistMLIP's upstream commit `e4d0a4e35`. If forward outputs diverge from the
  CUDA reference, retest against the upstream pin with Aurora patches.
- `cuEquivariance` paths in upstream MACE are disabled on XPU (no fallback
  needed — MACE detects and skips).
- Cross-tile transfer in `Distributed.aggregate` goes through Level Zero
  copies (may bounce via host if P2P unavailable). Bench test 4 to confirm.
