"""Test 6: tile-scaling benchmark.

Compares forward time at:
  1 tile  -- plain MACECalculator(device="xpu") (DistMLIP requires n_part>=2)
  2 tile  -- DistMLIP min
  6 tile
  12 tile -- full node

Cell must satisfy wall_width > 2*cutoff per partition for the largest
configuration. For MACE small (cutoff=6 A, 2*cutoff=12 A) and 12 partitions,
cell side >= 144 A => 42^3 supercell of 3.5 A cubic Pm-3m Li-Mn (148,176 atoms).

Per config: 1 warmup forward (compile, allocator, neighbor cache) then N timed
forwards. Reports min/median/mean.
"""
import sys
import time

import numpy as np


N_WARMUP = 1
N_TIMED = 3
SUPERCELL = (42, 42, 42)


def build_cell():
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(3.5), ["Li", "Mn"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    struct.perturb(0.05, seed=42)
    struct.make_supercell(SUPERCELL)
    return AseAtomsAdaptor().get_atoms(struct)


def time_forward(atoms, calc, label):
    import torch

    atoms.calc = calc

    # Warmup
    for _ in range(N_WARMUP):
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        if hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()

    times = []
    energies = []
    for _ in range(N_TIMED):
        # Reset calc results to force recompute on every call
        calc.results = {}
        t0 = time.perf_counter()
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        if hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        energies.append(e)

    arr = np.array(times)
    print(f"[{label}] n={N_TIMED} min={arr.min():.3f} s median={np.median(arr):.3f} s "
          f"mean={arr.mean():.3f} s std={arr.std():.3f} s")
    print(f"[{label}] E_avg={np.mean(energies):.4f} eV")
    return arr.min()


def main():
    import torch
    from mace.calculators import mace_mp
    from DistMLIP.implementations.mace import MACECalculator_Dist

    n_tiles = torch.xpu.device_count()
    if n_tiles < 12:
        print(f"FAIL: need 12 tiles, got {n_tiles}")
        return 1

    print(f"Building {SUPERCELL} Li-Mn cell ...")
    atoms = build_cell()
    print(f"atoms: {len(atoms)}")
    print()

    results = {}

    # 1 tile baseline: plain MACECalculator on xpu (no DistMLIP wrapper)
    print("==== 1 tile: plain MACECalculator(device='xpu')")
    calc1 = mace_mp(model="small", device="xpu")
    results[1] = time_forward(atoms, calc1, "1tile")
    del calc1
    print()

    # DistMLIP configs share the same source calc
    base = mace_mp(model="small", device="cpu")

    for n in (2, 6, 12):
        print(f"==== {n} tile: DistMLIP")
        calc = MACECalculator_Dist.from_existing(base)
        calc.enable_distributed_mode(list(range(n)))
        results[n] = time_forward(atoms, calc, f"{n}tile")
        del calc
        print()

    print("==== Summary (min wall time per forward)")
    t1 = results[1]
    print(f"{'tiles':>5}  {'time_s':>8}  {'speedup_vs_1':>12}  {'efficiency':>10}")
    for n in (1, 2, 6, 12):
        t = results[n]
        spd = t1 / t if t > 0 else 0.0
        eff = spd / n
        print(f"{n:>5}  {t:>8.3f}  {spd:>12.2f}x  {eff:>9.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
