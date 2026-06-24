"""Test 6: tile-scaling benchmark.

Single-tile MACE forward on 148k atoms OOMs (1 PVC tile = 64 GB, MACE-small
forward needs ~17 GiB extra per ~50k atoms). Run TWO benchmarks:

  (A) FIXED CELL (148k atoms) across 2 / 6 / 12 tiles
      DistMLIP partitioning; reports strong-scaling.

  (B) PER-TILE THROUGHPUT
      1 tile  -- plain MACECalculator on small cell (~12k atoms, fits memory)
      2/6/12 tile -- DistMLIP, scaled cell so each tile sees same atoms
      Reports atoms/sec per tile (weak-scaling proxy).

Per config: 1 warmup forward (compile, allocator, neighbor cache) then N timed
forwards w/ torch.xpu.synchronize.
"""
import sys
import time

import numpy as np


N_WARMUP = 1
N_TIMED = 3
ATOMS_PER_TILE = 12000  # safe under 64 GB tile w/ MACE small


def build_cell(supercell):
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(3.5), ["Li", "Mn"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    struct.perturb(0.05, seed=42)
    struct.make_supercell(supercell)
    return AseAtomsAdaptor().get_atoms(struct)


def time_forward(atoms, calc, label):
    import torch

    atoms.calc = calc

    for _ in range(N_WARMUP):
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        if hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()

    times = []
    for _ in range(N_TIMED):
        calc.results = {}
        t0 = time.perf_counter()
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        if hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    print(f"[{label}] n={N_TIMED} min={arr.min():.3f}s med={np.median(arr):.3f}s "
          f"mean={arr.mean():.3f}s")
    return arr.min()


def run_dist(base, atoms, n_tiles, label):
    from DistMLIP.implementations.mace import MACECalculator_Dist
    calc = MACECalculator_Dist.from_existing(base)
    calc.enable_distributed_mode(list(range(n_tiles)))
    return time_forward(atoms, calc, label)


def benchmark_strong(base):
    """Strong scaling: fixed 148k atom cell across 2/6/12 tiles.
    Cell side 147 A satisfies 12-partition wall constraint."""
    print("=" * 60)
    print("(A) STRONG SCALING -- fixed 148k atom cell")
    print("=" * 60)
    atoms = build_cell((42, 42, 42))
    print(f"atoms: {len(atoms)}")
    print()

    results = {}
    for n in (2, 6, 12):
        print(f"---- {n} tiles")
        results[n] = run_dist(base, atoms, n, f"strong_{n}t")
        print()

    t2 = results[2]
    print(f"{'tiles':>5} {'time_s':>8} {'speedup_vs_2t':>14} {'efficiency_vs_2t':>16}")
    for n in (2, 6, 12):
        t = results[n]
        spd = t2 / t if t > 0 else 0.0
        eff = spd / (n / 2)
        print(f"{n:>5} {t:>8.3f} {spd:>13.2f}x {eff:>15.1%}")


def benchmark_weak(base):
    """Weak scaling: ~12k atoms per tile.
    1 tile = plain MACECalculator (DistMLIP needs n_part>=2)."""
    print()
    print("=" * 60)
    print("(B) WEAK SCALING -- ~12k atoms/tile (atoms/sec/tile)")
    print("=" * 60)

    # Cubic supercell sized to give ~12k atoms/tile, satisfying wall constraint.
    # Atom count = 2 * sc^3.  Wall side = 3.5 * sc.  Need wall/n_part > 12 A.
    sc_map = {
        1: 19,   # 13718 atoms,  cell 66.5 A
        2: 24,   # 27648 atoms,  cell 84 A / 2 = 42 A
        6: 35,   # 85750 atoms,  cell 122.5 A / 6 = 20.4 A
        12: 42,  # 148176 atoms, cell 147 A / 12 = 12.25 A
    }

    results = {}
    for n, sc in sc_map.items():
        print(f"---- {n} tile(s)  supercell=({sc},{sc},{sc})")
        atoms = build_cell((sc, sc, sc))
        n_atoms = len(atoms)
        print(f"atoms: {n_atoms}  per_tile: {n_atoms // n}")

        if n == 1:
            from mace.calculators import mace_mp
            calc = mace_mp(model="small", device="xpu")
            t = time_forward(atoms, calc, f"weak_{n}t")
            del calc
        else:
            t = run_dist(base, atoms, n, f"weak_{n}t")

        results[n] = (n_atoms, t)
        print()

    print(f"{'tiles':>5} {'atoms':>8} {'time_s':>8} {'atoms/s':>12} {'atoms/s/tile':>14}")
    for n in (1, 2, 6, 12):
        n_atoms, t = results[n]
        thr = n_atoms / t
        per_tile = thr / n
        print(f"{n:>5} {n_atoms:>8} {t:>8.3f} {thr:>12.0f} {per_tile:>14.0f}")


def main():
    import torch
    from mace.calculators import mace_mp

    if torch.xpu.device_count() < 12:
        print(f"FAIL: need 12 tiles, got {torch.xpu.device_count()}")
        return 1

    base = mace_mp(model="small", device="cpu")

    benchmark_strong(base)
    benchmark_weak(base)

    return 0


if __name__ == "__main__":
    sys.exit(main())
