"""Test 5: full node, 12 tiles xpu.

Large cell scaled to stress all 12 PVC tiles. Success = forward returns
finite energies/forces in reasonable wall time.
"""
import sys
import time

import numpy as np


def build_cell(supercell=(42, 42, 42)):
    # 3.5 A * 42 = 147 A per side; / 12 partitions = 12.25 A wall (> 2*cutoff=12 A)
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(3.5), ["Li", "Mn"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    struct.perturb(0.05, seed=42)
    struct.make_supercell(supercell)
    return AseAtomsAdaptor().get_atoms(struct)


def main():
    import torch
    from mace.calculators import mace_mp
    from DistMLIP.implementations.mace import MACECalculator_Dist

    n_tiles = torch.xpu.device_count()
    if n_tiles < 12:
        print(f"FAIL: need 12 tiles, got {n_tiles}")
        return 1

    atoms = build_cell()
    print(f"atoms: {len(atoms)}")

    base = mace_mp(model="small", device="cpu")
    calc = MACECalculator_Dist.from_existing(base)
    calc.enable_distributed_mode(list(range(12)))

    atoms.calc = calc
    t0 = time.perf_counter()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    dt = time.perf_counter() - t0

    print(f"E      : {energy:.6f}")
    print(f"|F|max : {np.abs(forces).max():.6f}")
    print(f"forward: {dt:.2f} s")

    if not (np.isfinite(energy) and np.isfinite(forces).all()):
        print("FAIL: non-finite outputs")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
