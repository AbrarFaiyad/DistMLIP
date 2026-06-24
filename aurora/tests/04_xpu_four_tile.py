"""Test 4: 4 tiles on xpu (larger cell).

Validates cross-tile L0 transfer in Distributed.aggregate / distribute_node_features
beyond a single pair. Compares 4-tile result against 2-tile on same cell —
partition boundaries differ so allow small reduction noise.
"""
import sys

import numpy as np


def build_cell(supercell=(8, 8, 8)):
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

    if torch.xpu.device_count() < 4:
        print(f"FAIL: need >= 4 tiles, got {torch.xpu.device_count()}")
        return 1

    atoms = build_cell()
    print(f"atoms: {len(atoms)}")

    base = mace_mp(model="small", device="cpu")

    c2 = MACECalculator_Dist.from_existing(base)
    c2.enable_distributed_mode([0, 1])
    atoms.calc = c2
    e2 = atoms.get_potential_energy()
    f2 = atoms.get_forces().copy()

    c4 = MACECalculator_Dist.from_existing(base)
    c4.enable_distributed_mode([0, 1, 2, 3])
    atoms.calc = c4
    e4 = atoms.get_potential_energy()
    f4 = atoms.get_forces().copy()

    de = abs(e4 - e2)
    df = np.abs(f4 - f2).max()
    print(f"E_2tile: {e2:.6f}")
    print(f"E_4tile: {e4:.6f}")
    print(f"|dE|       : {de:.3e}")
    print(f"|dF|max    : {df:.3e}")

    if not (np.isfinite(e4) and np.isfinite(f4).all()):
        print("FAIL: non-finite")
        return 1
    if de > 5e-3 or df > 5e-3:
        print("FAIL: 4-tile diverges from 2-tile beyond tol")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
