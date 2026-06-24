"""Test 4: two tiles xpu.

enable_distributed_mode([0, 1]) on a larger cell. Forward must succeed
and forces must match a re-run on [0] within partition reduction noise.
This is the critical Path A check: cross-tile L0 transfer in
Distributed.aggregate / distribute_node_features.
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

    if torch.xpu.device_count() < 2:
        print(f"FAIL: need >= 2 tiles, got {torch.xpu.device_count()}")
        return 1

    atoms = build_cell()
    print(f"atoms: {len(atoms)}")

    base = mace_mp(model="small", device="cpu")

    # 1 tile reference
    c1 = MACECalculator_Dist.from_existing(base)
    c1.enable_distributed_mode([0])
    atoms.calc = c1
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces().copy()

    # 2 tile
    c2 = MACECalculator_Dist.from_existing(base)
    c2.enable_distributed_mode([0, 1])
    atoms.calc = c2
    e2 = atoms.get_potential_energy()
    f2 = atoms.get_forces().copy()

    de = abs(e2 - e1)
    df = np.abs(f2 - f1).max()
    print(f"E_1tile: {e1:.6f}")
    print(f"E_2tile: {e2:.6f}")
    print(f"|dE|       : {de:.3e}")
    print(f"|dF|max    : {df:.3e}")

    if de > 1e-3 or df > 1e-3:
        print("FAIL: 2-tile diverges from 1-tile beyond tol")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
