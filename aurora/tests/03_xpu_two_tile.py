"""Test 3: minimum supported partition count = 2 tiles on xpu.

DistMLIP C kernel hard-rejects num_partitions=1 (see subgraph_creation_utils.c:49).
Two-tile compare against test 2 CPU baseline. Force tol relaxed because partition
edge reductions order differently than single-process CPU forward.
"""
import sys
from pathlib import Path

import numpy as np


def build_cell():
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(3.5), ["Li", "Mn"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    struct.perturb(0.05, seed=42)
    struct.make_supercell((3, 3, 3))
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

    calc = mace_mp(model="small", device="cpu")
    dist_calc = MACECalculator_Dist.from_existing(calc)
    dist_calc.enable_distributed_mode([0, 1])

    atoms.calc = dist_calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"E_xpu2 : {energy:.6f}")
    print(f"|F|max : {np.abs(forces).max():.6f}")

    ref_path = Path(__file__).parent / "_baseline_cpu.npz"
    if not ref_path.exists():
        print(f"WARN: missing {ref_path}, run test 02 first")
        return 0

    ref = np.load(ref_path)
    de = abs(energy - float(ref["energy"]))
    df = np.abs(forces - ref["forces"]).max()
    print(f"|dE|       : {de:.3e}")
    print(f"|dF|max    : {df:.3e}")

    tol_E = 5e-3
    tol_F = 5e-3
    if not (np.isfinite(energy) and np.isfinite(forces).all()):
        print("FAIL: non-finite outputs")
        return 1
    if de > tol_E or df > tol_F:
        print(f"FAIL: tol exceeded (E_tol={tol_E}, F_tol={tol_F})")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
