"""Test 3: single tile xpu.

enable_distributed_mode([0]). Compare against test 2's CPU baseline.
Force tol 1e-4 eV/A (fp64 should be tighter, but accept noise from
different reduction order on xpu).
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

    if not torch.xpu.is_available():
        print("FAIL: torch.xpu not available")
        return 1

    atoms = build_cell()
    calc = mace_mp(model="small", device="cpu")
    dist_calc = MACECalculator_Dist.from_existing(calc)
    dist_calc.enable_distributed_mode([0])

    atoms.calc = dist_calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"E_xpu  : {energy:.6f}")
    print(f"|F|max : {np.abs(forces).max():.6f}")

    baseline_path = Path(__file__).parent / "_baseline_cpu.npz"
    if not baseline_path.exists():
        print(f"WARN: no baseline at {baseline_path}; run test 02 first")
        return 0

    ref = np.load(baseline_path)
    de = abs(energy - float(ref["energy"]))
    df = np.abs(forces - ref["forces"]).max()
    print(f"|dE|       : {de:.3e}")
    print(f"|dF|max    : {df:.3e}")

    tol_E = 1e-3
    tol_F = 1e-3
    if de > tol_E or df > tol_F:
        print(f"FAIL: tol exceeded (E_tol={tol_E}, F_tol={tol_F})")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
