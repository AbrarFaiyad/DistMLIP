"""Test 2: CPU baseline.

Plain MACECalculator (no DistMLIP) on small Li-Mn cell. Sanity that the
venv runs MACE forward at all before touching xpu. Saves baseline tensors
to disk for tests 3/4/5 to compare against.
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
    # Same supercell as test 3 (must be large enough for n_partitions=2 wall constraint)
    struct.make_supercell((8, 8, 8))
    return AseAtomsAdaptor().get_atoms(struct)


def main():
    from mace.calculators import mace_mp

    atoms = build_cell()
    print(f"atoms: {len(atoms)}")

    calc = mace_mp(model="small", device="cpu")
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"E_cpu  : {energy:.6f}")
    print(f"|F|max : {np.abs(forces).max():.6f}")
    print(f"|F|mean: {np.abs(forces).mean():.6f}")

    out = Path(__file__).parent / "_baseline_cpu.npz"
    np.savez(out, energy=energy, forces=forces, n_atoms=len(atoms))
    print(f"baseline saved -> {out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
