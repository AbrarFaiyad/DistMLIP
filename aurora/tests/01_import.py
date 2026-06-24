"""Test 1: imports + tile visibility.

Goal:
  - torch.xpu.is_available() == True on compute node
  - torch.xpu.device_count() == 12 (1 node, 6 GPUs * 2 tiles, FLAT hierarchy)
  - DistMLIP + C ext + MACE wrapper all import
"""
import sys


def main():
    import torch
    import DistMLIP
    from DistMLIP.distributed.subgraph_creation_fast import get_subgraphs_fast
    from DistMLIP.implementations.mace import MACECalculator_Dist
    from DistMLIP.implementations.mace.models import ScaleShiftMACE_Dist
    from mace.calculators import mace_mp  # noqa: F401

    print("torch       :", torch.__version__)
    print("has_xpu     :", hasattr(torch, "xpu"))
    print("xpu_avail   :", torch.xpu.is_available())
    n = torch.xpu.device_count()
    print("device_count:", n)
    if n != 12:
        print(f"WARN: expected 12 tiles (1 Aurora node, FLAT hierarchy), got {n}")
        print("      check ZE_FLAT_DEVICE_HIERARCHY=FLAT, no ZE_AFFINITY_MASK set")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
