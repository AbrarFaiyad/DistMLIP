"""Aurora XPU patches for DistMLIP MACE path.

Three idempotent edits keyed by `# AURORA-PATCHED` marker:

  1. DistMLIP/implementations/mace/models.py:
       `enable_distributed_mode` hardcodes "cuda:N". Replace with
       device-agnostic resolver (xpu first if torch.xpu.is_available()).

  2. DistMLIP/implementations/mace/mace.py:
       `enable_distributed_mode` calls mace.tools.torch_tools.init_device
       which only accepts bare "xpu" (not "xpu:N"). Bypass for xpu and
       call torch.xpu.set_device explicitly.

  3. DistMLIP/__init__.py:
       fp16 guard checks torch.cuda only. Also allow torch.xpu.

Usage:
    python aurora/patch_distmlip_for_aurora.py [REPO_ROOT]

REPO_ROOT defaults to the script's grandparent (the DistMLIP/ checkout).
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
import sys

MARKER = "# AURORA-PATCHED"


def _header(note: str) -> str:
    return (
        f"{MARKER} {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"{MARKER} {note}\n"
    )


def patch_models_py(path: Path) -> bool:
    text = path.read_text()
    if MARKER in text:
        print(f"  [skip] already patched: {path}")
        return False

    old = (
        "    def enable_distributed_mode(self, gpus):\n"
        "        self.gpus = []\n"
        "\n"
        "        for gpu_index in gpus:\n"
        "            if gpu_index == \"cpu\":\n"
        "                self.gpus.append(\"cpu\")\n"
        "            else:\n"
        "                self.gpus.append(\"cuda:\" + str(gpu_index))"
    )
    new = (
        "    def enable_distributed_mode(self, gpus):\n"
        "        self.gpus = []\n"
        "\n"
        "        _use_xpu = hasattr(torch, \"xpu\") and torch.xpu.is_available()\n"
        "        for gpu_index in gpus:\n"
        "            if gpu_index == \"cpu\":\n"
        "                self.gpus.append(\"cpu\")\n"
        "            elif _use_xpu:\n"
        "                self.gpus.append(\"xpu:\" + str(gpu_index))\n"
        "            else:\n"
        "                self.gpus.append(\"cuda:\" + str(gpu_index))"
    )
    if old not in text:
        print(f"  [warn] target block not found in {path}; skipping")
        return False
    new_text = _header("device-agnostic gpu list builder (xpu/cuda)") + text.replace(old, new)
    path.write_text(new_text)
    print(f"  [WRITE] {path}")
    return True


def patch_mace_py(path: Path) -> bool:
    text = path.read_text()
    if MARKER in text:
        print(f"  [skip] already patched: {path}")
        return False

    old = (
        "    def enable_distributed_mode(self, gpus):\n"
        "        for model in self.models:\n"
        "            model.to(gpus[0])\n"
        "            \n"
        "            model.enable_distributed_mode(gpus)\n"
        "\n"
        "        self.device = torch_tools.init_device(self.models[0].gpus[0])\n"
        "        self.gpus = gpus\n"
        "\n"
        "        self.dist_enabled = True"
    )
    new = (
        "    def enable_distributed_mode(self, gpus):\n"
        "        for model in self.models:\n"
        "            model.to(gpus[0])\n"
        "            \n"
        "            model.enable_distributed_mode(gpus)\n"
        "\n"
        "        # torch_tools.init_device rejects 'xpu:N' (only bare 'xpu').\n"
        "        # Bypass: set xpu device explicitly, build torch.device directly.\n"
        "        dev_str = self.models[0].gpus[0]\n"
        "        if isinstance(dev_str, str) and dev_str.startswith(\"xpu\"):\n"
        "            if \":\" in dev_str:\n"
        "                torch.xpu.set_device(int(dev_str.split(\":\")[1]))\n"
        "            self.device = torch.device(dev_str)\n"
        "        else:\n"
        "            self.device = torch_tools.init_device(dev_str)\n"
        "        self.gpus = gpus\n"
        "\n"
        "        self.dist_enabled = True"
    )
    if old not in text:
        print(f"  [warn] target block not found in {path}; skipping")
        return False
    new_text = _header("xpu init_device bypass") + text.replace(old, new)
    path.write_text(new_text)
    print(f"  [WRITE] {path}")
    return True


def patch_init_py(path: Path) -> bool:
    text = path.read_text()
    if MARKER in text:
        print(f"  [skip] already patched: {path}")
        return False

    old = (
        "    if type_ == \"float\" and size == 16 and not torch.cuda.is_available():\n"
        "        raise Exception(\n"
        "            \"torch.float16 is not supported because addmm_impl_cpu_ is not implemented\"\n"
        "            \" for this floating precision, please use size = 32, 64 or using 'cuda' instead !!\"\n"
        "        )"
    )
    new = (
        "    if type_ == \"float\" and size == 16:\n"
        "        _has_cuda = torch.cuda.is_available()\n"
        "        _has_xpu = hasattr(torch, \"xpu\") and torch.xpu.is_available()\n"
        "        if not (_has_cuda or _has_xpu):\n"
        "            raise Exception(\n"
        "                \"torch.float16 is not supported on CPU; use size=32 or 64,\"\n"
        "                \" or run on a cuda/xpu accelerator.\"\n"
        "            )"
    )
    if old not in text:
        print(f"  [warn] fp16 guard block not found in {path}; skipping")
        return False
    new_text = _header("fp16 guard accepts xpu") + text.replace(old, new)
    path.write_text(new_text)
    print(f"  [WRITE] {path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", nargs="?", default=None,
                    help="DistMLIP repo root (default: script's grandparent)")
    args = ap.parse_args()

    if args.repo_root:
        root = Path(args.repo_root).resolve()
    else:
        root = Path(__file__).resolve().parent.parent

    targets = [
        ("models.py", root / "DistMLIP" / "implementations" / "mace" / "models.py", patch_models_py),
        ("mace.py", root / "DistMLIP" / "implementations" / "mace" / "mace.py", patch_mace_py),
        ("__init__.py", root / "DistMLIP" / "__init__.py", patch_init_py),
    ]

    any_changed = False
    for name, path, fn in targets:
        if not path.exists():
            print(f"==> MISSING {path}")
            continue
        print(f"==> patching {name}")
        if fn(path):
            any_changed = True

    print("DONE" + ("" if any_changed else " (no changes)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
