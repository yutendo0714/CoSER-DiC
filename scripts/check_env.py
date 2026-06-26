from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path


REQUIRED = [
    "torch",
    "torchvision",
    "compressai",
    "wandb",
    "numpy",
    "PIL",
    "pytorch_msssim",
    "lpips",
    "torch_fidelity",
    "yaml",
]


def module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    print(f"python={sys.version.split()[0]} executable={sys.executable}")
    print(f"platform={platform.platform()}")
    print(f"workspace={Path.cwd()}")

    for name in REQUIRED:
        try:
            print(f"{name}={module_version(name)}")
        except Exception as exc:
            print(f"{name}=MISSING ({type(exc).__name__}: {exc})")
            raise

    import torch

    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.version.cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"cuda.device_count={torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            mem_gb = props.total_memory / 1024**3
            print(f"cuda:{idx} name={props.name} memory_gb={mem_gb:.2f}")

    try:
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
        print("nvidia-smi:")
        print(smi.strip())
    except Exception as exc:
        print(f"nvidia-smi unavailable: {exc}")


if __name__ == "__main__":
    main()

