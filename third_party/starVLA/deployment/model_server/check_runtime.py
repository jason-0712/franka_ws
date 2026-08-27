#!/usr/bin/env python3
"""Fail-fast validation for the vendored StarVLA GPU inference runtime."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Dependency:
    module: str
    package: str


REQUIRED_DEPENDENCIES = (
    Dependency("numpy", "numpy"),
    Dependency("torch", "torch (CUDA build)"),
    Dependency("torchvision", "torchvision (matching the torch build)"),
    Dependency("transformers", "transformers"),
    Dependency("accelerate", "accelerate"),
    Dependency("omegaconf", "omegaconf"),
    Dependency("PIL", "pillow"),
    Dependency("einops", "einops"),
    Dependency("pydantic", "pydantic"),
    Dependency("numpydantic", "numpydantic"),
    Dependency("albumentations", "albumentations"),
    Dependency("cv2", "opencv-python-headless"),
    Dependency("pytorch3d", "pipablepytorch3d"),
    Dependency("av", "av"),
    Dependency("pandas", "pandas"),
    Dependency("qwen_vl_utils", "qwen-vl-utils"),
    Dependency("diffusers", "diffusers"),
    Dependency("rich", "rich"),
    Dependency("websockets", "websockets"),
    Dependency("msgpack", "msgpack"),
)


def inspect_runtime(require_cuda: bool = True) -> tuple[list[str], dict[str, Any]]:
    """Return fatal issues and successfully imported modules.

    This module intentionally depends only on the standard library, allowing it
    to explain an incomplete environment instead of failing on the first import.
    """

    issues: list[str] = []
    imported: dict[str, Any] = {}

    if sys.version_info[:2] != (3, 10):
        issues.append(
            "Python version is "
            f"{platform.python_version()}; this checkpoint runtime requires Python 3.10.x"
        )

    for dependency in REQUIRED_DEPENDENCIES:
        try:
            imported[dependency.module] = importlib.import_module(dependency.module)
        except Exception as exc:  # Import incompatibilities matter as much as missing modules.
            issues.append(
                f"{dependency.package}: cannot import {dependency.module!r} "
                f"({type(exc).__name__}: {exc})"
            )

    torch = imported.get("torch")
    if require_cuda and torch is not None:
        try:
            if not torch.cuda.is_available():
                issues.append("PyTorch is installed, but torch.cuda.is_available() is False")
            elif torch.cuda.device_count() < 1:
                issues.append("PyTorch reports no CUDA devices")
        except Exception as exc:
            issues.append(f"CUDA validation failed ({type(exc).__name__}: {exc})")

    return issues, imported


def require_runtime(require_cuda: bool = True) -> None:
    """Exit with an actionable message when the inference runtime is unusable."""

    issues, imported = inspect_runtime(require_cuda=require_cuda)
    if issues:
        details = "\n".join(f"  - {issue}" for issue in issues)
        raise SystemExit(
            "STARVLA_RUNTIME_CHECK=FAIL\n"
            f"Executable: {sys.executable}\n"
            f"Problems:\n{details}\n\n"
            "Run the policy server on the GPU host in the existing Python 3.10 "
            "StarVLA CUDA environment. `pip install -e .` registers this source "
            "package; it does not install a compatible CUDA/PyTorch stack. See "
            "third_party/starVLA/README.md."
        )

    torch = imported["torch"]
    device_name = torch.cuda.get_device_name(0) if require_cuda else "not checked"
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device 0: {device_name}")
    print("STARVLA_RUNTIME_CHECK=PASS")


if __name__ == "__main__":
    require_runtime(require_cuda=True)
