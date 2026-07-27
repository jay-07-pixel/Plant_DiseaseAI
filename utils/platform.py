"""Platform detection helpers."""

from __future__ import annotations

import os
import platform
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def is_linux() -> bool:
    return sys.platform.startswith("linux")


@lru_cache(maxsize=1)
def is_arm_cpu() -> bool:
    machine = platform.machine().lower()
    return machine in {"aarch64", "arm64", "armv7l", "armv8", "armv6l"}


@lru_cache(maxsize=1)
def is_raspberry_pi() -> bool:
    if os.getenv("PLANT_DISEASE_FORCE_PI", "").lower() in {"1", "true", "yes"}:
        return True
    if not is_linux() or not is_arm_cpu():
        return False

    model_paths = (
        Path("/proc/device-tree/model"),
        Path("/sys/firmware/devicetree/base/model"),
    )
    for path in model_paths:
        if path.exists():
            try:
                model = path.read_text(encoding="utf-8", errors="ignore").lower()
                if "raspberry pi" in model:
                    return True
            except OSError:
                continue

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            if "raspberry pi" in cpuinfo.read_text(encoding="utf-8", errors="ignore").lower():
                return True
        except OSError:
            pass

    return False


def append_pi_system_packages() -> None:
    """Expose apt-only Pi packages (e.g. picamera2) without overriding venv deps.

    Do not set PYTHONPATH to /usr/lib/python3/dist-packages — that prepends old
    system packages like typing_extensions and breaks pydantic/albumentations.
    """
    if not is_raspberry_pi():
        return

    dist_packages = Path("/usr/lib/python3/dist-packages")
    if not dist_packages.is_dir():
        return

    dist_str = str(dist_packages)
    if dist_str not in sys.path:
        sys.path.append(dist_str)
