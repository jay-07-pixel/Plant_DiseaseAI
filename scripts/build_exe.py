#!/usr/bin/env python3
"""Build PlantDiseaseAI Windows desktop executable with PyInstaller."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_NAME = "PlantDiseaseAI"
ENTRY = PROJECT_ROOT / "scripts" / "run_app.py"
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"

BUNDLE_PATHS = [
    ("configs", "configs"),
    ("resources/translations", "resources/translations"),
    ("weights/grape/best_model.pth", "weights/grape"),
    ("weights/tomato/best_model.pth", "weights/tomato"),
    ("datasets/grape/reports/class_mapping.json", "datasets/grape/reports"),
    ("datasets/tomato/reports/class_mapping.json", "datasets/tomato/reports"),
]

# Smaller assets only — large weight files are copied after PyInstaller finishes.
PYINSTALLER_DATAS = [
    ("configs", "configs"),
    ("resources/translations", "resources/translations"),
    ("datasets/grape/reports/class_mapping.json", "datasets/grape/reports"),
    ("datasets/tomato/reports/class_mapping.json", "datasets/tomato/reports"),
]


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _validate_assets() -> None:
    missing = [src for src, _ in BUNDLE_PATHS if not (PROJECT_ROOT / src).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files required for the desktop build:\n" + "\n".join(f"  - {item}" for item in missing)
        )


def _copy_runtime_assets(target_root: Path) -> None:
    for src, dest in BUNDLE_PATHS:
        source = PROJECT_ROOT / src
        if source.is_dir():
            destination = target_root / dest
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
            continue

        target_file = target_root / dest / source.name
        dest_parent = target_file.parent
        if dest_parent.exists() and dest_parent.is_file():
            dest_parent.unlink()
        dest_parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_file)


def _desktop_dir() -> Path:
    one_drive = Path.home() / "OneDrive" / "Desktop"
    if one_drive.exists():
        return one_drive
    return Path.home() / "Desktop"


def deploy_to_desktop(app_dir: Path) -> Path:
    """Copy the built application folder to the user's Desktop."""
    target = _desktop_dir() / DIST_NAME
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(app_dir, target)
    return target / f"{DIST_NAME}.exe"


def _pyinstaller_command() -> list[str]:
    add_data: list[str] = []
    for src, dest in PYINSTALLER_DATAS:
        source = PROJECT_ROOT / src
        add_data.extend(["--add-data", f"{source}{os.pathsep}{dest}"])

    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        DIST_NAME,
        "--onedir",
        "--windowed",
        "--paths",
        str(PROJECT_ROOT),
        *add_data,
        "--hidden-import",
        "PySide6.QtCore",
        "--hidden-import",
        "PySide6.QtGui",
        "--hidden-import",
        "PySide6.QtWidgets",
        "--hidden-import",
        "torch",
        "--hidden-import",
        "torchvision",
        "--hidden-import",
        "torchvision.models",
        "--hidden-import",
        "cv2",
        "--hidden-import",
        "albumentations",
        "--hidden-import",
        "groq",
        "--hidden-import",
        "dotenv",
        "--hidden-import",
        "yaml",
        "--collect-binaries",
        "PySide6",
        "--collect-data",
        "PySide6",
        "--exclude-module",
        "tensorboard",
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "onnx",
        "--exclude-module",
        "onnxruntime",
        "--exclude-module",
        "pandas",
        "--exclude-module",
        "seaborn",
        "--exclude-module",
        "IPython",
        "--exclude-module",
        "tkinter",
        "--exclude-module",
        "tensorflow",
        "--exclude-module",
        "sklearn",
        "--exclude-module",
        "scipy",
        "--exclude-module",
        "torchaudio",
        "--exclude-module",
        "h5py",
        str(ENTRY),
    ]


def build(*, clean: bool = True, deploy_desktop: bool = True) -> Path:
    _validate_assets()

    if clean:
        for folder in (BUILD_DIR, DIST_DIR):
            if folder.exists():
                shutil.rmtree(folder)
        spec_file = PROJECT_ROOT / f"{DIST_NAME}.spec"
        if spec_file.exists():
            spec_file.unlink()

    print("Running PyInstaller (this may take 5-15 minutes)...")
    subprocess.run(_pyinstaller_command(), cwd=PROJECT_ROOT, check=True)

    app_dir = DIST_DIR / DIST_NAME
    _copy_runtime_assets(app_dir)

    exe_path = app_dir / f"{DIST_NAME}.exe"
    if not exe_path.exists():
        raise FileNotFoundError(f"Expected executable not found: {exe_path}")

    print()
    print("Build complete.")
    print(f"  Project EXE: {exe_path}")
    print(f"  EXE size:    {_format_size(exe_path.stat().st_size)}")
    print(f"  Folder size: {_format_size(_dir_size(app_dir))}")

    if deploy_desktop:
        desktop_exe = deploy_to_desktop(app_dir)
        desktop_folder = desktop_exe.parent
        print()
        print("Copied to Desktop.")
        print(f"  Desktop EXE: {desktop_exe}")
        print(f"  EXE size:    {_format_size(desktop_exe.stat().st_size)}")
        print(f"  Folder size: {_format_size(_dir_size(desktop_folder))}")
        print(f"  Folder path: {desktop_folder}")

    return exe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PlantDiseaseAI desktop executable")
    parser.add_argument("--no-clean", action="store_true", help="Keep previous build artifacts")
    parser.add_argument("--no-desktop", action="store_true", help="Skip copying build output to Desktop")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        build(clean=not args.no_clean, deploy_desktop=not args.no_desktop)
    except subprocess.CalledProcessError as exc:
        print(f"PyInstaller failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
