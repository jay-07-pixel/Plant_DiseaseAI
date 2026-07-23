#!/usr/bin/env python3
"""Export trained model to TorchScript and ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exports.exporter import ModelExporter
from utils.config import load_config
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export model to deployment formats",
    )
    parser.add_argument("--crop", type=str, default="grape", help="Crop identifier")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: from config)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    logger = setup_logging(config, log_name="export", log_subdir="export")

    weights_path = Path(args.weights) if args.weights else None
    if weights_path and not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    logger.info("PlantDiseaseAI Model Export")
    exporter = ModelExporter(config, weights_path=weights_path)
    result = exporter.export()

    if result.success:
        logger.info("Export completed successfully")
        for path in result.exported_files:
            logger.info("  %s", path)
        return 0

    for error in result.errors:
        logger.error("Error: %s", error)
    return 1 if not result.exported_files else 0


if __name__ == "__main__":
    sys.exit(main())
