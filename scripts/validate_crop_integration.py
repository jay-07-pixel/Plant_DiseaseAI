#!/usr/bin/env python3
"""Validate grape and tomato inference integration without launching the GUI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sanitize_import_path() -> None:
    vendor_root = (PROJECT_ROOT / ".vendor").resolve()
    sys.path = [
        entry for entry in sys.path if entry and Path(entry).resolve() != vendor_root
    ]


_sanitize_import_path()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from desktop_app.services.model_manager import ModelManager
from utils.config import load_config


SAMPLE_IMAGES = {
    "grape": PROJECT_ROOT
    / "datasets/grape/test/Black_Rot/003d09ef-e16c-4e8a-badf-847d46cb3dc0___FAM_B.Rot 3184.JPG",
    "tomato": PROJECT_ROOT
    / "datasets/tomato/split/test/Tomato___healthy/0326b4b6-0f25-47af-bfd9-d8fec314a4f5___RS_HL 0621.jpg",
}


def validate_crop(manager: ModelManager, crop: str) -> dict:
    image_path = SAMPLE_IMAGES[crop]
    if not image_path.exists():
        raise FileNotFoundError(f"Sample image missing for {crop}: {image_path}")

    config = manager.load_config(crop)
    service = manager.get_or_load(crop)
    result = service.predict(image_path)

    checks = {
        "model_loaded": service.is_loaded,
        "num_classes": len(config.class_names),
        "top3_count": len(result.top_predictions),
        "confidence_range_ok": 0.0 <= result.confidence <= 1.0,
        "gradcam_overlay": bool(result.overlay_path and Path(result.overlay_path).exists()),
        "gradcam_heatmap": bool(result.heatmap_path and Path(result.heatmap_path).exists()),
        "predicted_class": result.predicted_class,
        "confidence": round(result.confidence, 4),
    }
    return checks


def main() -> int:
    manager = ModelManager(PROJECT_ROOT)
    results: dict[str, dict] = {}

    for crop in ("grape", "tomato"):
        results[crop] = validate_crop(manager, crop)

    # Cache check: second load should reuse cached service
    grape_first = manager.get_or_load("grape")
    grape_second = manager.get_or_load("grape")
    cache_ok = grape_first is grape_second

    print("=== Crop Integration Validation ===")
    for crop, checks in results.items():
        print(f"\n{crop.upper()}:")
        for key, value in checks.items():
            print(f"  {key}: {value}")

    print(f"\nmodel_cache_reuse: {cache_ok}")

    all_ok = cache_ok and all(
        checks["model_loaded"]
        and checks["top3_count"] == 3
        and checks["confidence_range_ok"]
        and checks["gradcam_overlay"]
        for checks in results.values()
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
