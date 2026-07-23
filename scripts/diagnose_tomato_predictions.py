#!/usr/bin/env python3
"""Diagnose tomato prediction bias across all classes."""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sanitize() -> None:
    vendor = (PROJECT_ROOT / ".vendor").resolve()
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != vendor]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


_sanitize()

import csv

from desktop_app.services.model_manager import ModelManager
from inference.predictor import Predictor
from utils.config import load_config


def per_class_samples(n: int = 5) -> None:
    config = load_config(crop="tomato", project_root=PROJECT_ROOT)
    pred = Predictor(config)
    test_root = PROJECT_ROOT / "datasets/tomato/split/test"

    print("=" * 70)
    print("TOMATO PER-CLASS PREDICTION TEST")
    print("weights:", pred.weights_path)
    print("num_classes:", len(pred.class_names))
    print("=" * 70)

    folder_to_display = {c.folder_name: c.display_name for c in config.class_configs}
    total = 0
    correct = 0

    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir():
            continue
        expected = folder_to_display.get(class_dir.name, class_dir.name)
        imgs = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        picks = random.sample(imgs, min(n, len(imgs)))
        class_ok = 0
        for img in picks:
            r = pred.predict(img)
            ok = r.predicted_class == expected
            total += 1
            correct += int(ok)
            class_ok += int(ok)
            status = "OK" if ok else "MISS"
            top3 = ", ".join(f"{p.class_name}({p.confidence:.2f})" for p in r.top_predictions)
            print(f"{status:4} GT={expected:35} -> {r.predicted_class:35} conf={r.confidence:.3f}")
            if not ok:
                print(f"      top3: {top3}")
        print(f"      class accuracy: {class_ok}/{len(picks)}")
        print()

    print(f"Overall sample accuracy: {correct}/{total} ({100*correct/total:.1f}%)")


def prediction_distribution() -> None:
    csv_path = PROJECT_ROOT / "reports/tomato_test_predictions.csv"
    counts = Counter()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["predicted_class"]] += 1

    print("=" * 70)
    print("FULL TEST SET PREDICTION DISTRIBUTION (2713 images)")
    print("=" * 70)
    for name, count in counts.most_common():
        print(f"  {name:45} {count:5} ({100*count/2713:.1f}%)")


def grape_vs_tomato_same_image() -> None:
    """Ensure grape and tomato models give different outputs on tomato leaf."""
    sample = next(
        (PROJECT_ROOT / "datasets/tomato/split/test/Tomato___healthy").glob("*.jpg")
    )
    mgr = ModelManager(PROJECT_ROOT)
    grape = mgr.get_or_load("grape")
    tomato = mgr.get_or_load("tomato")
    rg = grape.predict(sample)
    rt = tomato.predict(sample)
    print("=" * 70)
    print("GRAPE vs TOMATO MODEL on same tomato healthy image")
    print("=" * 70)
    print(f"  grape weights: {grape.weights_path}")
    print(f"  tomato weights: {tomato.weights_path}")
    print(f"  grape pred: {rg.predicted_class} ({rg.confidence:.3f})")
    print(f"  tomato pred: {rt.predicted_class} ({rt.confidence:.3f})")
    same_model = str(grape.weights_path) == str(tomato.weights_path)
    print(f"  same weights file: {same_model}")


if __name__ == "__main__":
    random.seed(42)
    per_class_samples(n=3)
    print()
    prediction_distribution()
    print()
    grape_vs_tomato_same_image()
